"""Safety checks for browser destinations and consequential interactions.

The webpage is untrusted. In particular, a model-supplied URL or an anchor's
``href`` must be checked before the browser follows it. This policy is also
used by the request route so redirects cannot quietly reach local services.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from fisher.models import Observation, RiskLevel, ToolCall


class SecurityError(ValueError):
    """A destination or action is outside the browser's allowed scope."""


def safe_url(url: str) -> str:
    """Strip URL credentials, query parameters and fragments for diagnostics."""
    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.hostname:
            return "[invalid URL]"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))[:500]
    except (TypeError, ValueError):
        return "[invalid URL]"


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reason: str


_CONSEQUENTIAL = re.compile(
    r"\b(?:buy|purchase|checkout|pay|order|delete|remove|publish|post|"
    r"send|submit|transfer|confirm|subscribe|save changes|sign out|log out)\b",
    re.IGNORECASE,
)
_ACCOUNT = re.compile(r"\b(?:sign in|log in|account|profile|settings)\b", re.IGNORECASE)
_DOWNLOAD = re.compile(r"\.(?:zip|exe|msi|dmg|pdf|csv|xlsx?|docx?)(?:[?#]|$)", re.IGNORECASE)


class SecurityPolicy:
    """Classify tool calls and restrict navigation to web destinations.

    ``allow_private_network`` is an explicit test/developer override. Ordinary
    runs default to blocking loopback, link-local, and private destinations.
    """

    def __init__(self, *, allow_private_network: bool = False) -> None:
        self.allow_private_network = allow_private_network
        self._dns_cache: dict[str, tuple[float, bool]] = {}

    def validate_url(self, url: str) -> str:
        if not isinstance(url, str) or not url or len(url) > 4096:
            raise SecurityError("URL must be a nonempty web address under 4096 characters")
        if "\\" in url or any(ord(ch) < 32 for ch in url):
            raise SecurityError("URL contains control characters")
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").rstrip(".").lower()
            port = parsed.port  # validates malformed ports
        except ValueError as exc:
            raise SecurityError("Malformed URL") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not host:
            raise SecurityError("Only absolute HTTP and HTTPS URLs are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise SecurityError("Credentials in URLs are not allowed")
        if port is not None and port < 1:
            raise SecurityError("Invalid URL port")
        if not self.allow_private_network and self._is_private_host(host):
            raise SecurityError("Local and private network addresses are blocked")
        return url

    async def validate_url_destination(self, url: str) -> str:
        """Also check resolved IPs when the local resolver knows the host.

        Proxy-only hostnames may not resolve on the client. In that case we
        preserve ordinary browsing but still block private literals and local
        hostnames. This is a best-effort DNS guard, not a network firewall.
        """
        self.validate_url(url)
        if self.allow_private_network:
            return url
        host = urlsplit(url).hostname or ""
        now = time.monotonic()
        cached = self._dns_cache.get(host)
        if cached is not None and now - cached[0] < 15:
            if cached[1]:
                raise SecurityError("Destination resolves to a private network address")
            return url
        try:
            records = await asyncio.wait_for(
                asyncio.to_thread(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM),
                timeout=2.0,
            )
        except (OSError, asyncio.TimeoutError):
            return url
        private = False
        for record in records:
            try:
                address = ipaddress.ip_address(record[4][0].split("%", 1)[0])
            except (IndexError, ValueError, TypeError):
                continue
            if not address.is_global:
                private = True
                break
        self._dns_cache[host] = (now, private)
        if private:
            raise SecurityError("Destination resolves to a private network address")
        return url

    @staticmethod
    def _is_private_host(host: str) -> bool:
        if host in {"localhost", "localhost.localdomain"} or host.endswith(
            (".localhost", ".local", ".internal")
        ):
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # inet_aton recognizes legacy spellings such as 127.1 and 2130706433.
            try:
                address = ipaddress.ip_address(socket.inet_aton(host))
            except (OSError, ValueError):
                return False
        return not address.is_global

    def classify(self, call: ToolCall, observation: Observation | None = None) -> RiskAssessment:
        name = call.name
        args = call.arguments
        if name == "navigate":
            self.validate_url(args.get("url", ""))
            return RiskAssessment(RiskLevel.LOW, "Open a web page")
        if name in {"read_page", "scroll", "switch_tab", "close_tab"}:
            return RiskAssessment(RiskLevel.LOW, "Inspect or move around the browser")
        if name == "click_coordinates":
            return RiskAssessment(
                RiskLevel.HIGH, "The clicked target cannot be identified semantically"
            )
        if name == "press_key":
            key = str(args.get("key", "")).lower()
            if key in {"enter", "numpadenter"}:
                if observation and any(
                    item.input_type == "submit" and _CONSEQUENTIAL.search(item.name)
                    for item in observation.elements
                ):
                    return RiskAssessment(RiskLevel.HIGH, "Enter could submit a consequential form")
                return RiskAssessment(RiskLevel.MEDIUM, "Enter could submit a form")
            return RiskAssessment(RiskLevel.LOW, "Press a browser key")
        if name in {"fill_element", "type_text"}:
            target = self._element(args.get("element_id"), observation)
            if target and (target.sensitive or target.input_type in {"password", "hidden"}):
                return RiskAssessment(RiskLevel.HIGH, "Entering data in a sensitive field")
            return RiskAssessment(RiskLevel.LOW, "Enter text in a visible field")
        if name == "click_element":
            target = self._element(args.get("element_id"), observation)
            if target:
                if target.href:
                    self.validate_url(target.href)
                    if _DOWNLOAD.search(target.href):
                        return RiskAssessment(RiskLevel.MEDIUM, "Download a file")
                label = f"{target.name} {target.text}".strip()
                if _CONSEQUENTIAL.search(label) or target.input_type == "submit":
                    # A search button is not a consequential submission.
                    if not re.search(r"\bsearch\b", label, re.IGNORECASE):
                        return RiskAssessment(
                            RiskLevel.HIGH,
                            f"Clicking {label[:80] or 'submit'} may submit or change data",
                        )
                if _ACCOUNT.search(label):
                    return RiskAssessment(RiskLevel.MEDIUM, "Interact with account controls")
            return RiskAssessment(RiskLevel.LOW, "Click a visible page element")
        return RiskAssessment(RiskLevel.MEDIUM, "Unrecognized browser action")

    @staticmethod
    def _element(element_id: object, observation: Observation | None):
        if observation is None:
            return None
        return next((item for item in observation.elements if item.id == element_id), None)
