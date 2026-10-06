"""Read a bounded, accessible summary from the active page.

The JavaScript below only inspects the DOM and marks observed elements with
short-lived IDs. It never executes page-provided instructions.
"""

from __future__ import annotations

from playwright.async_api import Page

from fisher.browser.models import GroundedTarget, PerceivedPage
from fisher.models import ElementInfo

_INSPECT_SCRIPT = r"""([prefix, maxElements, maxText]) => {
  const query = 'a[href],button,input:not([type="hidden"]),textarea,select,' +
    '[role="button"],[role="link"],[role="textbox"],[role="checkbox"],' +
    '[role="radio"],[role="combobox"],[role="tab"],[contenteditable="true"]';
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
  };
  const clean = (s, n=100) => String(s || '').replace(/\s+/g, ' ').trim().slice(0, n);
  const roleOf = (el) => {
    if (el.getAttribute('role')) return el.getAttribute('role');
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'combobox';
    if (tag === 'input') {
      const t = (el.type || 'text').toLowerCase();
      if (['button','submit','reset','image'].includes(t)) return 'button';
      if (['checkbox','radio'].includes(t)) return t;
      return 'textbox';
    }
    return 'textbox';
  };
  const nameOf = (el) => {
    const labelledBy = el.getAttribute('aria-labelledby');
    const named = labelledBy && labelledBy.split(/\s+/).map(id =>
      document.getElementById(id)?.textContent || '').join(' ');
    return clean(el.getAttribute('aria-label') || named ||
      (el.labels && Array.from(el.labels).map(x => x.textContent).join(' ')) ||
      el.getAttribute('placeholder') || el.getAttribute('title') ||
      el.getAttribute('alt') || el.innerText ||
      ((el.type === 'button' || el.type === 'submit') ? el.value : ''));
  };
  const nodes = [];
  for (const el of document.querySelectorAll(query)) {
    if (nodes.length >= maxElements) break;
    if (!visible(el)) continue;
    const id = `${prefix}-e${nodes.length + 1}`;
    el.setAttribute('data-fisher-id', id);
    const type = (el.getAttribute('type') ||
      (el.tagName === 'BUTTON' && el.closest('form') ? el.type : '') || '').toLowerCase();
    const sensitive = type === 'password' || type === 'hidden' ||
      /(?:password|cc-|one-time-code)/i.test(el.getAttribute('autocomplete') || '') ||
      /(?:password|passcode|api\s*key|token|cvv|card\s*number|security\s*code|verification\s*code)/i.test(nameOf(el));
    const rawValue = ('value' in el) ? el.value : null;
    nodes.push({
      id, role: roleOf(el), name: nameOf(el), tag: el.tagName.toLowerCase(),
      text: clean(el.innerText || '', 120), visible: true,
      enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
      href: el.href || null,
      value: rawValue == null ? null : sensitive ?
        (rawValue ? '[redacted]' : '[empty]') : clean(rawValue, 120),
      input_type: type || null, sensitive,
      dom_id: clean(el.id, 120), test_id: clean(el.getAttribute('data-testid'), 120),
      name_attr: clean(el.getAttribute('name'), 120)
    });
  }
  const root = document.querySelector('main,[role="main"]') || document.body;
  const text = clean(root?.innerText || '', maxText);
  return {title: clean(document.title, 200), text, nodes,
    scroll_x: Math.round(window.scrollX), scroll_y: Math.round(window.scrollY)};
}"""


async def inspect_page(
    page: Page, generation: int, *, max_elements: int = 100, max_text: int = 7000
) -> PerceivedPage:
    raw = await page.evaluate(_INSPECT_SCRIPT, [f"o{generation}", max_elements, max_text])
    targets: dict[str, GroundedTarget] = {}
    for item in raw["nodes"]:
        info = ElementInfo.model_validate(item)
        targets[info.id] = GroundedTarget(
            info=info,
            dom_id=item.get("dom_id", ""),
            test_id=item.get("test_id", ""),
            name_attr=item.get("name_attr", ""),
        )
    return PerceivedPage(
        title=raw["title"],
        text=raw["text"],
        targets=targets,
        scroll_x=int(raw["scroll_x"]),
        scroll_y=int(raw["scroll_y"]),
    )
