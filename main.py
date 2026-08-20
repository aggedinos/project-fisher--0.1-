"""Entry point — parse CLI args, load env, and run the browser agent.

Backends:  gemini (cloud, API key) | ollama (local, no key)
Modes:     default automatic | --supervised (y/n/s/r) | --replay <session.json>
           | --gui (desktop app)
"""

import argparse
import asyncio
import os
import sys
from asyncio import CancelledError

from dotenv import load_dotenv

import paths  
from actions import Action
from agent import BrowserAgent
from config import config
from llm import build_provider


def build_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser."""
    parser = argparse.ArgumentParser(
        description="Browser automation agent (Gemini Cloud Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Find the top story on Hacker News" --url https://news.ycombinator.com
  python main.py "Research a topic" --thinking-level pro
  python main.py --replay sessions/session_20250515_143022.json
  python main.py --gui
        """,
    )
    parser.add_argument("task", nargs="?", help="Natural-language task")
    parser.add_argument(
        "--url", default="https://duckduckgo.com", metavar="URL",
        help="Starting URL (default: https://duckduckgo.com)",
    )
    parser.add_argument(
        "--thinking-level", choices=["fast", "thinking", "pro"],
        default=config.thinking_level,
        help=f"Gemini Thinking Level: fast, thinking, pro (default: {config.thinking_level})",
    )
    parser.add_argument(
        "--click-mode", choices=["vision", "text"], default=config.click_mode,
        help="Clicking strategy: vision = x,y coordinates (default); text",
    )
    parser.add_argument(
        "--real-chrome", action="store_true",
        help="Launch with the user's real Chrome profile instead of Chromium",
    )
    parser.add_argument(
        "--chrome-profile", default=config.chrome_profile, metavar="NAME",
        help='Chrome profile directory name',
    )
    parser.add_argument(
        "--supervised", action="store_true",
        help="Ask y/n/s/r before every action",
    )
    parser.add_argument(
        "--replay", metavar="FILE",
        help="Replay a saved session JSON visually and exit",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch the desktop GUI instead of the CLI",
    )
    return parser


def cli_approve(action: Action, step: int, screenshot_path: str) -> str:
    """Supervised-mode prompt (Improvement 9). Returns y / n / s / r."""
    print(
        f"\nStep {step}: [{action.action_type.upper()}] — {action.reason} "
        f"(confidence: {action.confidence}%)"
    )
    print(f"Screenshot saved to {screenshot_path}")
    try:
        choice = input("Continue? [y/n/s(skip)/r(replan)]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"
    return choice[:1] if choice else "y"


async def run_cli(args) -> None:
    config.use_real_chrome = args.real_chrome
    config.chrome_profile = args.chrome_profile

    config.thinking_level = args.thinking_level
    gemini_model = config.gemini_model

    try:
        provider = build_provider(
            provider="gemini",
            temperature=config.temperature,
            gemini_model=gemini_model,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Task     : {args.task}")
    print(f"Start    : {args.url}")
    print(f"Level    : {config.thinking_level.upper()} ({provider.model})")
    print(f"Mode     : {'SUPERVISED' if args.supervised else 'automatic'}")
    print(f"Clicking : {args.click_mode}")
    if args.real_chrome:
        print(f"Browser  : real Chrome profile ({config.chrome_profile})")
    else:
        print("Browser  : Chromium (isolated)")
    print()

    agent = BrowserAgent(provider=provider)
    try:
        result = await agent.run(
            task=args.task,
            start_url=args.url,
            approve=cli_approve if args.supervised else None,
            supervised=args.supervised,
            click_mode=args.click_mode,
        )
        print(f"\nResult: {result}")
    except (KeyboardInterrupt, CancelledError):
        print("\n[Interrupted by user]")


def main() -> None:
    load_dotenv(paths.env_file())
    args = build_parser().parse_args()

    if args.gui:
        from webgui import launch

        launch()
        return

    if args.replay:
        from session import replay

        asyncio.run(replay(args.replay))
        return

    if not args.task:
        build_parser().error("a task is required (or pass --gui / --replay)")

    asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
