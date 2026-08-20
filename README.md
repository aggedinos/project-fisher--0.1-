# project-fisher--0.1-

# Autonomous Browser Agent

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Node Version](https://img.shields.io/badge/node-18%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An AI-powered browser automation agent that can navigate, click, type, and extract data from any website autonomously. Powered by Google Gemini and built with Playwright and React.

## 🌟 Features

- **Autonomous Navigation**: Provide a natural language task and watch the agent navigate, click, and read.
- **Three AI Tiers**: 
  - `fast`: Low latency for simpler tasks
  - `thinking`: Advanced reasoning for complex navigation
  - `pro`: Ultimate capacity for deep multi-step workflows
- **Multiple Providers**: Support for Gemini (cloud), Ollama (local), and NVIDIA (cloud).
- **Desktop GUI**: A beautiful React-based desktop app via PyWebview to monitor and intervene during sessions.
- **CLI Mode**: Run tasks directly from the terminal for automation workflows.
- **Persistent Profiles**: Use your real Chrome profile to maintain logins and session states across runs.
- **Session Replay**: Record sessions and replay them visually for debugging or demonstration.

## 🚀 Quick Start

**Prerequisites:** Python 3.10+, Node.js 18+, and a [Gemini API key](https://aistudio.google.com/apikey).

1. Clone the repository.
2. Run `setup\install.bat` to install all Python and Node.js dependencies.
3. Copy `setup\.env.example` to `.env` in the root folder and add your `GEMINI_API_KEY`.
4. Run `setup\start.bat` to launch the Desktop GUI application.

## 🛠️ Manual Installation

If you prefer to install manually or are not on Windows:

```bash
# 1. Install Python requirements
pip install -r requirements.txt

# 2. Install Playwright browsers
playwright install chromium

# 3. Install Node.js dependencies
npm install

# 4. Build the React frontend
npm run build

# 5. Set up your API key
cp setup/.env.example .env
# Edit .env and add GEMINI_API_KEY=your_api_key_here

# 6. Launch the application
python main.py --gui
```

## 💻 CLI Usage

You can run the agent directly from the command line:

```bash
# Run a simple search task
python main.py "Find the top story on Hacker News" --url https://news.ycombinator.com

# Use the 'pro' thinking level for a complex research task
python main.py "Research the latest advancements in quantum computing" --thinking-level pro

# Replay a previously recorded session
python main.py --replay sessions/session_20250515_143022.json

# Launch the desktop GUI
python main.py --gui
```

### CLI Options

```text
positional arguments:
  task                  Natural-language task

options:
  -h, --help            show this help message and exit
  --url URL             Starting URL (default: https://duckduckgo.com)
  --thinking-level {fast,thinking,pro}
                        Gemini Thinking Level (default: fast)
  --click-mode {vision,text}
                        Clicking strategy (default: vision)
  --real-chrome         Launch with the user's real Chrome profile instead of Chromium
  --chrome-profile NAME Chrome profile directory name
  --supervised          Ask y/n/s/r before every action
  --replay FILE         Replay a saved session JSON visually and exit
  --gui                 Launch the desktop GUI instead of the CLI
```

## 🧠 How It Works

- **Playwright**: Handles browser automation and DOM manipulation.
- **React + Vite**: Powers the interactive frontend user interface.
- **PyWebview**: Bridges the Python backend with the React frontend to create a seamless desktop application.
- **Google Gemini**: Interprets visual (screenshots) and textual (DOM) information to decide the next action.

## ⚠️ Troubleshooting

- **"pywebview bridge not available"**: Ensure you run the application with the `--gui` flag (`python main.py --gui`).
- **"GEMINI_API_KEY is not set"**: Make sure you have created a `.env` file in the project root containing your API key.
- **Browser doesn't launch**: You may need to manually install the Chromium browser for Playwright using `playwright install chromium`.
- **Frontend build errors**: Ensure Node.js 18+ is installed. Run `npm install` followed by `npm run build`.
