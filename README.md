# Basic LLM Chat CLI

An interactive command-line chat application with streaming responses, conversation history, and slash commands — built with Python and the Anthropic SDK.

## Features

- **Streaming responses** — real-time token-by-token output with live Markdown rendering
- **Conversation history** — multi-turn context preserved across messages
- **Slash commands** — `/system`, `/clear`, `/history`, `/info`, `/save`, `/help`, `/quit`
- **Customizable system prompt** — loaded from file, changeable at runtime
- **Multiline input** — end a line with `\` to continue on the next
- **Token tracking** — monitors input/output tokens across the session
- **Rich terminal UI** — styled output with `rich` for Markdown, colors, and live updates

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Runtime |
| [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) | Claude API client with streaming |
| [Rich](https://github.com/Textualize/rich) | Terminal styling, Markdown rendering, live updates |
| python-dotenv | Environment variable management |

## How It Works

```
User Input → REPL Loop → ChatSession → Anthropic Streaming API
                                              ↓
                              Live Markdown Rendering (Rich)
                                              ↓
                              History + Token Accumulation
```

The CLI runs a Read-Eval-Print loop that sends messages to Claude via the Anthropic streaming API. Responses are rendered in real-time as Markdown using Rich's `Live` widget at 10 refreshes/second. Full conversation history is maintained in memory for multi-turn context.

## Project Structure

```
basic-llm-chat-cli/
├── main.py              # REPL loop and command routing
├── chat.py              # ChatSession class (streaming, history, error handling)
├── config.py            # Configuration from env vars + system prompt file
├── system-prompt.txt    # Default system prompt (customizable)
├── requirements.txt     # Dependencies
└── .python-version      # Python 3.12
```

## Getting Started

### Prerequisites

- Python 3.12+
- [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
git clone https://github.com/brandaogabriel7/basic-llm-chat-cli.git
cd basic-llm-chat-cli

pip install -r requirements.txt

# Create .env with your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### Run

```bash
python main.py
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key |
| `CHAT_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `TEMPERATURE` | `0.7` | Response creativity (0.0–1.0) |
| `MAX_TOKENS` | `1024` | Max response length |

## Commands

| Command | Description |
|---------|-------------|
| `/system [prompt]` | View or change the system prompt (clears history) |
| `/clear` | Clear conversation history |
| `/history` | Display full chat history |
| `/info` | Show session stats (model, tokens, message count) |
| `/save <filename>` | Save chat transcript to file |
| `/help` | Show available commands |
| `/quit` | Exit the application |

## Technical Highlights

- **Streaming with live rendering** — uses `client.messages.stream()` with Rich's `Live` widget for smooth, real-time Markdown output
- **Graceful error recovery** — handles rate limits, connection errors, and auth failures with specific exception types; removes failed messages from history to prevent corrupted state
- **Zero abstraction layers** — direct Anthropic SDK integration without LangChain or other frameworks
- **Clean separation** — config, session management, and REPL loop in dedicated modules
