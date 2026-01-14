import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-20250514")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

console = Console()


def get_system_prompt():
    """Read and return the contents of the system prompt from a text file."""
    try:
        with open("./system-prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful assistant."


system_prompt = get_system_prompt()
client = Anthropic()

messages = []
try:
    console.print("[bold green]CLI LLM Chat started[/]\nHow can I help you today?")
    while True:
        user_input = console.input(
            "\n[dim italic]Enter something (Ctrl+C to exit): [/]"
        )
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        ) as stream:
            console.print("\nAssistant:\n", style="bold blue")
            with Live(refresh_per_second=10) as live:
                full_text = ""
                for text in stream.text_stream:
                    full_text += text
                    live.update(Markdown(full_text))

            messages.append(
                {
                    "role": "assistant",
                    "content": full_text,
                }
            )
except KeyboardInterrupt:
    console.print("\nExiting...", style="yellow")
    sys.exit(0)
