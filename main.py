import os
import sys
from textwrap import dedent

from dotenv import load_dotenv
from anthropic import Anthropic
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-20250514")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))


class ChatSession:
    def __init__(self, console: Console):
        self._client = Anthropic()
        self._messages = []
        self._system_prompt = self._get_system_prompt("./system-prompt.txt")
        self._console = console
        self._input_tokens = 0
        self._output_tokens = 0

        console.print("[bold green]CLI LLM Chat started[/]\nHow can I help you today?")

    def send_message(self, user_input: str):
        self._messages.append({"role": "user", "content": user_input})

        with self._client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=self._system_prompt,
            messages=self._messages,
        ) as stream:
            self._console.print("\nAssistant:\n", style="bold blue")
            with Live(refresh_per_second=10) as live:
                full_text = ""
                for text in stream.text_stream:
                    full_text += text
                    live.update(Markdown(full_text))

                self._messages.append(
                    {
                        "role": "assistant",
                        "content": full_text,
                    }
                )
                usage = stream.get_final_message().usage
                self._input_tokens += usage.input_tokens
                self._output_tokens += usage.output_tokens

    def handle_command(self, user_input: str):
        parts = user_input.split(maxsplit=1)
        command_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            "/system": self._handle_system_command,
            "/clear": self._handle_clear_command,
            "/quit": self._handle_quit_command,
            "/history": self._handle_history_command,
            "/tokens": self._handle_tokens_command,
            "/save": self._handle_save_command,
            "/help": self._handle_help_command,
        }

        if command_name not in commands:
            self._console.print(f"Unknown command: {command_name}", style="red")
            return

        command = commands[command_name]
        command(args)

    def _get_system_prompt(self, filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "You are a helpful assistant."

    def _handle_system_command(self, args):
        if not args:
            self._console.print("\nCurrent system prompt: ", style="bold")
            self._console.print(self._system_prompt, style="dim")
        else:
            self._system_prompt = args
            self._messages = []
            self._console.print("History cleared. New system prompt: ", style="bold")
            self._console.print(self._system_prompt, style="dim")

    def _handle_clear_command(self, _):
        self._messages = []
        self._console.print("Chat history cleared.", style="yellow")

    def _handle_quit_command(self, _):
        self._console.print("Exiting...", style="yellow")
        sys.exit(0)

    def _handle_history_command(self, _):
        if not self._messages:
            self._console.print("No chat history available.", style="yellow")
            return

        self._console.print("\nChat History:", style="bold")
        for msg in self._messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            self._console.print(f"\n[bold]{role}:[/]\n{content}", style="dim")

    def _handle_tokens_command(self, _):
        self._console.print(
            f"\nTotal Input Tokens: {self._input_tokens}\nTotal Output Tokens: {self._output_tokens}",
            style="bold",
        )

    def _handle_save_command(self, args):
        if not args:
            self._console.print(
                "Please provide a filename to save the chat history.", style="red"
            )
            return

        try:
            with open(args, "w", encoding="utf-8") as f:
                for msg in self._messages:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    f.write(f"{role}:\n{content}\n\n")
            self._console.print(f"Chat history saved to {args}", style="green")
        except Exception as e:
            self._console.print(f"Failed to save chat history: {e}", style="red")

    def _handle_help_command(self, _):
        help_text = dedent("""
        Available commands:
        /system [new prompt] - View or set the system prompt.
        /clear - Clear chat history.
        /history - View chat history.
        /tokens - View token usage.
        /quit - Exit the chat.
        /save [filename] - Save chat history to a file.
        /help - Show this help message.
        """)
        self._console.print(help_text, style="bold")


console = Console()
chat_session = ChatSession(console)
try:
    while True:
        user_input = console.input(
            "\n[dim italic]Enter something (Ctrl+C to exit): [/]"
        )
        if not user_input.strip():
            continue
        if user_input.startswith("/"):
            chat_session.handle_command(user_input)
            continue

        chat_session.send_message(user_input)

except KeyboardInterrupt:
    console.print("\nExiting...", style="yellow")
    sys.exit(0)
