import sys
from textwrap import dedent
from anthropic import (
    APIConnectionError,
    APIStatusError,
    Anthropic,
    AuthenticationError,
    RateLimitError,
    NOT_GIVEN,
)
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from config import get_system_prompt

PRESETS = {
    "precise": {"temperature": 0.0, "top_p": 0.5, "top_k": 10},
    "balanced": {"temperature": 0.7, "top_p": 1.0, "top_k": 0},
    "creative": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
}


class ChatSession:
    def __init__(
        self,
        console: Console,
        chat_model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_sequences: list[str] | None = None,
    ):
        self._chat_model = chat_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._stop_sequences = stop_sequences or []
        self._client = Anthropic()
        self._messages = []
        self._system_prompt = get_system_prompt("./system-prompt.txt")
        self._console = console
        self._input_tokens = 0
        self._output_tokens = 0
        self._active_preset = None

        self._console.print(
            "[bold green]CLI LLM Chat started[/]\nHow can I help you today?"
        )

    def send_message(self, user_input: str):
        self._messages.append({"role": "user", "content": user_input})

        try:
            with self._client.messages.stream(
                model=self._chat_model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k if self._top_k > 0 else NOT_GIVEN,
                stop_sequences=self._stop_sequences if self._stop_sequences else NOT_GIVEN,
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
        except RateLimitError as e:
            self._messages.pop()
            self._console.print(f"Rate Limit Exceeded: {e}", style="red")
        except APIConnectionError as e:
            self._messages.pop()
            self._console.print(f"Connection Error: {e}", style="red")
        except AuthenticationError as e:
            self._console.print(f"Authentication Error: {e}", style="red")
            sys.exit(1)
        except APIStatusError as e:
            self._messages.pop()
            self._console.print(f"API Status Error: {e}", style="red")

    def handle_command(self, user_input: str):
        parts = user_input.split(maxsplit=1)
        command_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            "/system": self._handle_system_command,
            "/clear": self._handle_clear_command,
            "/quit": self._handle_quit_command,
            "/history": self._handle_history_command,
            "/info": self._handle_info_command,
            "/save": self._handle_save_command,
            "/help": self._handle_help_command,
            "/params": self._handle_params_command,
            "/preset": self._handle_preset_command,
        }

        if command_name not in commands:
            self._console.print(f"Unknown command: {command_name}", style="red")
            return

        command = commands[command_name]
        command(args)

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

    def _handle_info_command(self, _):
        preset_display = self._active_preset or "none"
        stop_display = ", ".join(self._stop_sequences) if self._stop_sequences else "none"
        info_text = dedent(f"""
        Chat Model: {self._chat_model}
        Temperature: {self._temperature}
        Max Tokens: {self._max_tokens}
        Top P: {self._top_p}
        Top K: {self._top_k}
        Stop Sequences: {stop_display}
        Active Preset: {preset_display}
        Total Messages: {len(self._messages)}
        Total Input Tokens: {self._input_tokens}
        Total Output Tokens: {self._output_tokens}
        """)
        self._console.print(info_text, style="bold")

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
        /info - Show chat session information.
        /params [name] [value] - View or set generation parameters.
        /preset [name] - Apply a parameter preset (precise, balanced, creative).
        /quit - Exit the chat.
        /save [filename] - Save chat history to a file.
        /help - Show this help message.

        Input multi-line messages by ending lines with a backslash (\\).
        """)
        self._console.print(help_text, style="bold")

    def _handle_params_command(self, args):
        if not args:
            self._console.print("\nCurrent parameters:", style="bold")
            self._console.print(f"  temperature  {self._temperature}")
            self._console.print(f"  max_tokens   {self._max_tokens}")
            self._console.print(f"  top_p        {self._top_p}")
            self._console.print(f"  top_k        {self._top_k}")
            return

        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            self._console.print(
                "Usage: /params <name> <value>", style="red"
            )
            return

        name, value = parts[0].lower(), parts[1]

        try:
            if name == "temperature":
                val = float(value)
                if not 0.0 <= val <= 1.0:
                    self._console.print("temperature must be between 0.0 and 1.0", style="red")
                    return
                self._temperature = val
            elif name == "max_tokens":
                val = int(value)
                if val < 1:
                    self._console.print("max_tokens must be at least 1", style="red")
                    return
                self._max_tokens = val
            elif name == "top_p":
                val = float(value)
                if not 0.0 <= val <= 1.0:
                    self._console.print("top_p must be between 0.0 and 1.0", style="red")
                    return
                self._top_p = val
            elif name == "top_k":
                val = int(value)
                if val < 0:
                    self._console.print("top_k must be 0 or greater", style="red")
                    return
                self._top_k = val
            else:
                self._console.print(
                    f"Unknown parameter: {name}. Available: temperature, max_tokens, top_p, top_k",
                    style="red",
                )
                return
        except ValueError:
            self._console.print(f"Invalid value for {name}: {value}", style="red")
            return

        self._active_preset = None
        self._console.print(f"{name} set to {value}", style="green")

    def _handle_preset_command(self, args):
        if not args:
            self._console.print("\nAvailable presets:", style="bold")
            for name, params in PRESETS.items():
                desc = ", ".join(f"{k}={v}" for k, v in params.items())
                self._console.print(f"  {name:10s} {desc}")
            return

        name = args.strip().lower()
        if name not in PRESETS:
            self._console.print(
                f"Unknown preset: {name}. Available: {', '.join(PRESETS)}",
                style="red",
            )
            return

        preset = PRESETS[name]
        self._temperature = preset["temperature"]
        self._top_p = preset["top_p"]
        self._top_k = preset["top_k"]
        self._active_preset = name
        self._console.print(f"Preset '{name}' applied.", style="green")
