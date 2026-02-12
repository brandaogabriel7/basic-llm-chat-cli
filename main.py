import sys

from rich.console import Console

from chat import ChatSession
from config import CHAT_MODEL, MAX_TOKENS, STOP_SEQUENCES, TEMPERATURE, TOP_K, TOP_P

console = Console()
chat_session = ChatSession(
    console,
    chat_model=CHAT_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    top_k=TOP_K,
    stop_sequences=STOP_SEQUENCES,
)


def get_multiline_input(initial_line: str) -> str:
    lines = [initial_line.rstrip("\\")]
    while True:
        next_line = console.input("... ")
        if next_line.endswith("\\"):
            lines.append(next_line.rstrip("\\"))
        else:
            lines.append(next_line)
            break

    return "\n".join(lines)


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
        if user_input.endswith("\\"):
            user_input = get_multiline_input(user_input)

        chat_session.send_message(user_input)

except KeyboardInterrupt:
    console.print("\nExiting...", style="yellow")
    sys.exit(0)
