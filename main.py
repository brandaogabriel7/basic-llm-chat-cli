from dotenv import load_dotenv
from anthropic import Anthropic

import sys
import os

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-20250514")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))


def get_system_prompt():
    """Read and return the contents of the system prompt from a text file."""
    with open("./system-prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


system_prompt = get_system_prompt()
client = Anthropic()

messages = []
try:
    print("CLI LLM Chat started\n\nHow can I help you today?\n")
    while True:
        user_input = input("Enter something (Ctrl+C to exit): ")
        messages.append({"role": "user", "content": user_input})

        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        ) as stream:
            print("Assistant: ")
            for text in stream.text_stream:
                print(text, end="", flush=True)

            messages.append(
                {
                    "role": "assistant",
                    "content": stream.get_final_text(),
                }
            )
            print("\n")
except KeyboardInterrupt:
    print("\nExiting...")
    sys.exit(0)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
