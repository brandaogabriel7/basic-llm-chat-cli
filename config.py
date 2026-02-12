import os

from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-20250514")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = int(os.environ.get("TOP_K", "0"))
STOP_SEQUENCES = []


def get_system_prompt(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful assistant."
