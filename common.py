from constants import SYSTEM_PROMPT

def build_prompt(instruction: str) -> str:
    """Construct the standard chat-style prompt used across the project.

    Parameters:
        instruction: The natural language request/instruction from the user.

    Returns:
        A single string forming the prompt for the assistant to generate one Linux command.
    """
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nTask: Turn the request into a single Linux command.\n"
        f"Request: {instruction}\n"
        f"Answer with one line only.\n"
        f"<|assistant|>\n"
    )
