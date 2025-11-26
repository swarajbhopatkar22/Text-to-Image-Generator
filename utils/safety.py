# utils/safety.py

def is_prompt_allowed(prompt: str) -> bool:
    banned_words = ["nude", "sex", "nsfw", "blood", "kill", "gore", "violence"]
    text = prompt.lower()
    return not any(word in text for word in banned_words)
