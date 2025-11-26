# utils/prompts.py

def build_full_prompt(base_prompt: str, style: str) -> str:
    styles = {
        "photorealistic": "high detail, ultra realistic, 8k, cinematic lighting",
        "artistic": "digital art, painting, vibrant colors, detailed brush strokes",
        "cartoon": "pixar style, cartoon, colorful, soft shading",
    }

    style_tag = styles.get(style.lower(), "")
    return f"{base_prompt}, {style_tag}".strip()


def build_negative_prompt(negative: str | None) -> str:
    default_negative = (
        "blurry, distorted, low resolution, bad anatomy, watermark, text, deformed face"
    )
    if negative and negative.strip():
        return f"{default_negative}, {negative}"
    return default_negative
