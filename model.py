import os
import base64
from pathlib import Path
from typing import Optional
from openai import OpenAI


def verify_image_caption(
    image_path: str,
    caption: str,
    instructions: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> dict:
    """
    Verify if a caption matches an image using OpenAI's vision model.

    Args:
        image_path: Path to the image file
        caption: The caption to verify
        instructions: Optional specific instructions for verification
        client: Optional OpenAI client (will create one if not provided)

    Returns:
        dict: Response from the OpenAI API
    """
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Get the image file extension and validate it
    image_ext = Path(image_path).suffix.lower().lstrip(".")
    if image_ext not in ["png", "jpeg", "jpg", "gif", "webp"]:
        raise ValueError(
            f"Unsupported image format. Please use one of: png, jpeg, gif, webp"
        )

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare the messages for the API
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please verify if this caption accurately describes the image: '{caption}'",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_ext};base64,{encoded_image}"
                    },
                },
            ],
        }
    ]

    if instructions:
        messages[0]["content"][0]["text"] += (
            f"\n\nAdditional instructions: {instructions}"
        )

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=500
    )

    return {
        "verification": response.choices[0].message.content,
        "raw_response": response,
    }
