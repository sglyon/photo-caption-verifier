import os
import base64
import io
from pathlib import Path
from typing import Optional
from openai import OpenAI
from PIL import Image


def _call_openai_vision_api(
    encoded_image: str,
    image_format: str,
    caption: str,
    instructions: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> dict:
    """
    Helper function to call the OpenAI vision API.

    Args:
        encoded_image: Base64 encoded image string.
        image_format: The format of the image (e.g., 'png', 'jpeg').
        caption: The caption to verify.
        instructions: Optional specific instructions for verification.
        client: Optional OpenAI client.

    Returns:
        dict: Response from the OpenAI API.
    """
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
                        "url": f"data:image/{image_format};base64,{encoded_image}"
                    },
                },
            ],
        }
    ]

    if instructions:
        # Ensure instructions are added correctly to the text part
        if isinstance(messages[0]["content"][0], dict) and messages[0]["content"][0]["type"] == "text":
             messages[0]["content"][0]["text"] += (
                f"\n\nAdditional instructions: {instructions}"
            )
        else:
             # Fallback or error handling if structure is unexpected
             messages[0]["content"].append({
                 "type": "text",
                 "text": f"Additional instructions: {instructions}"
             })

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=500
    )

    return {
        "verification": response.choices[0].message.content,
        "raw_response": response,
    }


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
        dict: Response from the OpenAI API, including 'verification' and 'raw_response'.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Get the image file extension and validate it
    image_format = Path(image_path).suffix.lower().lstrip(".")
    if image_format not in ["png", "jpeg", "jpg", "gif", "webp"]:
        raise ValueError(
            f"Unsupported image format '{image_format}'. Please use one of: png, jpeg, jpg, gif, webp"
        )

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Call the helper function
    return _call_openai_vision_api(
        encoded_image=encoded_image,
        image_format=image_format,
        caption=caption,
        instructions=instructions,
        client=client,
    )


def verify_pil_image_caption(
    image: Image.Image,
    caption: str,
    instructions: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> dict:
    """
    Verify if a caption matches a PIL Image object using OpenAI's vision model.

    Args:
        image: PIL Image object
        caption: The caption to verify
        instructions: Optional specific instructions for verification
        client: Optional OpenAI client (will create one if not provided)

    Returns:
        dict: Response from the OpenAI API, including 'verification' and 'raw_response'.
    """
    # Determine image format, default to PNG if not available
    image_format = image.format if image.format else "PNG"
    image_format_lower = image_format.lower()

    if image_format_lower not in ["png", "jpeg", "jpg", "gif", "webp"]:
        raise ValueError(
            f"Unsupported image format '{image_format}'. Please use one of: png, jpeg, jpg, gif, webp"
        )

    # Save image to an in-memory buffer
    buffer = io.BytesIO()
    # PIL uses 'JPEG' format name, map it for MIME type consistency if needed
    save_format = 'JPEG' if image_format_lower == 'jpg' else image_format
    image.save(buffer, format=save_format)
    image_bytes = buffer.getvalue()

    # Encode the image bytes
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Call the helper function
    return _call_openai_vision_api(
        encoded_image=encoded_image,
        image_format=image_format_lower,
        caption=caption,
        instructions=instructions,
        client=client,
    )
