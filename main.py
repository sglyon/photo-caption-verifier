import os
import base64
from pathlib import Path
from typing import Optional
from openai import OpenAI
import typer
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from tempfile import NamedTemporaryFile


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


app = typer.Typer(
    name="photo-caption-verifier",
    help="Verify if a caption matches an image using OpenAI's vision model.",
    add_completion=False,
)


@app.command()
def verify(
    image: str = typer.Option(
        "mountains.jpg",
        "--image",
        "-i",
        help="Path to the image file",
    ),
    caption: str = typer.Option(
        "A beautiful sunset over mountains",
        "--caption",
        "-c",
        help="Caption to verify",
    ),
    instructions: str = typer.Option(
        "Please check if the colors and composition match the caption. Your response must start with `correct` or `incorrect`.",
        "--instructions",
        "-ins",
        help="Additional instructions for verification",
    ),
):
    """
    Verify if a caption matches an image using OpenAI's vision model.
    """
    try:
        result = verify_image_caption(image, caption, instructions)
        print("Verification Result:", result["verification"])

    except Exception as e:
        print(f"Error: {str(e)}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """
    Start a FastAPI server to serve the verify function.
    """
    api = FastAPI(
        title="Photo Caption Verifier API",
        description="API to verify if a caption matches an image using OpenAI's vision model",
        version="1.0.0",
    )

    @api.post("/verify")
    async def verify_endpoint(
        image: UploadFile = File(...),
        caption: str = Form(...),
        instructions: str = Form(None),
    ):
        try:
            # Save uploaded file temporarily
            with NamedTemporaryFile(
                delete=False, suffix=Path(image.filename).suffix
            ) as temp_file:
                content = await image.read()
                temp_file.write(content)
                temp_file.flush()

                # Verify the image
                result = verify_image_caption(temp_file.name, caption, instructions)

                # Clean up
                os.unlink(temp_file.name)

                # Return only serializable data
                return JSONResponse(
                    content={
                        "verification": result["verification"],
                        "model": result["raw_response"].model,
                        "created": result["raw_response"].created,
                        "usage": result["raw_response"].usage.model_dump(),
                    }
                )

        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
