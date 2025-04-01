import streamlit as st
from model import verify_image_caption
from PIL import Image
import tempfile
import os


st.set_page_config(
    page_title="Photo Caption Verifier",
    page_icon="📸",
    layout="wide",
)

st.title("📸 Photo Caption Verifier 📸")

st.markdown(
    """
    Upload an image and provide a caption to verify if the caption accurately describes the image.
    The verification will be performed using OpenAI's vision model.
    """
)

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        help="Supported formats: PNG, JPG, JPEG, GIF, WEBP",
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

with col2:
    st.subheader("Caption Verification")
    caption = st.text_area(
        "Enter the caption to verify",
        value="A beautiful sunset over mountains",
        height=100,
    )

    instructions = st.text_area(
        "Additional instructions (optional)",
        value="Please check if the colors and composition match the caption. Your response must start with 'correct' or 'incorrect'.",
        height=100,
    )

    if st.button("Verify Caption", type="primary"):
        if uploaded_file is None:
            st.error("Please upload an image first!")
        else:
            with st.spinner("Verifying caption..."):
                try:
                    # Create a temporary file with the correct extension
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=file_extension
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.flush()

                        # Verify the caption
                        result = verify_image_caption(
                            temp_file.name, caption, instructions
                        )

                        # Display the result
                        st.markdown("### Verification Result")
                        st.write(result["verification"])

                        # Display additional information in an expander
                        with st.expander("Show Details"):
                            st.json(
                                {
                                    "model": result["raw_response"].model,
                                    "created": result["raw_response"].created,
                                    "usage": result["raw_response"].usage.model_dump(),
                                }
                            )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_file.name)
                    except Exception as _e:
                        pass  # Ignore errors during cleanup

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and OpenAI's Vision API")
