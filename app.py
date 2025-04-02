import streamlit as st
from model import verify_pil_image_caption
from PIL import Image


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
                    # The 'image' variable already holds the PIL Image object
                    # from the 'if uploaded_file is not None:' block above.
                    result = verify_pil_image_caption(
                        image, caption, instructions
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

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and OpenAI's Vision API")
