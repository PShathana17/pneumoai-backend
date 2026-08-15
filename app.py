import streamlit as st
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #f0f9ff 0%,
            #ffffff 50%,
            #eff6ff 100%
        );
    }

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        text-align: center;
        color: #0f3d56;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Upload card */
    .upload-card {
        background: white;
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(15, 61, 86, 0.10);
        border: 1px solid #dbeafe;
        margin-bottom: 25px;
    }

    /* Result cards */
    .normal-result {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
    }

    .pneumonia-result {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
    }

    /* Probability */
    .probability {
        text-align: center;
        color: #475569;
        font-size: 17px;
        margin-top: 12px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 15px;
        border-radius: 12px;
        margin-top: 30px;
        font-size: 13px;
        text-align: center;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 35px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
MODEL_PATH = "pneumonia_model.keras"


# --------------------------------------------------
# Load Model Only When Needed
# --------------------------------------------------
@st.cache_resource
def load_pneumonia_model():

    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)

    return model


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🫁 Pneumonia Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered chest X-ray analysis using a CNN deep learning model'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Upload Section
# --------------------------------------------------
st.subheader("📤 Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    st.write("")

    # --------------------------------------------------
    # Predict Button
    # --------------------------------------------------
    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        try:

            # --------------------------------------------------
            # Load model only when Predict is clicked
            # --------------------------------------------------
            with st.spinner(
                "Loading AI model and analyzing X-ray..."
            ):

                model = load_pneumonia_model()

                # --------------------------------------------------
                # Resize Image
                # --------------------------------------------------
                image_resized = image.resize(
                    (224, 224)
                )

                # --------------------------------------------------
                # Convert Image to NumPy
                # --------------------------------------------------
                image_array = np.array(
                    image_resized
                ).astype("float32")

                # --------------------------------------------------
                # Normalize Pixel Values
                # --------------------------------------------------
                image_array = image_array / 255.0

                # --------------------------------------------------
                # Add Batch Dimension
                # --------------------------------------------------
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # --------------------------------------------------
                # Model Prediction
                # --------------------------------------------------
                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                probability = float(
                    prediction[0][0]
                )

            # --------------------------------------------------
            # Prediction Result
            # --------------------------------------------------

            if probability >= 0.5:

                st.markdown(
                    """
                    <div class="pneumonia-result">
                        ⚠️ Pneumonia Detected
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                confidence = probability

            else:

                st.markdown(
                    """
                    <div class="normal-result">
                        ✅ Normal
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                confidence = 1 - probability

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            st.markdown(
                f"""
                <div class="probability">
                    Prediction Confidence:
                    <strong>{confidence:.2%}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                "❌ An error occurred while analyzing the X-ray."
            )

            st.exception(e)


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------
st.markdown(
    """
    <div class="disclaimer">
        ⚠️ This application is an AI/ML demonstration project.
        It is not intended to replace professional medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Pneumonia Detection • Deep Learning • TensorFlow • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)