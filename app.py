import streamlit as st
from PIL import Image
from functions import load_trained_model, predict_label
from signature import display_signature

# 🟢 Page Setup (Must be first)
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 🟠 Custom CSS for Centering & Styling
st.markdown(
    """
    <style>
        /* Center everything */
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        /* Title Styling (White) */
        .title {
            text-align: center;
            color: white !important;
            font-size: 36px;
            font-weight: bold;
        }
        /* Image Container */
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }
        /* Predicted Label - White */
        .predicted-label {
            font-size: 24px;
            font-weight: bold;
            color: white !important;
            text-align: center;
        }
        /* Prediction Result - RED */
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            color: red !important;
        }
        /* Upload Label Text White */
        label, p {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🟣 Title (Centered & White)
st.markdown("<h1 class='title'>🚦 Traffic Sign Recognition</h1>", unsafe_allow_html=True)

# 🔵 Load Model
with st.spinner('Loading model...'):
    model = load_trained_model()

if model is None:
    st.warning("⚠️ Failed to load the model. Please check the model files.")

# 🟢 Upload Image
uploaded_file = st.file_uploader("📤 Upload an image (JPEG, JPG, PNG)", type=["jpg", "jpeg", "png"])

# 🟠 Image Processing
if uploaded_file is not None:
    try:
        with st.spinner('🔍 Analyzing image...'):
            image = Image.open(uploaded_file)

            # **Reduce Image Size**
            new_size = (250, 250)  # Slightly smaller for better view
            image = image.resize(new_size)

            # **Center Image Using Columns**
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:  # Middle Column (Perfect Center)
                st.image(image, caption="📷 Uploaded Image", use_container_width=False)

            # **Prediction**
            label = predict_label(image, model)
            if label is not None:
                # 🟢 **Predicted Label + Answer in One Line**
                st.markdown(f"""
                    <h3 class="predicted-label">Predicted Label: 
                        <span class="prediction-result">{label}</span>
                    </h3>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# 🟣 Display Signature
display_signature()

# 🔴 Background Color (Dark Mode)
st.markdown("""
    <style>
        .stApp {
            background-color: #1e1e1e; /* Dark Gray */
        }
    </style>
""", unsafe_allow_html=True)
