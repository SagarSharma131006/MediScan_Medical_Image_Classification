
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from huggingface_hub import hf_hub_download

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# Configuration
# ============================================================

HF_REPO_ID = "SagarsS9812/mediscan-efficientnet-b0"
MODEL_FILENAME = "efficientnet_b0_best_day8.pth"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DEVICE = torch.device("cpu")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="MediScan — Brain MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MediScan")
st.subheader("Brain MRI Classification with EfficientNet-B0")

st.write(
    "Upload a brain MRI image to classify it into one of four classes "
    "and visualize the model's decision region using Grad-CAM."
)


# ============================================================
# Download Model from Hugging Face
# ============================================================

@st.cache_resource
def get_model_path():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    return model_path


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    model_path = get_model_path()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = models.efficientnet_b0(weights=None)

    dropout = checkpoint.get("dropout", 0.1)
    num_classes = checkpoint.get("num_classes", 4)

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Prediction Function
# ============================================================

def predict(image):

    image_rgb = image.convert("RGB")

    input_tensor = transform(
        image_rgb
    ).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = CLASS_NAMES[
        predicted_idx.item()
    ]

    return (
        predicted_class,
        confidence.item(),
        probabilities[0].cpu().numpy(),
        input_tensor,
        predicted_idx.item()
    )


# ============================================================
# Grad-CAM Function
# ============================================================

def generate_gradcam(
    input_tensor,
    predicted_idx
):

    target_layer = model.features[-1]

    targets = [
        ClassifierOutputTarget(
            predicted_idx
        )
    ]

    with GradCAM(
        model=model,
        target_layers=[target_layer]
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    return grayscale_cam


# ============================================================
# File Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Main Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Original MRI")

        st.image(
            image,
            use_container_width=True
        )

    (
        predicted_class,
        confidence,
        probabilities,
        input_tensor,
        predicted_idx
    ) = predict(image)

    with col2:

        st.write("### Prediction")

        st.success(
            f"Predicted Class: {predicted_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.write("### Class Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            st.write(
                f"**{class_name}**: "
                f"{probability * 100:.2f}%"
            )

    # ========================================================
    # Grad-CAM
    # ========================================================

    st.divider()

    st.write("### 🔥 Grad-CAM Explainability")

    grayscale_cam = generate_gradcam(
        input_tensor,
        predicted_idx
    )

    resized_image = image.resize(
        (224, 224)
    )

    rgb_image = np.asarray(
        resized_image
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    st.image(
        visualization,
        caption=(
            f"Grad-CAM — "
            f"{predicted_class.upper()}"
        ),
        use_container_width=True
    )

    st.info(
        "Grad-CAM highlights image regions that "
        "contributed to the model's prediction. "
        "It should be treated as an explainability "
        "aid, not a clinical diagnosis."
    )

else:

    st.info(
        "👆 Upload a brain MRI image to begin prediction."
    )
