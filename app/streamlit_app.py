from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.gradcam import overlay_cam, GradCAM
from src.inference import load_model_for_inference, open_medical_image, predict_pil_image
from src.models import get_gradcam_target_layer
from src.data import get_transforms


st.set_page_config(page_title="MediScan", page_icon=":brain:", layout="wide")
st.title("MediScan - Medical Image Classification")
st.caption("Brain MRI tumor classification with transfer learning and Grad-CAM explainability.")

checkpoint_path = st.sidebar.text_input("Model checkpoint", "models/efficientnet_b0_best.pt")
image_size = st.sidebar.number_input("Image size", min_value=128, max_value=512, value=224, step=32)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)


@st.cache_resource
def cached_model(path):
    return load_model_for_inference(path)


if not Path(checkpoint_path).exists():
    st.warning("Train the model first, or update the checkpoint path in the sidebar.")
    st.stop()

model, class_to_idx, idx_to_class, checkpoint = cached_model(checkpoint_path)
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
architecture = checkpoint.get("architecture", "efficientnet_b0")
target_layer = get_gradcam_target_layer(model, architecture)

uploaded_files = st.file_uploader(
    "Upload MRI image(s)",
    type=["jpg", "jpeg", "png", "dcm"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more JPEG, PNG, or DICOM files to run classification.")
    st.stop()

rows = []
for uploaded in uploaded_files:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    image = open_medical_image(tmp_path)
    result = predict_pil_image(model, image, idx_to_class, image_size=image_size)
    rows.append({"filename": uploaded.name, **result["probabilities"], "prediction": result["pred_label"], "confidence": result["confidence"]})

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)
    with right:
        st.subheader(result["pred_label"])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        st.bar_chart(pd.Series(result["probabilities"]).sort_values(ascending=False))

    if show_gradcam:
        transform = get_transforms("test", image_size=image_size)
        tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
        cam_runner = GradCAM(model, target_layer)
        cam, _ = cam_runner(tensor, class_idx=result["pred_idx"])
        cam_runner.remove_hooks()
        rgb = image.convert("RGB").resize((image_size, image_size))
        import numpy as np

        overlay = overlay_cam(np.asarray(rgb).astype("float32") / 255.0, cam)
        st.image(overlay, caption="Grad-CAM decision region", use_container_width=True)

results_df = pd.DataFrame(rows)
st.download_button(
    "Download predictions CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="mediscan_predictions.csv",
    mime="text/csv",
)
st.dataframe(results_df, use_container_width=True)
