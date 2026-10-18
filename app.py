import json

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(page_title="CNN Image Classifier", page_icon="🖼️")


@st.cache_resource
def load_everything():
    model = keras.models.load_model("model.keras")
    with open("class_names.json") as f:
        meta = json.load(f)
    return model, meta["classes"], meta["img_size"]


model, classes, img_size = load_everything()

st.title("🖼️ CNN Image Classifier")
st.caption(f"Upload an image and the model predicts one of {len(classes)} classes: "
           + ", ".join(classes))

files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"],
                         accept_multiple_files=True)

for file in files or []:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    x = np.expand_dims(np.array(img.resize((img_size, img_size)), dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption=file.name, use_container_width=True)
    with col2:
        st.success(f"**{classes[top]}**  ({probs[top]:.1%})")
        order = np.argsort(probs)[::-1][:5]
        st.bar_chart({classes[i]: float(probs[i]) for i in order})
        if probs[top] < 0.6:
            st.warning("Low confidence: the image may not belong to any trained class.")
    st.divider()
