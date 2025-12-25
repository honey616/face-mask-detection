import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "models/saved_model/face_mask_detector.keras"
)

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

st.title("Face Mask Detection App")

uploaded_file = st.file_uploader(
    "Upload an Image", type=["jpg", "png", "jpeg"]
)

def letterbox(image, size=224):
    h, w, _ = image.shape
    scale = size / max(h, w)

    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2

    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = image_resized
    return canvas, scale, pad_x, pad_y


if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    H, W, _ = img_np.shape

    # Letterbox resize
    img_lb, scale, pad_x, pad_y = letterbox(img_np)

    # Normalize
    img_norm = img_lb / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)

    # Model prediction
    bbox_pred, class_pred = model.predict(img_norm, verbose=0)

    # ===== BOUNDING BOX =====
    x1, y1, x2, y2 = bbox_pred[0]

    # Clamp 0–1
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)

    # Convert to 224 space
    x1, x2 = x1 * 224, x2 * 224
    y1, y2 = y1 * 224, y2 * 224

    # Remove padding and rescale to original image
    x1 = (x1 - pad_x) / scale
    x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    y2 = (y2 - pad_y) / scale

    # Final clamp to image size
    x1, x2 = int(max(0, x1)), int(min(W, x2))
    y1, y2 = int(max(0, y1)), int(min(H, y2))

    # ===== CLASS =====
    class_id = np.argmax(class_pred[0])
    label = CLASSES[class_id]

    # ===== COLOR LOGIC =====
    if label == "with_mask":
        color = (0, 255, 0)   # GREEN
    else:
        color = (0, 0, 255)   # RED

    # Draw bounding box
    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

    # Draw label
    cv2.putText(
        img_np,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # Show output
    st.image(img_np, caption="Prediction with Bounding Box", width=500)

    if label == "with_mask":
        st.success("Prediction: WITH MASK ✅")
    else:
        st.error(f"Prediction: {label.upper()} ❌")


