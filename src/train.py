import os, cv2, numpy as np, tensorflow as tf
from xml_parser import parse_xml
from model import build_model

IMG_SIZE = 224

def load_data(img_dir, xml_dir):
    images, boxes, labels = [], [], []

    for img_name in os.listdir(img_dir):
        img = cv2.imread(f"{img_dir}/{img_name}")
        h, w, _ = img.shape
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE)) / 255.0

        b, l = parse_xml(f"{xml_dir}/{img_name.replace('.png','.xml')}")
        boxes.append(np.array(b[0]) / [w,h,w,h])
        labels.append(l[0])
        images.append(img)

    return np.array(images), np.array(boxes), np.array(labels)

X_train, yb_train, y_train = load_data(
    "dataset/images/train", "dataset/annotations/train"
)
X_val, yb_val, y_val = load_data(
    "dataset/images/val", "dataset/annotations/val"
)

y_train = tf.keras.utils.to_categorical(y_train, 3)
y_val   = tf.keras.utils.to_categorical(y_val, 3)

model = build_model()
model.compile(
    optimizer="adam",
    loss={"bbox":"mse","class":"categorical_crossentropy"},
    metrics={"class":"accuracy"}
)

model.fit(
    X_train, {"bbox":yb_train,"class":y_train},
    validation_data=(X_val,{"bbox":yb_val,"class":y_val}),
    epochs=20
)

model.save("models/saved_model/face_mask_detector.keras")

