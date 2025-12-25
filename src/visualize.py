import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/saved_model/face_mask_detector.keras"
IMG_DIR = "dataset/images/test"

LABELS = ["with_mask", "without_mask", "mask_weared_incorrect"]

model = tf.keras.models.load_model(MODEL_PATH)


def draw(image, box, label_id):
    xmin, ymin, xmax, ymax = box.astype(int)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(
        image,
        LABELS[label_id],
        (xmin, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    return image


images = os.listdir(IMG_DIR)[:10]

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)

    # original image (NO resize)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # model input (only normalize)
    input_img = image_rgb / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # prediction
    pred_box, pred_class = model.predict(input_img, verbose=0)
    label_id = np.argmax(pred_class[0])

    # draw bounding box
    image = draw(image, pred_box[0], label_id)

    cv2.imshow("Face Mask Detection", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()




