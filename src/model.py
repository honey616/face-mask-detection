import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)

    bbox = layers.Dense(4, activation="sigmoid", name="bbox")(x)
    cls  = layers.Dense(3, activation="softmax", name="class")(x)

    return models.Model(base.input, [bbox, cls])
