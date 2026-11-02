"""MobileNetV2 transfer-learning classifier for single-channel thermal images."""
import tensorflow as tf
from tensorflow.keras import layers, applications
from . import config


def build_model(lr=1e-3):
    base = applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(config.IMG_H, config.IMG_W, 3))
    base.trainable = False  # frozen backbone for a quick prototype

    inp = layers.Input((config.IMG_H, config.IMG_W, 1))
    x = layers.Concatenate()([inp, inp, inp])    # 1 -> 3 channels
    x = layers.Rescaling(2.0, offset=-1.0)(x)    # [0, 1] -> [-1, 1] (MobileNetV2 input range)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
