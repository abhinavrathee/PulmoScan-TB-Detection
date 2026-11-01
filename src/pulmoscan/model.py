import tensorflow as tf
from tensorflow.keras import layers


def build_model(image_size: int, dropout: float = 0.5):
    inputs = layers.Input(shape=(image_size, image_size, 3), name="chest_xray")
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    features = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = layers.Dense(256, activation="relu", name="classifier_dense")(features)
    features = layers.Dropout(dropout, name="classifier_dropout")(features)
    outputs = layers.Dense(1, activation="sigmoid", name="tb_probability")(features)
    return tf.keras.Model(inputs, outputs, name="pulmoscan_efficientnetb3"), backbone
