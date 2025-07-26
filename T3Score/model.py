#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    26-Jul-2025 17:57:34

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input = keras.Input(shape=(3,3,3))
    conv1 = layers.Conv2D(32, (3,3), padding="same", name="conv1_")(input)
    bn1 = layers.BatchNormalization(epsilon=0.000010, name="bn1_")(conv1)
    relu1 = layers.ReLU()(bn1)
    conv2 = layers.Conv2D(64, (3,3), padding="same", name="conv2_")(relu1)
    bn2 = layers.BatchNormalization(epsilon=0.000010, name="bn2_")(conv2)
    relu2 = layers.ReLU()(bn2)
    fc1 = layers.Reshape((1, 1, -1), name="fc1_preFlatten1")(relu2)
    fc1 = layers.Dense(128, name="fc1_")(fc1)
    relu3 = layers.ReLU()(fc1)
    fc_out = layers.Reshape((1, 1, -1), name="fc_out_preFlatten1")(relu3)
    fc_out = layers.Dense(1, name="fc_out_")(fc_out)
    regression_output = layers.Flatten()(fc_out)

    model = keras.Model(inputs=[input], outputs=[regression_output])
    return model
