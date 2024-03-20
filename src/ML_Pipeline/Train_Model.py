from ML_Pipeline import Utils
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

# Function to train ML model
def train(model, x_train, y_train):
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    model.fit(x_train, y_train, batch_size=64, epochs=20)

    return model

# Function to initiate model and training data
def fit(data):
    columns = data.columns

    x_train = data.drop(Utils.TARGET, axis=1).values
    y_train = data[Utils.TARGET].values

    print(x_train.shape, y_train.shape)

    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(x_train.shape[1])),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    model = train(model, x_train, y_train)


    return model, columns
