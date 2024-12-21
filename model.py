from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization


def create_model(nodes_n):
    return Sequential(
        [
            Dense(128, activation="relu", input_shape=(nodes_n, nodes_n, 1)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.1),
            BatchNormalization(),
            Dense(128, activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )
