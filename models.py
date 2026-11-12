import tensorflow as tf
from tensorflow.keras import layers, models


NUM_SAMPLES = 88200
NUM_CLASSES = 10


def build_baseline_cnn():
    model = models.Sequential([
        layers.Input(shape=(NUM_SAMPLES, 1)),

        layers.Conv1D(32, kernel_size=9, strides=4, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_cnn_bigru():
    model = models.Sequential([
        layers.Input(shape=(NUM_SAMPLES, 1)),

        layers.Conv1D(32, kernel_size=9, strides=4, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # IMPORTANT: keep the sequence for BiGRU
        layers.Bidirectional(
            layers.GRU(64, return_sequences=False)
        ),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model