

# BASELINE 1D CNN MODEL
import tensorflow as tf
from config import NUM_SAMPLES, NUM_CLASSES, LEARNING_RATE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

def build_cnn_model():

    model = Sequential([
        Input(shape=(NUM_SAMPLES, 1)),

        # First convolution block
        Conv1D(32, kernel_size=9, strides=4, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),

        # Second convolution block
        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),

        # Third convolution block
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Classification layers
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Build the model
cnn_model = build_cnn_model()

print("Baseline CNN created successfully!")
cnn_model.summary()
