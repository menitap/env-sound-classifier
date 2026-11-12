import numpy as np
from augmentation import add_noise


def create_noisy_dataset(X, snr_db):
    """
    Add noise to every audio sample in X.
    """

    noisy_X = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        noisy_X[i] = add_noise(X[i], snr_db)

    return noisy_X


def evaluate_noise_levels(model, X_test, y_test):
    """
    Evaluate model accuracy at 20, 10 and 5 dB SNR.
    """

    results = {}

    for snr in [20, 10, 5]:

        print(f"Testing at {snr} dB SNR...")

        X_noisy = create_noisy_dataset(X_test, snr)

        predictions = model.predict(
            X_noisy,
            verbose=0
        )

        predicted_labels = np.argmax(predictions, axis=1)

        accuracy = np.mean(
            predicted_labels == y_test
        )

        results[snr] = accuracy

        print(f"{snr} dB SNR Accuracy: {accuracy:.4f}")

    return results