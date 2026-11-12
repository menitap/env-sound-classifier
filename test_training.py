from training_dataset import load_dataset
from models import build_cnn_bigru


print("Loading fold 1...")
X_train, y_train = load_dataset(
    folds=[1],
    augmented=False
)

print("\nBuilding CNN-BiGRU...")
model = build_cnn_bigru()

print("\nStarting 1-epoch training test...")

history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=1,
    validation_split=0.1,
    verbose=1
)

print("\nTraining test successful!")