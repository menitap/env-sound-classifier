from training_dataset import load_dataset


# Test only one fold first
X, y = load_dataset(
    folds=[1],
    augmented=True
)

print("\nDataset loading successful!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)