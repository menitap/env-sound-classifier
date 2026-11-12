from training_dataset import load_train_val_test


(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
) = load_train_val_test(augmented=False)


print("\nDataset split successful!")

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nData types:")
print("X_train:", X_train.dtype)
print("y_train:", y_train.dtype)

print("\nSample shape:", X_train[0].shape)
print("Sample label:", y_train[0])