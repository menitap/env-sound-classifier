from models import build_baseline_cnn, build_cnn_bigru


print("Creating baseline CNN...")
cnn = build_baseline_cnn()

print("\nBaseline CNN:")
cnn.summary()


print("\nCreating CNN-BiGRU...")
bigru = build_cnn_bigru()

print("\nCNN-BiGRU:")
bigru.summary()

print("\nModel creation successful!")