
# PROJECT CONFIGURATION

# Audio settings
SAMPLE_RATE = 22050
DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * DURATION

# Dataset settings
NUM_CLASSES = 10

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

# Model training settings
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Noise testing settings
SNR_LEVELS = [20, 10, 5]

print("Project configuration initialized!")
print("Sample rate:", SAMPLE_RATE)
print("Audio duration:", DURATION, "seconds")
print("Number of classes:", NUM_CLASSES)
print("Number of audio samples:", NUM_SAMPLES)
