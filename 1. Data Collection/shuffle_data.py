import os
import random

# Base data folder path
base_folder = "D:/Downloads/Ishara_setu/data"
train_folder = os.path.join(base_folder, "train")
test_folder = os.path.join(base_folder, "test")

# Romanized Devanagari labels
roman_labels = [
    "ka", "kha", "ga", "gha", "nga",
    "cha", "chha", "ja", "jha", "yna",
    "tta", "ttha", "dda", "ddha", "nna",
    "ta", "tha", "da", "dha", "na",
    "pa", "pha", "ba", "bha", "ma",
    "ya", "ra", "la", "wa",
    "sha", "ssha", "sa", "ha",
    "ksha", "tra", "gya"
]

# Iterate over train and test folders
for folder_path in [train_folder, test_folder]:
    for label in roman_labels:
        subfolder_path = os.path.join(folder_path, label)
        if not os.path.exists(subfolder_path):
            print(f"Skipping non-existent folder: {subfolder_path}")
            continue

        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".jpg")]
        random.shuffle(images)

        for i, image in enumerate(images):
            old_path = os.path.join(subfolder_path, image)
            new_filename = f"{i+1}_{image}"
            new_path = os.path.join(subfolder_path, new_filename)
            os.rename(old_path, new_path)

        print(f"Renamed {len(images)} images in {label} ({folder_path})")