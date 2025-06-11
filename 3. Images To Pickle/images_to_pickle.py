import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

# Map Romanized Devanagari labels to integer classes
roman_labels = [
    "ka", "kha", "ga", "gha", "nga",
    "cha", "chha", "ja", "jha", "yna",
    "tta", "ttha", "dda", "ddha", "nna",
    "ta", "tha", "da", "dha", "na",
    "pa", "pha", "ba", "bha", "ma",
    "ya", "ra", "la", "wa",
    "sha", "ssha", "sa", "ha",
    "ksha", "tra", "gya","khali"
]
label_to_index = {label: idx for idx, label in enumerate(roman_labels)}

def pickle_images_labels(data_dir="data/test", output_prefix="test"):
    images_labels = []
    images = glob(f"{data_dir}/*/*.jpg")
    print(f"Found {len(images)} images in {data_dir}")
    images.sort()

    for image in images:
        label_str = os.path.basename(os.path.dirname(image))
        if label_str not in label_to_index:
            print(f"Unknown label folder: {label_str}, skipping...")
            continue
        label = label_to_index[label_str]
        img = cv2.imread(image, 0)
        if img is None:
            print(f"failed to read image: {image}")
            continue
        images_labels.append((np.array(img, dtype=np.uint8), label))

    images_labels = shuffle(images_labels)
    if images_labels:
        images, labels = zip(*images_labels)
    else:
        print("No images found or loaded. Exiting.")
        return

    print("Total samples:", len(images_labels))

    # Save to pickle files in nsldata directory
    os.makedirs("nsldata", exist_ok=True)
    with open(f"nsldata/{output_prefix}_images", "wb") as f:
        pickle.dump(images, f)
    print(f"Saved nsldata/{output_prefix}_images")

    with open(f"nsldata/{output_prefix}_labels", "wb") as f:
        pickle.dump(labels, f)
    print(f"Saved nsldata/{output_prefix}_labels")

# Call for test set
pickle_images_labels("data/test", "test")

# Call for train set (optional)
pickle_images_labels("data/train", "train")
