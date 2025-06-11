import pickle
import numpy as np
from collections import Counter

with open("newdata/test_labels", "rb") as f:
    test_labels = pickle.load(f)

label_counts = Counter(test_labels)
print("Label counts in test_labels pickle:")
for label, count in sorted(label_counts.items()):
    print(f"Class {label}: {count} samples")

unique_labels = np.unique(test_labels)
print("Unique labels:", unique_labels)
