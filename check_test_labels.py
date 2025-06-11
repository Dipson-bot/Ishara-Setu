import pickle
import numpy as np

with open("nsldata/test_labels", "rb") as f:
    test_labels = pickle.load(f)

unique_labels = np.unique(test_labels)
print("Unique labels in test_labels:", unique_labels)
