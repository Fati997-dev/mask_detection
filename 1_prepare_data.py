import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

IMG_SIZE = (128, 128)
DATASET_DIR = "data/"
CATEGORIES = ["with_mask", "without_mask"]

def load_dataset():
    data, labels = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_DIR, category)
        print(f"Chargement {category}...")
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0
                data.append(img_array)
                labels.append(label)
            except:
                pass
    return np.array(data), np.array(labels)

print("Chargement du dataset...")
X, y = load_dataset()
print(f"Total images : {len(X)}")
print(f"with_mask    : {np.sum(y == 0)}")
print(f"without_mask : {np.sum(y == 1)}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTrain : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

np.save("X_train.npy", X_train); np.save("y_train.npy", y_train)
np.save("X_val.npy",   X_val);   np.save("y_val.npy",   y_val)
np.save("X_test.npy",  X_test);  np.save("y_test.npy",  y_test)

print("\nDonnees sauvegardees !")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(CATEGORIES[y_train[i]], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("dataset_examples.png")
plt.show()
print("Terminé !")