import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Chargement
model  = load_model("best_model.keras")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
print(f"Test : {len(X_test)} images")

# Prédictions
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# Rapport
print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred,
      target_names=["with_mask", "without_mask"]))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["with_mask", "without_mask"],
            yticklabels=["with_mask", "without_mask"])
plt.title("Matrice de confusion")
plt.ylabel("Vrai label")
plt.xlabel("Prédit")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Évaluation terminée !")