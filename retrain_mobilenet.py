import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Chargement des données...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val   = np.load("X_val.npy")
y_val   = np.load("y_val.npy")

# ── INVERSER LES LABELS
# 0 (with_mask) → 1  |  1 (without_mask) → 0
y_train_inv = 1 - y_train
y_val_inv   = 1 - y_val

print(f"Après inversion — with_mask(1): {(y_train_inv==1).sum()} | without_mask(0): {(y_train_inv==0).sum()}")

# Remettre pixels entre 0-255 pour MobileNetV2
X_train_mb = (X_train * 255).astype("uint8")
X_val_mb   = (X_val   * 255).astype("uint8")

# ── Modèle MobileNetV2
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train_mb)

callbacks = [
    ModelCheckpoint("best_model.keras", monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1)
]

# ── Phase 1
print("\nPhase 1 — Entraînement classificateur (10 epochs)...")
history1 = model.fit(
    datagen.flow(X_train_mb, y_train_inv, batch_size=32),
    epochs=10,
    validation_data=(X_val_mb, y_val_inv),
    callbacks=callbacks,
    verbose=1
)

# ── Phase 2 Fine-tuning
print("\nPhase 2 — Fine-tuning (20 epochs)...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    datagen.flow(X_train_mb, y_train_inv, batch_size=32),
    epochs=20,
    validation_data=(X_val_mb, y_val_inv),
    callbacks=callbacks,
    verbose=1
)

# ── Courbes
all_acc   = history1.history['accuracy']     + history2.history['accuracy']
all_val   = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss  = history1.history['loss']         + history2.history['loss']
all_vloss = history1.history['val_loss']     + history2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(all_acc,  label='Train')
ax1.plot(all_val,  label='Val')
ax1.axvline(x=10, color='gray', linestyle='--', label='Fine-tuning')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(all_loss, label='Train')
ax2.plot(all_vloss, label='Val')
ax2.axvline(x=10, color='gray', linestyle='--', label='Fine-tuning')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_mobilenet.png")
plt.show()

val_acc = max(history1.history['val_accuracy'] + history2.history['val_accuracy'])
print(f"\nMeilleure val_accuracy : {val_acc*100:.2f}%")
print("Modèle sauvegardé dans best_model.keras")