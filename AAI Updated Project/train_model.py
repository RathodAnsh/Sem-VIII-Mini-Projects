"""
train_model.py
──────────────
Transfer Learning with MobileNetV2 — two-phase training.

Phase 1 : Freeze base, train custom top layers  (LR = 1e-3, 10 epochs)
Phase 2 : Unfreeze last 20 layers, fine-tune    (LR = 1e-4, 10 epochs)

Usage:
    python train_model.py
    python train_model.py --epochs1 15 --epochs2 15 --img_size 224
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                      Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau, TensorBoard)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42
MODEL_PATH = "models/eye_classifier.h5"


def build_model(img_size: int = 224, dropout: float = 0.3) -> Model:
    """Build MobileNetV2 with custom classification head."""
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False   # Phase 1: freeze everything

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout / 2)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base.input, outputs=output), base


def get_data_generators(img_size: int, batch_size: int):
    """Create train/val data generators with augmentation."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_aug.flow_from_directory(
        "dataset/train",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    val_data = val_aug.flow_from_directory(
        "dataset/val",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_data, val_data


def plot_history(h1, h2, save_path: str = "static/training_curves.png"):
    """Plot accuracy and loss curves for both phases."""
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    vloss= h1.history["val_loss"]      + h2.history["val_loss"]
    ep   = range(1, len(acc) + 1)
    ph2_start = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_color("#333")

    axes[0].plot(ep, acc,  color="#4ade80", lw=2,   label="Train Accuracy")
    axes[0].plot(ep, val,  color="#60a5fa", lw=2,   label="Val Accuracy")
    axes[0].axvline(ph2_start, color="#f59e0b", lw=1.5, ls="--", label="Fine-tune starts")
    axes[0].set_title("Accuracy", color="white", fontsize=13)
    axes[0].set_xlabel("Epoch", color="#aaa")
    axes[0].legend(facecolor="#1a1d27", labelcolor="white")

    axes[1].plot(ep, loss,  color="#f87171", lw=2,  label="Train Loss")
    axes[1].plot(ep, vloss, color="#c084fc", lw=2,  label="Val Loss")
    axes[1].axvline(ph2_start, color="#f59e0b", lw=1.5, ls="--", label="Fine-tune starts")
    axes[1].set_title("Loss", color="white", fontsize=13)
    axes[1].set_xlabel("Epoch", color="#aaa")
    axes[1].legend(facecolor="#1a1d27", labelcolor="white")

    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved → {save_path}")


def plot_confusion(model, val_data, save_path: str = "static/confusion_matrix.png"):
    """Generate and save confusion matrix."""
    val_data.reset()
    y_pred = (model.predict(val_data, verbose=1) > 0.5).astype(int).flatten()
    y_true = val_data.classes

    cm = confusion_matrix(y_true, y_pred)
    labels = list(val_data.class_indices.keys())

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5)
    ax.set_title("Confusion Matrix", color="white", pad=12)
    ax.set_xlabel("Predicted", color="#aaa")
    ax.set_ylabel("Actual", color="#aaa")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved → {save_path}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))


def train(epochs1: int = 8, epochs2: int = 7, img_size: int = IMG_SIZE):
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train_data, val_data = get_data_generators(img_size, BATCH_SIZE)
    model, base_model    = build_model(img_size)
    model.summary()

    callbacks_phase1 = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint("checkpoints/best_phase1.h5", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — Training top layers only")
    print("="*60)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(train_data, validation_data=val_data,
                   epochs=epochs1, callbacks=callbacks_phase1)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 — Fine-tuning last 20 layers")
    print("="*60)
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    callbacks_phase2 = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        ModelCheckpoint("checkpoints/best_phase2.h5", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(train_data, validation_data=val_data,
                   epochs=epochs2, callbacks=callbacks_phase2)

    # ── Save & evaluate ───────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    plot_history(h1, h2)
    plot_confusion(model, val_data)

    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    print(f"\nFinal Validation Accuracy : {val_acc*100:.2f}%")
    print(f"Final Validation Loss     : {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1",  type=int, default=10)
    parser.add_argument("--epochs2",  type=int, default=10)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()
    train(args.epochs1, args.epochs2, args.img_size)
