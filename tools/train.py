#!/bin/python

import argparse
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30


def train(training_path: str, validation_path: str, test_path: str, output: str):
    # Augment only training data
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Only rescale validation and test data
    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    data = train_gen.flow_from_directory(
        training_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )

    val_data = val_test_gen.flow_from_directory(
        validation_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )

    print("✅ Dataset loaded successfully!")
    print("🔍 Class indices:", data.class_indices)
    print(
        "📊 Training class distribution:", dict(zip(*np.unique(data.classes, return_counts=True)))
    )

    num_classes = len(data.class_indices)

    # Build CNN
    model = models.Sequential(
        [
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),  # Increased from 0.25
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),  # Increased from 0.25
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),  # Increased from 0.25
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            # Softmax for multi-class
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("✅ Model built successfully!")
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001)

    model.fit(data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stop, reduce_lr])
    # Evaluate on validation set
    loss, acc = model.evaluate(val_data)
    print(f"🎯 Final Validation Accuracy: {acc:.2f}, loss: {loss:.2f}")

    if not output.endswith(".keras"):
        output += ".keras"

    model.save(output)
    print(f"✅ Saved model at {output}")

    # Final evaluation on the test set, if provided
    if test_path:
        print("\nEvaluating on the test set...")
        test_data = val_test_gen.flow_from_directory(
            test_path,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )
        test_loss, test_acc = model.evaluate(test_data)
        print(f"🎯 Final Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train a multi-class CNN image classifier")
    parser.add_argument(
        "-t", "--training", required=True, type=str, help="Path to training dataset"
    )
    parser.add_argument(
        "-v", "--validation", required=True, type=str, help="Path to validation dataset"
    )
    parser.add_argument(
        "-e", "--test", type=str, help="Optional path to test dataset for final evaluation"
    )
    parser.add_argument("-o", "--output", required=True, type=str, help="Output model path")

    args = parser.parse_args()

    train(args.training, args.validation, args.test, args.output)


if __name__ == "__main__":
    main()
