#!/bin/python

import argparse
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess_input,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100


def train(
    training_path: str,
    validation_path: str,
    test_path: str,
    output: str,
    use_transfer_learning: bool,
    base_model_name: str,
):
    if use_transfer_learning:
        print(f"🚀 Using transfer learning with {base_model_name}")
        if base_model_name == "mobilenet":
            preprocess_function = mobilenet_preprocess_input
        elif base_model_name == "efficientnet":
            preprocess_function = efficientnet_preprocess_input

        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_function,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_function)
    else:
        print("🛠️ Using custom CNN model")
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

    if use_transfer_learning:
        # Use a base model
        if base_model_name == "mobilenet":
            base_model = MobileNetV2(
                input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
            )
        elif base_model_name == "efficientnet":
            base_model = EfficientNetB0(
                input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
            )
        base_model.trainable = False
        inputs = layers.Input(shape=IMAGE_SIZE + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
    else:
        # Use a custom model
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

    initial_epochs = 20
    history = model.fit(data, validation_data=val_data, epochs=initial_epochs, callbacks=[early_stop, reduce_lr])

    if use_transfer_learning:
        print("\n--- Starting Fine-Tuning ---")
        # Unfreeze the base model
        base_model.trainable = True

        # Freeze all layers before the last 20
        fine_tune_at = -20
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Keep BatchNormalization layers in inference mode
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        # Re-compile the model with a very low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        fine_tune_epochs = EPOCHS - initial_epochs
        model.fit(
            data,
            validation_data=val_data,
            epochs=EPOCHS,
            initial_epoch=history.epoch[-1] + 1, # Continue from where we left off
            callbacks=[early_stop, reduce_lr],
        )

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
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Use transfer learning with MobileNetV2 instead of the custom model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mobilenet",
        choices=["mobilenet", "efficientnet"],
        help="Base model to use for transfer learning: 'mobilenet' or 'efficientnet'",
    )

    args = parser.parse_args()

    train(
        args.training,
        args.validation,
        args.test,
        args.output,
        args.transfer_learning,
        args.base_model,
    )


if __name__ == "__main__":
    main()
