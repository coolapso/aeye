#!/bin/python

import argparse
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


# Define image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15  # Increase for better accuracy


def train(dataset: str, validation_data: str, output: str):
    # Load dataset with preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values
        validation_split=0.2  # 80% train, 20% validation
    )

    data = datagen.flow_from_directory(
        dataset, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="training"
    )

    val_data = datagen.flow_from_directory(
        validation_data, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation"
    )

    print("✅ Dataset loaded successfully!")

    # Build CNN Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Binary classification
    ])

    # Compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("✅ Model built successfully!")
    model.summary()

    # Train the model
    model.fit(data, validation_data=val_data, epochs=EPOCHS)

    # Evaluate
    _, acc = model.evaluate(val_data)
    print(f"🎯 Validation Accuracy: {acc:.2f}")

    if ".keras" not in output:
        output = f"{output}.keras"

    model.save(output)
    print("✅ keras model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train the CNN AI model from set of images")

    parser.add_argument(
        '-d',
        '--dataset',
        dest="dataset",
        required=True,
        help='Path to the dataset with training data',
        type=str,
    )

    parser.add_argument(
        '-v',
        '--validation',
        dest="validation",
        help='''Path to dataset to be used as validation data,
if no validation data provided, training data will be used''',
        type=str,
    )

    parser.add_argument(
        '-o',
        '--output',
        dest="output",
        help="Model destination and name",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    if args.validation is None:
        args.validation = args.dataset

    train(args.dataset, args.validation, args.output)


if __name__ == "__main__":
    main()
