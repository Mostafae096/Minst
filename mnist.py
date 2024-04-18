import os.path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_data():
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    # Build the neural network model
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 input images
        layers.Dense(128, activation='relu'),   # Fully connected layer with 128 neurons and ReLU activation
        layers.Dropout(0.2),                    # Dropout layer to reduce overfitting
        layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for each digit) and softmax activation
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, train_images, train_labels, epochs=5):
    # Train the model
    model.fit(train_images, train_labels, epochs=epochs)

def evaluate_model(model, test_images, test_labels):
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

def save_model(model, filename='mnist_model.h5'):
    # Save the trained model
    model.save(filename)
    print(f"Model saved as {filename}")

def load_saved_model(filename='mnist_model.h5'):
    # Load the saved model if it exists, otherwise return None
    if os.path.exists(filename):
        model = models.load_model(filename)
        return model
    else:
        return None

def make_predictions(model, images):
    # Make predictions on the input images
    predictions = model.predict(images)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    return predicted_labels

def load_and_predict_single_image(model, image_path):
    # Load the image and preprocess it
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize pixel values

    # Reshape the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)

    # Make predictions
    predicted_label = make_predictions(model, image_array)
    return predicted_label[0]

# Main function to orchestrate the process
def main():
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Build model
    model = build_model()

    # Load saved model if it exists
    loaded_model = load_saved_model()
    if loaded_model:
        print("Loaded existing model.")
    else:
        print("No existing model found. Training a new model.")
        # Train model
        train_model(model, train_images, train_labels)

        # Save model
        save_model(model)

        # Load the newly trained model
        loaded_model = load_saved_model()

    # Evaluate model
    evaluate_model(loaded_model, test_images, test_labels)

    # Take an example image and predict its label
    image_path = "example_image.png"
    predicted_label = load_and_predict_single_image(loaded_model, image_path)
    print(f"Predicted label for image {image_path}: {predicted_label}")

# Run the main function
if __name__ == "__main__":
    main()
