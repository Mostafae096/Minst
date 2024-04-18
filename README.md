# Minst Handwritten Number Recognition Model

This is a model for recognizing handwritten numbers using the MNIST dataset and TensorFlow's Keras library. It's developed as part of a school project.

## Usage

### Prerequisites

Make sure you have Python installed along with the necessary libraries:

```bash
pip install tensorflow matplotlib pillow
```

## Explanation

- **load_data**: Loads and preprocesses the MNIST dataset.
- **build_model**: Constructs the neural network model.
- **train_model**: Trains the model using the training data.
- **evaluate_model**: Evaluates the model's performance on the test data.
- **save_model**: Saves the trained model to a file.
- **load_saved_model**: Loads a previously saved model if available.
- **make_predictions**: Makes predictions on input images.
- **load_and_predict_single_image**: Loads and predicts a single image.
- **main**: Orchestrates the entire process, including loading data, building the model, training or loading a saved model, evaluating the model, and predicting a single image.