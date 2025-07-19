# src/inference.py
import joblib # For loading the trained model (joblib is commonly used for model serialization)
from sklearn.datasets import load_digits # To get the same dataset (Digits dataset)
from sklearn.metrics import accuracy_score # To evaluate predictions (optional, but good for verification)

def main():
    """
    This script loads a pre-trained model and performs inference on the digits dataset.
    It follows these steps:
    1. Loads the trained model (model_train.pkl).
    2. Loads the digits dataset (the same one used for training).
    3. Generates predictions using the loaded model.
    4. Evaluates the accuracy of the predictions (to verify the model is working).
    """
    print("Starting model inference pipeline...")

    # 1. Load the trained model
    # The model_train.pkl file is expected to be in the root directory
    # where the script is executed from (or where the artifact is downloaded).
    try:
        model = joblib.load('model_train.pkl')
        print("Trained model loaded successfully from model_train.pkl")
    except FileNotFoundError:
        # If the model file is not found, print an error and exit.
        # This is important for CI/CD to fail if the dependency isn't met.
        print("Error: model_train.pkl not found. Make sure the training pipeline ran and uploaded it.")
        exit(1) # Exit with a non-zero code to indicate an error

    # 2. Load the digits dataset (using the same dataset for inference as training for simplicity)
    digits = load_digits()
    X_inference = digits.data # Features for inference
    y_true = digits.target    # True labels (for evaluation of accuracy)
    print(f"Inference dataset loaded: X shape {X_inference.shape}, y shape {y_true.shape}")

    # 3. Generate predictions using the loaded model
    predictions = model.predict(X_inference)
    print("Predictions generated.")

    # 4. Evaluate predictions (optional, but good for verifying the model's performance)
    inference_accuracy = accuracy_score(y_true, predictions)
    # --- ADDED THESE TWO LINES ---
    print(f"Inference Accuracy: {inference_accuracy:.4f}") # Display the calculated accuracy
    print("Inference pipeline finished.") # Indicate the completion of the pipeline


if __name__ == "__main__":
    # This block ensures that the main() function is called only when the script is executed directly,
    # not when it's imported as a module into another script.
    main()