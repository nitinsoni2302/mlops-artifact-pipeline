# src/train.py
# Import functions from our utility module
from src.utils import load_config, load_data, create_and_train_model, save_model
from sklearn.metrics import accuracy_score # To evaluate model performance

def main():
    """
    This is the main function that executes the entire training pipeline.
    It follows these steps:
    1. Loads hyperparameters from config.json.
    2. Loads the dataset.
    3. Trains the Logistic Regression model.
    4. Evaluates the model's accuracy (optional, for feedback).
    5. Saves the trained model to a file.
    """
    print("Starting model training pipeline...")

    # 1. Load hyperparameters from config/config.json
    config = load_config() # Call the function from utils.py
    print(f"Loaded configuration: {config}")

    # 2. Load the digits dataset
    X, y = load_data() # Call the function from utils.py
    print(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")

    # 3. Train a LogisticRegression model with parameters from config
    model = create_and_train_model(X, y, config) # Call the function from utils.py
    print("Model training complete.")

    # Evaluate the model on the training data (optional, but good for feedback)
    # This helps confirm the model learned something
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Training Accuracy: {accuracy:.4f}") # Print the accuracy, formatted to 4 decimal places

    # 4. Save the model as model_train.pkl
    save_model(model, 'model_train.pkl') # Call the function from utils.py
    print("Training pipeline finished.")

# This block ensures that main() is called only when the script is executed directly
if __name__ == "__main__":
    main()