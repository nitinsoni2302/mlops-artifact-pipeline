# src/train.py
import os # NEW: Import the os module for path manipulation

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
    # Determine the absolute path to the project root
    # os.path.dirname(__file__) gets the directory of the current script (src/)
    # '..' goes up one level to the project root
    # os.path.join correctly concatenates path components for different OS
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_file_path = os.path.join(project_root, 'config', 'config.json')

    try:
        config = load_config(config_file_path) # Pass the determined path to load_config
        print(f"Loaded configuration: {config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}. Please ensure it exists.")
        exit(1) # Exit if config is not found, as it's critical for training


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
    # The model will be saved in the project root because the script is run with -m src.train
    save_model(model, 'model_train.pkl') # Call the function from utils.py
    print("Training pipeline finished.")

# This block ensures that main() is called only when the script is executed directly
if __name__ == "__main__":
    main()