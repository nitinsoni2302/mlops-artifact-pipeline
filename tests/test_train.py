# tests/test_train.py
import pytest # The pytest testing framework
import json   # For working with JSON files (like config.json)
import os     # For path manipulations (to find config.json correctly)

# Import functions from src.utils.
# When pytest runs, it often sets the project root as the current working directory,
# so 'src.utils' can be imported like a regular package.
from src.utils import load_config, load_data, create_and_train_model

# Scikit-learn components used for assertions and model creation
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split # Included for completeness, though not directly used in these tests
from sklearn.metrics import accuracy_score

# --- Fixtures for reusable test setup ---

# @pytest.fixture is a decorator that marks a function as a fixture.
# Fixtures provide data or setup/teardown logic for tests.
# scope="module" means this fixture will be run only ONCE for all tests in this test file.
@pytest.fixture(scope="module")
def config_file_path():
    """
    Fixture to determine the correct path to config.json.
    This ensures tests can find the config file regardless of where pytest is run from.
    os.path.dirname(__file__) gets the directory of the current test file (e.g., 'mlops-artifact-pipeline/tests').
    '..' navigates up one directory (to 'mlops-artifact-pipeline/').
    'config', 'config.json' then points to the actual config file.
    """
    return os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')

# autouse=True means this fixture will be automatically used by all tests in this module.
# This fixture creates a dummy config.json before tests run and deletes it afterwards,
# ensuring a clean and consistent state for testing the config loading.
@pytest.fixture(autouse=True)
def setup_and_teardown_dummy_config(config_file_path):
    """
    Fixture to create a temporary dummy config.json for tests and clean it up after.
    This ensures that tests for config loading don't depend on the actual config.json
    being perfectly structured, and it guarantees a clean state.
    """
    dummy_config_content = {
        "C": 1.0, # A float or int
        "solver": "lbfgs", # A string
        "max_iter": 100, # An integer
        "random_state": 42 # An integer
    }
    # Step 1: Create the dummy config file BEFORE any tests in this module run
    with open(config_file_path, 'w') as f: # Open in write mode ('w') to create/overwrite
        json.dump(dummy_config_content, f) # Write the dictionary as JSON

    yield # This is where the actual tests run. Control is yielded to the test functions.

    # Step 2: Clean up (teardown) the dummy config file AFTER all tests are done
    os.remove(config_file_path) # Delete the file

@pytest.fixture(scope="module")
def digits_dataset():
    """
    Fixture to load the digits dataset once for all relevant tests in this module.
    Using a fixture prevents reloading the dataset for every single test, saving time.
    """
    digits = load_digits()
    X = digits.data  # Features (the image data)
    y = digits.target # Labels (the actual digit value)
    return X, y # Return both features and labels

# --- Actual Test Cases ---

# Each function starting with 'test_' is recognized by pytest as a test.
# Fixtures are passed as arguments to the test functions.

def test_config_file_loads(config_file_path):
    """
    Test that the configuration file can be loaded successfully by load_config.
    It checks if the returned object is a dictionary and not empty.
    """
    config = load_config(config_file_path) # Call the function from src.utils
    assert isinstance(config, dict) # Check if the loaded config is indeed a Python dictionary
    assert len(config) > 0 # Ensure the dictionary contains some entries (is not empty)

def test_required_hyperparameters_exist(config_file_path):
    """
    Test that all expected hyperparameters (keys) are present in the loaded configuration.
    This ensures our training script won't fail due to missing parameters.
    """
    config = load_config(config_file_path)
    assert 'C' in config          # Check if 'C' key exists
    assert 'solver' in config     # Check if 'solver' key exists
    assert 'max_iter' in config   # Check if 'max_iter' key exists
    assert 'random_state' in config # Check if 'random_state' key exists

def test_hyperparameter_data_types(config_file_path):
    """
    Test that the values of the hyperparameters have the correct data types.
    This helps prevent runtime errors in the model training if parameters are of the wrong type.
    """
    config = load_config(config_file_path)
    assert isinstance(config['C'], (float, int)) # 'C' should be a float or integer
    assert isinstance(config['solver'], str)     # 'solver' should be a string
    assert isinstance(config['max_iter'], int)   # 'max_iter' should be an integer
    assert isinstance(config['random_state'], int) # 'random_state' should be an integer

def test_model_creation(digits_dataset, config_file_path):
    """
    Test that the create_and_train_model function correctly creates
    and returns an instance of sklearn's LogisticRegression, and that it has been fitted.
    """
    X, y = digits_dataset # Get the dataset (features and labels) from the fixture
    config = load_config(config_file_path) # Load the config (from the dummy one created by fixture)
    model = create_and_train_model(X, y, config) # Call the function that creates and trains the model

    assert isinstance(model, LogisticRegression) # Verify the returned object is a LogisticRegression instance
    # Check if the model has been fitted (i.e., it has learned coefficients)
    assert hasattr(model, 'coef_') # 'coef_' attribute stores the learned coefficients after fitting
    assert hasattr(model, 'intercept_') # 'intercept_' attribute stores the learned intercept after fitting
    assert hasattr(model, 'n_iter_') # 'n_iter_' attribute shows the number of iterations taken for fitting

def test_model_accuracy(digits_dataset, config_file_path):
    """
    Test that the trained model achieves a reasonable (basic) accuracy on the training data.
    This is a simple sanity check to ensure the model isn't completely broken.
    """
    X, y = digits_dataset
    config = load_config(config_file_path)
    model = create_and_train_model(X, y, config)
    predictions = model.predict(X) # Make predictions on the training data
    accuracy = accuracy_score(y, predictions) # Calculate the accuracy by comparing true labels (y) to predictions

    # The digits dataset is relatively easy to classify.
    # An accuracy above 0.95 (95%) is generally expected for a basic Logistic Regression model on this data.
    print(f"\nTest Model Training Accuracy: {accuracy:.4f}") # Print accuracy in test output for debugging
    assert accuracy > 0.95 # Assert that the accuracy is greater than the specified threshold