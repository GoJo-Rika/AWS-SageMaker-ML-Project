import argparse
import os
from io import StringIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)


# SageMaker's default inference handler calls this function to load a model.
# The 'model_dir' argument points to the directory where the model artifacts are saved.
# This function should return the loaded model.
def model_fn(model_dir: str) -> None:
    """
    Loads the trained model from the specified directory.
    """
    print("Loading model from an endpoint.")
    # The model is saved as 'model.joblib' in the main training script.
    clf = joblib.load(Path(model_dir) / "model.joblib")
    return clf


# This block is executed when the script is run as the main program,
# which is what SageMaker does to start a training job.
if __name__ == "__main__":
    print("[Info] Extracting arguments")
    parser = argparse.ArgumentParser()

    ## --- Hyperparameters ---
    # These are arguments that you can tune to improve your model's performance.
    # They are passed from the SageMaker Estimator in the notebook.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    ## --- SageMaker Environment Variables ---
    # These arguments are automatically provided by the SageMaker environment.
    # They specify the paths to input data, output directories, and model storage.
    # SM_MODEL_DIR: A directory where the trained model artifacts should be saved.
    # SM_CHANNEL_TRAIN: The directory containing the training data.
    # SM_CHANNEL_TEST: The directory containing the testing data.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    # The parse_known_args() method is used to ignore any other arguments
    # that SageMaker might pass to the script.
    args, _ = parser.parse_known_args()

    # Print version information for debugging and reproducibility.
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    # Construct file paths using the directories provided by SageMaker.
    train_df = pd.read_csv(Path(args.train) / args.train_file)
    test_df = pd.read_csv(Path(args.test) / args.test_file)

    # Assume the last column is the target variable (label).
    features = list(train_df.columns)
    label = features.pop(-1)

    print("Building training and testing datasets")
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print(f"Column order: {features}")
    print(f"Label column: {label}")
    print("Data Shape:")
    print(f"---- SHAPE OF TRAINING DATA (85%) ----: {X_train.shape}")
    print(f"---- SHAPE OF TESTING DATA (15%) ----: {X_test.shape}")

    print("Training RandomForest Model ....")
    # Initialize the model with hyperparameters passed from the command line.
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        verbose=2,  # Print tree-building progress.
        n_jobs=1,   # Use all available CPU cores.
    )

    model.fit(X_train, y_train)

    # --- Model Persistence ---
    # After training, the model must be saved to the 'model_dir' path.
    # SageMaker will take the contents of this directory, create a 'model.tar.gz' archive,
    # and save it to the specified S3 location.
    model_path = Path(args.model_dir) / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    # --- Model Evaluation ---
    # Evaluate the model on the test set and print the results.
    # These metrics will be visible in the CloudWatch logs for the training job.
    print("Evaluating model on test data...")
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print("\n---- METRICS RESULTS FOR TESTING DATA ----")
    print(f"Total Rows: {X_test.shape[0]}")
    print(f"[TESTING] Model Accuracy: {test_acc}")
    print("[TESTING] Classification Report: \n")
    print(test_rep)
