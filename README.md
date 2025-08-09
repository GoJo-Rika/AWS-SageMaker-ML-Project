# AWS SageMaker Machine Learning Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Learning Journey](#learning-journey)
3. [Prerequisites](#prerequisites)
4. [Getting Started](#getting-started)
5. [Project Structure](#project-structure)
6. [Understanding the Code](#understanding-the-code)
7. [Running the Project](#running-the-project)
8. [External Tools and Services](#external-tools-and-services)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Next Steps](#next-steps)

## Project Overview

This project demonstrates how to build, train, and deploy a machine learning model using Amazon SageMaker, AWS's fully managed machine learning service. It implements a mobile phone price classification system using a Random Forest classifier to predict phone price categories based on various features, showcasing a complete end-to-end cloud-based ML workflow.

This project serves as a practical guide to moving beyond local machine learning. You will see how to leverage AWS's powerful infrastructure to orchestrate data preparation, model training, and deployment, all from within a familiar Jupyter Notebook environment. SageMaker handles the heavy lifting of infrastructure, allowing you to focus on the ML logic.

## Learning Journey

This project is designed to take you through a progressive learning experience:

**Foundation Level**: Understand the basic structure of a SageMaker training script (`script.py`) and how the SageMaker Python SDK orchestrates jobs.

**Intermediate Level**: Learn about AWS IAM roles for security, S3 bucket management for data storage, and how SageMaker executes training jobs on dedicated cloud instances.

**Advanced Level**: Explore model deployment, creating real-time inference endpoints, and managing production-ready ML workflows.

## Prerequisites

Before diving in, you should have:

**Technical Knowledge**:
- Basic understanding of Python programming.
- Familiarity with pandas and scikit-learn.
- Foundational machine learning concepts (classification, training/testing splits).
- Basic knowledge of command-line operations.

**AWS Requirements**:
- An active AWS account with permissions to manage SageMaker, S3, and IAM roles.
- AWS CLI installed and configured locally.

**Development Environment**:
- Python 3.8 or higher.
- Jupyter Notebook or JupyterLab.

## Getting Started

### Step 1: Clone the Repository

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/GoJo-Rika/AWS-SageMaker-ML-Project.git
cd AWS-SageMaker-ML-Project
```

### Step 2: Step-by-Step Installation

We recommend using **`uv`**, a fast, next-generation Python package manager, for setup.

#### Recommended Approach (using `uv`)

1.  **Install `uv`** on your system if you haven't already.
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Create a virtual environment and install dependencies** with a single command:
    ```bash
    uv sync
    ```
    This command creates a `.venv` folder and installs all packages from `requirements.txt`.

> **Note**: For a comprehensive guide on `uv`, you can visit this detailed tutorial: [uv-tutorial-guide](https://github.com/GoJo-Rika/uv-tutorial-guide).

#### Alternative Approach (using `venv` and `pip`)

If you prefer to use the standard `venv` and `pip`:

1.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Configure AWS Credentials

Create a `.env` file by copying the sample template. This file will store your AWS-specific configuration.

```bash
cp .env.sample .env
```

Now, edit the `.env` file with your details:
- `AWS_S3_BUCKET_NAME`: A **globally unique** S3 bucket name for storing your data and models.
- `AWS_SAGEMAKER_ROLE`: The ARN of an IAM role with SageMaker execution permissions.

The S3 bucket acts as your data storage layer, while the SageMaker role grants your training jobs the necessary permissions to access resources like S3.

## Project Structure

The project is organized to separate orchestration, training logic, and data.

```
aws-sagemaker/
├── .env.sample                 # Template for environment variables
├── research.ipynb              # Jupyter notebook to orchestrate the entire workflow
├── script.py                   # Core training and inference logic for SageMaker
├── requirements.txt            # Python dependencies
├── research.ipynb              # Jupyter notebook for data exploration
├── train-V-1.csv               # Training dataset
├── test-V-1.csv                # Testing dataset
└── README.md                   # This documentation
```

- **`research.ipynb`**: This is the main control center. You will execute cells here to upload data, start the SageMaker training job, and deploy the model.
- **`script.py`**: This script contains the pure ML code that SageMaker runs on a cloud instance. The notebook submits this script for execution.

## Understanding the Code

### The Orchestrator (`research.ipynb`)

The Jupyter notebook uses the **SageMaker Python SDK**, a high-level library for interacting with AWS services. The key component is the `SKLearn` estimator:

```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role=AWS_SAGEMAKER_ROLE,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    hyperparameters={"n_estimators": 100, "random_state": 0},
)
```
This code block tells SageMaker to:
1.  Run the code in `script.py`.
2.  Use the specified IAM `role` for permissions.
3.  Launch one `ml.m5.large` machine for the job.
4.  Pass `n_estimators` and `random_state` as hyperparameters to the script.

### The Training Script (`script.py`)

The heart of the ML logic resides in `script.py`.

**Argument Parsing**: The script uses `argparse` to receive hyperparameters and essential paths from the SageMaker environment. SageMaker automatically injects environment variables like `SM_MODEL_DIR` and `SM_CHANNEL_TRAIN`, which the script accesses.

```python
# Hyperparameters sent from the notebook
parser.add_argument("--n_estimators", type=int, default=100)
# Special paths provided by the SageMaker environment
parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
```

**Model Persistence**: The script saves the trained model to a specific directory provided by SageMaker. SageMaker then automatically packages this artifact and uploads it to S3.

```python
# The path /opt/ml/model is mapped to the SM_MODEL_DIR environment variable
model_path = Path(args.model_dir) / "model.joblib"
joblib.dump(model, model_path)
```

**Inference Functions**: The `model_fn` is a hook for SageMaker's inference service. When you deploy the model, SageMaker calls this function to load the model into memory.

```python
def model_fn(model_dir: str):
    # Load the model from the disk
    clf = joblib.load(Path(model_dir) / "model.joblib")
    return clf
```
For standard scikit-learn models, this is often the only function needed. For more complex cases, you can also implement `input_fn`, `predict_fn`, and `output_fn` to control data pre-processing and post-processing at the endpoint.

## Running the Project

### Primary Workflow: Jupyter Notebook

The intended way to run this project is to **execute the cells in `research.ipynb` sequentially**. The notebook will guide you through:
1.  Setting up the SageMaker session and S3 bucket.
2.  Exploring the dataset.
3.  Splitting the data and uploading it to S3.
4.  Defining the `SKLearn` estimator.
5.  Launching the training job with a call to `.fit()`.
6.  Deploying the trained model to a real-time endpoint with `.deploy()`.
7.  Making predictions with the endpoint.
8.  Cleaning up by deleting the endpoint.

### Optional: Local Testing

Before running on SageMaker, you can test `script.py` locally to catch bugs quickly. This command simulates the SageMaker environment by providing local paths as arguments.

```bash
# Create a dummy model output directory
mkdir -p model_output

# Run the script, pointing the train/test channels to the current directory
python script.py --train . --test . --model-dir ./model_output
```
This helps ensure the script runs without syntax errors before incurring cloud costs.

## External Tools and Services

### Amazon SageMaker

SageMaker is an ML platform that simplifies the machine learning lifecycle.
- **Training Jobs**: Managed compute instances that execute your `script.py` with the specified data.
- **Model Artifacts**: The `model.tar.gz` file that SageMaker creates and versions in S3.
- **Endpoints**: A fully managed, scalable HTTP endpoint for real-time model inference.

### Amazon S3

S3 is the central data repository. In this project, it stores the raw training/testing datasets and the final model artifacts generated by the SageMaker training job.

### AWS IAM

IAM roles are crucial for security. The role specified in your `.env` file must grant SageMaker permissions to read from your S3 bucket, write logs to CloudWatch, and create training jobs and endpoints.

## Best Practices

- **Separation of Concerns**: The notebook handles orchestration, while `script.py` handles the core ML logic. This makes the code modular and reusable.
- **Environment Management**: Using a `.env` file prevents hardcoding secrets and makes the project portable.
- **Data Versioning**: The project uses specific file names (`train-V-1.csv`), highlighting the importance of versioning both code and data.
- **Reproducibility**: Setting a `random_state` ensures that the model's results are consistent across runs, which is critical for debugging and comparison.

## Troubleshooting

- **Permissions Errors**: `AccessDeniedException` errors usually mean your `AWS_SAGEMAKER_ROLE` is missing permissions. Ensure it has `AmazonSageMakerFullAccess` and `S3FullAccess` (or more restrictive policies) for your project bucket.
- **Data Loading Failures**: Check CloudWatch logs for the training job. `FileNotFoundError` often means the S3 path in the `.fit()` call is incorrect or the data wasn't uploaded properly.
- **Endpoint Failures**: If the endpoint fails to deploy or returns errors, check the CloudWatch logs for the endpoint. This can indicate an issue in your `model_fn` or other inference functions in `script.py`.

## Next Steps

### Immediate Improvements
- **Add Model Evaluation**: Enhance `script.py` to save evaluation metrics (like the classification report) as a JSON file in the model directory, making them easily accessible artifacts.
- **Implement Cross-Validation**: Add k-fold cross-validation inside `script.py` for more robust model evaluation.

### Advanced Features
- **Hyperparameter Tuning**: Use SageMaker's built-in automatic hyperparameter tuning capabilities to find the optimal `n_estimators`.
- **SageMaker Pipelines**: Re-architect the project into a SageMaker Pipeline for a fully automated, multi-step MLOps workflow.
- **Model Monitoring**: Implement data drift and model quality monitoring on the deployed endpoint to detect performance degradation over time.

### Learning Extensions
- **Explore Different Algorithms**: Experiment with other scikit-learn algorithms or deep learning frameworks like `TensorFlow` or `PyTorch`.
- **Distributed Training**: Learn about SageMaker's distributed training capabilities for **handling larger datasets**.
- **MLOps Integration**: Investigate SageMaker Pipelines for creating **end-to-end ML workflows with automated testing and deployment**.

This project provides a solid foundation for understanding cloud-based machine learning workflows. By working through each component systematically, you'll develop the skills necessary to build and deploy production-ready ML systems on AWS. Remember that mastering cloud ML is an iterative process—start with the basics, experiment with different approaches, and gradually incorporate more advanced features as your understanding deepens.