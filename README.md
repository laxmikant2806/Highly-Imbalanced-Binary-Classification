# Production-Ready Fraud Detection Pipeline

This project provides a complete, production-ready pipeline for training and deploying a fraud detection model. It is designed to handle severe class imbalance and is structured for scalability and maintainability.

## Project Structure

The project is organized into the following directories:

-   `config/`: Contains the configuration file (`config.yaml`) for managing all pipeline parameters.
-   `data/`: Intended for storing the raw data (e.g., `creditcard.csv`).
-   `models/`: Stores the trained model and pipeline artifacts.
-   `notebooks/`: For exploratory data analysis and experimentation.
-   `src/`: Contains the main source code, organized into modules:
    -   `app/`: The FastAPI application for serving the model.
    -   `etl/`: The data loading and synthetic data generation logic.
    -   `features/`: The feature engineering pipeline (splitting, scaling, resampling).
    -   `modeling/`: The core machine learning code (model architecture, losses, training, prediction).
    -   `utils/`: Utility functions (config loading, evaluation).
-   `tests/`: Contains unit tests for the project.
-   `Dockerfile`: Defines the Docker container for deploying the application.
-   `requirements.txt`: Lists the Python dependencies for the project.
-   `run_pipeline.sh`: A script to run the entire data processing and training pipeline.
-   `run_docker.sh`: A script to build and run the Docker container for the API.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the data (optional):**
    If you are using the Kaggle credit card fraud dataset, download it and place it in the `data/` directory.
    -   [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

    If the dataset is not found, the pipeline will automatically generate synthetic data based on the settings in `config/config.yaml`.

## Running the Pipeline

To run the entire data processing, training, and evaluation pipeline, execute the following script:

```bash
bash run_pipeline.sh
```

This will:
1.  Load the data (or generate synthetic data).
2.  Split, scale, and resample the data.
3.  Train the fraud detection model.
4.  Evaluate the model and save the results.
5.  Save the trained model and pipeline artifacts to the `models/` directory.

## Running the API with Docker

To build and run the FastAPI application in a Docker container, execute the following script:

```bash
bash run_docker.sh
```

This will:
1.  Build the Docker image.
2.  Run the container, exposing the API on port 8000.

You can then access the API documentation at `http://localhost:8000/docs`.

To make a prediction, you can send a POST request to the `/predict` endpoint with a JSON payload containing the transaction features:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0, 0, ..., 0]}' # Replace with 29 features
```

## Running the Tests

To run the unit tests, use `pytest`:

```bash
pytest
```
