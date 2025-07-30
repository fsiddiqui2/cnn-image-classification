# Cat-Dog Image Classification with PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying images of cats and dogs. The project includes scripts for data loading, model definition, training, and testing.

## Project Structure

The repository is organized into the following key files:

*   `dataset.py`: Contains the `CatDogDataset` class, a custom PyTorch `Dataset` for loading and splitting the cat and dog image data. It handles the creation of training, validation, and test sets.
*   `model.py`: Defines the `CNN` class, which is the neural network architecture used for classification.
*   `train_model.py`: The main script for training the CNN model. It handles data loading, model initialization, training loops, validation, and saving the best-performing model.
*   `test_model.py`: A script to evaluate a trained model on the test dataset.

## Dataset

This project uses the "Dogs vs. Cats" dataset, which can be downloaded from Kaggle. The dataset should be organized in a directory named `PetImages` with the following structure:

```
PetImages/
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

The `CatDogDataset` class in `dataset.py` is configured to work with this directory structure.

## Installation

To run this project, you need to have Python and PyTorch installed. You can set up your environment by following these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fsiddiqui2/cnn-image-classification.git
    cd cnn-image-classification
    ```

2.  **Install the required libraries:**
    ```bash
    pip install torch torchvision tqdm numpy pandas scikit-learn torchmetrics
    ```

## Usage

### Training the Model

You can train the model by running the `train_model.py` script. The script accepts several command-line arguments to configure the training process.

**Basic Usage:**

```bash
python train_model.py
```

This will run the training with default parameters and save the model and results in a directory named based on the hyperparameters (e.g., `CNN_lr0.001_bs128_ep20_seed42`).

**Command-Line Arguments:**

*   `--save-dir`: (Optional) Specify a directory to save the trained model, training history, and results. If not provided, a directory name is generated based on hyperparameters.
*   `--lr`: Learning rate for the optimizer (default: `1e-3`).
*   `--epochs`: Number of training epochs (default: `20`).
*   `--batch-size`: Batch size for training and validation (default: `128`).
*   `--data-seed`: Random seed for the train-validation-test split to ensure reproducibility (default: `42`).

**Example with custom arguments:**

```bash
python train_model.py --lr 0.0001 --batch-size 64 --epochs 30 --save-dir my_experiment
```

### Testing the Model

To evaluate a trained model, use the `test_model.py` script. You need to provide the directory where the trained model is saved.

**Usage:**

```bash
python test_model.py --save-dir 
```

**Command-Line Arguments:**

*   `--save-dir`: **(Required)** The directory containing the `best_model.pth` file.
*   `--batch-size`: Batch size for testing (default: `128`).
*   `--data-seed`: Random seed for the data split, which should match the one used during training (default: `42`).

**Example:**

```bash
python test_model.py --save-dir CNN_lr0.001_bs128_ep20_seed42 --data-seed 42
```

## Model Architecture

The `CNN` model defined in `model.py` is a sequential network consisting of three convolutional blocks followed by two fully connected (linear) layers.

Each convolutional block includes:
*   A 2D Convolutional layer (`nn.Conv2d`)
*   A Max Pooling layer (`nn.MaxPool2d`)
*   A Batch Normalization layer (`nn.BatchNorm2d`)
*   A ReLU activation function

The output from the convolutional blocks is flattened and passed through the linear layers to produce the final classification output.

## Results

The model was trained and evaluated using three different random seeds for the data split (`0`, `1`, and `42`) to ensure the robustness of the results. The performance of the best model on the test set for each seed is summarized below. The default hyperparameters used for these runs were a learning rate of `1e-3`, a batch size of `128`, and training for `20` epochs.

### Test Set Performance

| Data Seed | Test Loss | Test Accuracy | Test F1-Score | Test AUROC |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.3465 | 84.78% | 0.8462 | 0.9270 |
| 1 | 0.3492 | 86.11% | 0.8621 | 0.9336 |
| 42 | 0.3414 | 85.31% | 0.8526 | 0.9308 |

The results demonstrate consistent performance across different data splits, with test accuracy ranging from approximately 84.8% to 86.1%. The F1-scores and AUROC values are also stable, indicating that the model is not overly sensitive to the specific train-test split and generalizes well to unseen data.

