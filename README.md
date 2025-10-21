## 1. Project Setup

Goal: Prepare a clean structure for modular development.

Create a directory structure:

```
mlp_project/
├── data/
│   ├── dataset.csv
├── src/
│   ├── data_split.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
├── saved_models/
├── plots/
├── requirements.txt
└── README.md
```

Allowed libraries: `numpy`, `matplotlib`, `pandas`, `argparse`.

## 2. Data Preparation (`data_split.py`)

Goal: Prepare the dataset for training and validation.

Steps:

- Load the CSV (Breast Cancer Wisconsin dataset).
- Explore columns and check for missing values.
- Encode the target:

  - `M` → `1`, `B` → `0`

- Normalize the features (mean 0, std 1).
- Split into:

  - 80% training (`data_train.csv`)
  - 20% validation (`data_valid.csv`)

- Save both splits.

Deliverable:

```
python data_split.py --dataset data/wdbc.csv --train_ratio 0.8 --seed 42
```

## 3. Model Architecture (`model.py`)

Goal: Define the class structure for the MLP.

Main classes:

- `DenseLayer`: manages weights, bias, activation, and forward/backward passes.
- Activation functions: `Sigmoid`, `Tanh`, `ReLU`, `Softmax`.
- `MLP`: list of layers + `forward()` and `backward()` + `update_weights()`.

Components to implement:

- Initialization of weights (He or Xavier).
- Feedforward computation.
- Backpropagation with gradient descent.
- Loss functions:

  - Binary cross-entropy for final evaluation.
  - Mean loss per epoch for training.

## 4. Training Logic (`train.py`)

Goal: Train the model on training data and evaluate on validation data.

Features:

- Parse command-line arguments:

```
python train.py --layers 30 24 24 1 --activation sigmoid --epochs 80 \
                --lr 0.03 --batch_size 8 --loss binary_crossentropy
```

- Loop over epochs:

  - Shuffle batches.
  - Compute loss & accuracy on training and validation sets.
  - Print metrics at each epoch.

- Save learned weights:

  - `saved_models/model_weights.npy`

Plot:

- Training vs Validation Loss
- Training vs Validation Accuracy

## 5. Prediction Program (`predict.py`)

Goal: Use the saved model to predict on a test dataset.

Steps:

- Load the model structure and weights.
- Run forward propagation on test data.
- Output predicted probabilities and labels.
- Compute binary cross-entropy and accuracy.

Example:

```
python predict.py --dataset data_valid.csv --model saved_models/model_weights.npy
```

## 6. Visualization & Evaluation

Goal: Display metrics and curves.

- Plot learning curves (loss, accuracy) with `matplotlib`.
- Optionally compare models if you implement multiple training runs.

---

If you'd like, I can also add an example `requirements.txt`, minimal example stubs for each `src/` file, or generate the training & plotting code the README references. Let me know which next step you want me to take.