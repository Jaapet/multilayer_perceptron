## 1. Project Setup --------------- OK

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

## 2. Data Preparation (`data_split.py`) --------------- OK

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

### 3.1 Dense Layer Implementation

Goal: Represent one fully connected layer in the network.

#### Components:

1. **Weights and Biases**
   - Initialize using:
     - Xavier/Glorot initialization for tanh/sigmoid
     - He initialization for ReLU
   
2. **Activation Function**
   - Layer stores reference to activation function
   
3. **Forward Pass**
   - Compute $Z = XW + b$
   - Apply activation: $A = activation(Z)$
   
4. **Backward Pass**
   - Compute gradients: dW, db, dX
   - Store gradients for weight update

#### Class Skeleton:
```python
class DenseLayer:
    def __init__(self, input_size, output_size, activation, initialization="he"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.initialization = initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier or He initialization

    def forward(self, X):
        # Compute Z = X W + b
        # Apply activation
        # Store X and Z for backward

    def backward(self, dA):
        # Compute gradients: dW, db, dX
        # Using chain rule
        # Return dX to propagate backward
```

### 3.2 Activation Functions

Goal: Compute both forward and derivative for backpropagation.

| Activation | Forward | Backward/Derivative |
|------------|---------|-------------------|
| Sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x)$ | $1-\tanh^2(x)$ |
| ReLU | $\max(0,x)$ | 1 if x>0 else 0 |
| Softmax | $\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}$ | Use derivative in cross-entropy combined |

Implementation tips:
- Each can be a class or function returning both forward and derivative
- For Softmax, subtract max before exponentiating for numerical stability

### 3.3 MLP Implementation

Goal: Represent the full network as a sequence of layers.

#### Components:

1. **Layer Management**
   - Store list of DenseLayer objects
   
2. **Forward Pass**
   - Chain: X → layer1.forward() → layer2.forward() → ... → output
   
3. **Backward Pass**
   - Chain: dLoss → last_layer.backward() → ... → first_layer.backward()
   
4. **Weight Updates**
   - W = W - lr * dW
   - b = b - lr * db

#### Class Skeleton:
```python
class MLP:
    def __init__(self, layers_config, learning_rate=0.01):
        self.layers = []
        for config in layers_config:
            layer = DenseLayer(**config)
            self.layers.append(layer)
        self.learning_rate = learning_rate

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dLoss):
        dA = dLoss
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_weights(self):
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
```

### 3.4 Loss Functions

Binary Cross-Entropy (for binary classification):

$L = -\frac{1}{m}\sum_i y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)$

Gradient with respect to predictions:

$dA = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$

### 3.5 Weight Initialization

1. **He Initialization** (for ReLU):
   $W \sim N(0, \sqrt{\frac{2}{n_{in}}})$

2. **Xavier Initialization** (for Sigmoid/Tanh):
   $W \sim N(0, \sqrt{\frac{1}{n_{in}}})$

Biases can be initialized to 0.

### 3.6 Implementation Tips

1. Keep layer and activation classes separate
2. Each class should store its forward pass info for backprop
3. Test components independently:
   - Forward of one layer
   - Backward of one layer
   - Activation derivatives
4. Start with small network (1 hidden layer) and small dataset for debugging

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