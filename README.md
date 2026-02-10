# Neural Network from Scratch in C 🧠

A raw, mathematical implementation of a Multi-Layer Perceptron (MLP) written in pure C. This project builds a neural network completely from scratch—without TensorFlow, PyTorch, or any external matrix libraries—to demonstrate a deep understanding of the algorithms behind Deep Learning.

## 🚀 Overview

The goal of this project is to demystify the "black box" of AI. It implements the core mathematics of deep learning:
* **Matrix Operations** (Dot products, transposition)
* **Activation Functions** (ReLU)
* **Forward Propagation**
* **Backpropagation** (Gradient Descent)
* **Data Normalization** (Min-Max Scaling)

Currently, the model is trained on a regression dataset to **predict house prices** based on features like number of rooms, area, and age.

## ✨ Key Features

* **Pure C Implementation:** No dependencies (`<math.h>`, `<stdlib.h>`, `<time.h>` only).
* **Custom Backpropagation Engine:** Manually calculated partial derivatives for weight updates.
* **Numerical Stability:** Implements **Xavier-like initialization** and **Target Normalization** to prevent Exploding Gradients.
* **Smart Preprocessing:** Automatic Min-Max scaling for both inputs and targets to ensure convergence.
* **Optimized Architecture:**
    * **Input Layer:** 3 Features (Rooms, Area, Age)
    * **Hidden Layer:** Configurable (Currently 4 neurons with ReLU)
    * **Output Layer:** 1 Neuron

## 🛠️ Installation & Usage

1. Clone the Repository
```bash
git clone [https://github.com/youssofhossam/Neural-Network-from-Scratch-in-C.git](https://github.com/youssofhossam/Neural-Network-from-Scratch-in-C.git)
cd Neural-Network-from-Scratch-in-C
```
2. Compile
Since this is pure C, you only need gcc.

``` Bash
gcc main.c -o neural_net -lm
```
3. Run
``` Bash
./main.c
```

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.