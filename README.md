# cs128hons
## neuro-rust

### Group Members
- Shubh (shubhj3)
- Jessica (jl321)
- Dexian (dexian2)

---

## Project Introduction

neuro-rust is a feedforward neural network library built from scratch in Rust. The project implements core neural network components including forward propagation, backpropagation, and gradient descent, trained and validated on the XOR problem.

XOR is a classic benchmark for neural networks because it is the simplest problem that cannot be solved by a single-layer perceptron, requiring at least one hidden layer to learn a non-linear decision boundary. By targeting XOR, we can focus on getting the math and architecture right without the overhead of large dataset management.

Our goals are to build a modular, well-documented neural network that correctly learns XOR, and to deepen our understanding of how neural networks function at a low level by implementing everything ourselves rather than relying on high-level ML frameworks.

---

## Technical Overview

The project has four major components.

**Matrix Operations Layer** — We will use the ndarray crate for matrix and vector math. This layer handles all the linear algebra that forward and backward passes depend on, including matrix multiplication, element-wise operations, and broadcasting.

**Network Architecture** — A Network struct will hold a vector of Layer structs. Each Layer contains a weight matrix and a bias vector. The network will support configurable layer sizes at construction time, so the user can specify input size, hidden layer sizes, and output size.

**Forward and Backward Pass** — Forward propagation will compute activations layer by layer using a configurable activation function (sigmoid to start, with the option to add ReLU). Backpropagation will compute gradients via the chain rule, propagating error from the output layer back through each hidden layer. Gradient descent will update weights and biases using a configurable learning rate.

**Training and Evaluation** — A training loop will run forward pass, compute loss (mean squared error), run backpropagation, and update weights for a configurable number of epochs. We will print loss at intervals so we can verify the network is converging. After training, a predict function will run the forward pass on new inputs and return the output.

---

## Checkpoint Goals

### Checkpoint 1 (April 13-17)
- Project compiles and runs with `cargo run`
- ndarray integrated and working
- Layer and Network structs defined with configurable sizes
- Forward pass implemented and producing output on hardcoded XOR inputs
- Basic unit tests for matrix operations and forward pass

### Checkpoint 2 (April 27-May 1)
- Backpropagation fully implemented
- Gradient descent updating weights and biases each epoch
- Training loop running on XOR dataset
- Loss decreasing over epochs (network is learning)
- Predict function returns correct XOR outputs after training

### Final Submission (May 6)
- Code cleaned up, documented, and well-commented
- CLI interface for running training and prediction
- RUN.md complete and verified
- Presentation video recorded

---

## Possible Challenges

**Backpropagation correctness.** Getting the chain rule right across multiple layers is error-prone. A small bug in gradient computation can cause the network to silently fail to learn, making debugging difficult.

**Numerical stability.** Sigmoid activation can cause vanishing gradients. We may need to experiment with learning rates and initialization strategies to get reliable convergence even on a simple problem like XOR.

**Rust ownership and ndarray.** Matrix operations involve a lot of borrowing and cloning. Figuring out when to clone vs borrow ndarray matrices while keeping the borrow checker happy will take some iteration.

**Testing neural network code.** Unlike deterministic algorithms, neural networks involve randomness (weight initialization) and approximate outputs. Writing meaningful tests that verify correctness without being brittle is a challenge we will need to work through.

---

## References

- Neural Networks and Deep Learning (Michael Nielsen) for backpropagation math
- ndarray crate documentation
- CS128 Honors course project ideas list for inspiration
- 3Blue1Brown Neural Network series for visual intuition on forward/backward passes
