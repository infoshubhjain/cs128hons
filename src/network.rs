use ndarray::{Array1, Array2};

use crate::activation::ActivationFunction;
use crate::layer::Layer;

/// Mean squared error loss: 0.5 * sum((output - target)^2).
/// Returns (loss, gradient dL/d_output).
pub fn mse_loss(output: &Array1<f64>, target: &Array1<f64>) -> (f64, Array1<f64>) {
    let diff = output - target;
    let loss = 0.5 * diff.mapv(|v| v * v).sum();
    (loss, diff)
}

/// A feedforward neural network composed of fully-connected layers.
pub struct Network {
    pub layers: Vec<Layer>,
    pub activation: ActivationFunction,
}

/// Per-layer cache produced by a forward pass, needed for backpropagation.
pub struct LayerCache {
    /// Input to the layer (activation from previous layer, or network input).
    pub input: Array1<f64>,
    /// Pre-activation values z = W·input + b.
    pub z: Array1<f64>,
}

impl Network {
    /// Create a new network with the given layer sizes and activation function.
    ///
    /// `layer_sizes` must have at least two elements: input size and output size.
    /// Hidden layers are specified as intermediate elements, e.g. `&[2, 4, 1]`.
    pub fn new(layer_sizes: &[usize], activation: ActivationFunction) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input and output sizes");
        let layers = layer_sizes
            .windows(2)
            .map(|pair| Layer::new(pair[0], pair[1]))
            .collect();
        Network { layers, activation }
    }

    /// Forward pass returning just the output (used for inference).
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut current = input.clone();
        for layer in &self.layers {
            let z = layer.forward_z(&current);
            current = z.mapv(|v| self.activation.apply(v));
        }
        current
    }

    /// Forward pass that also records per-layer caches for backprop.
    ///
    /// Returns (output, caches) where caches[i] corresponds to layers[i].
    pub fn forward_with_cache(&self, input: &Array1<f64>) -> (Array1<f64>, Vec<LayerCache>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut current = input.clone();

        for layer in &self.layers {
            let z = layer.forward_z(&current);
            let a = z.mapv(|v| self.activation.apply(v));
            caches.push(LayerCache {
                input: current,
                z,
            });
            current = a;
        }

        (current, caches)
    }

    /// Run a forward pass and return the output without updating weights (inference).
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        self.forward(input)
    }

    /// Train the network on a batch of samples for `epochs` iterations.
    ///
    /// `inputs`  — shape (n_samples, input_size)
    /// `targets` — shape (n_samples, output_size)  (one row per sample)
    /// Prints MSE loss every `print_every` epochs (set to 0 to suppress).
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,
        targets: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
        print_every: usize,
    ) {
        let n = inputs.nrows() as f64;
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input_row, target_row) in inputs.rows().into_iter().zip(targets.rows()) {
                let input: Array1<f64> = input_row.to_owned();
                let target: Array1<f64> = target_row.to_owned();

                let (output, caches) = self.forward_with_cache(&input);
                let (loss, grad) = mse_loss(&output, &target);
                total_loss += loss;
                self.backward(&caches, &grad, learning_rate);
            }

            if print_every > 0 && (epoch + 1) % print_every == 0 {
                println!("Epoch {:>5} — MSE loss: {:.6}", epoch + 1, total_loss / n);
            }
        }
    }

    /// Backward pass: given the per-layer caches from `forward_with_cache` and
    /// the upstream gradient of the loss w.r.t. the network output (`d_output`),
    /// compute gradients and update every layer's weights and biases in place.
    ///
    /// Uses vanilla gradient descent with the supplied `learning_rate`.
    ///
    /// Chain rule per layer (working right to left):
    ///   delta  = upstream_grad ⊙ activation'(z)   — error at pre-activation
    ///   dW     = delta ⊗ input                     — weight gradient (outer product)
    ///   db     = delta                              — bias gradient
    ///   next upstream = Wᵀ · delta                 — passed to the layer below
    pub fn backward(
        &mut self,
        caches: &[LayerCache],
        d_output: &Array1<f64>,
        learning_rate: f64,
    ) {
        let mut upstream = d_output.clone();

        for (layer, cache) in self.layers.iter_mut().zip(caches.iter()).rev() {
            // delta = upstream ⊙ activation'(z)
            let delta: Array1<f64> = upstream * cache.z.mapv(|v| layer_activation_deriv(layer, v, &self.activation));

            // Gradient w.r.t. input of this layer (passed upstream to the layer below).
            upstream = layer.weights.t().dot(&delta);

            // Update weights: W -= lr * delta ⊗ input
            // outer product: shape (output_size, input_size)
            for (mut row, &d) in layer.weights.rows_mut().into_iter().zip(delta.iter()) {
                row.scaled_add(-learning_rate * d, &cache.input);
            }

            // Update biases: b -= lr * delta
            layer.biases.scaled_add(-learning_rate, &delta);
        }
    }
}

/// Activation derivative at pre-activation value z. Kept outside Network to avoid
/// borrow conflicts inside the backward loop.
fn layer_activation_deriv(
    _layer: &Layer,
    z: f64,
    activation: &ActivationFunction,
) -> f64 {
    activation.derivative(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn output_shape_matches_final_layer() {
        let net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        let input = Array1::from_vec(vec![0.0, 1.0]);
        let output = net.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn output_is_bounded_for_sigmoid() {
        let net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        for &(a, b) in &[(0.0_f64, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
            let input = Array1::from_vec(vec![a, b]);
            let output = net.forward(&input);
            assert!(output[0] > 0.0 && output[0] < 1.0, "sigmoid output must be in (0,1)");
        }
    }

    #[test]
    fn forward_with_cache_matches_forward() {
        let net = Network::new(&[2, 3, 1], ActivationFunction::Sigmoid);
        let input = array![0.5, -0.5];
        let out1 = net.forward(&input);
        let (out2, caches) = net.forward_with_cache(&input);
        assert!((out1[0] - out2[0]).abs() < 1e-12);
        assert_eq!(caches.len(), 2);
        assert_eq!(caches[0].z.len(), 3);
        assert_eq!(caches[1].z.len(), 1);
    }

    /// Manually verify backward on a 1-layer network (1 input → 1 output, sigmoid).
    ///
    /// Setup:
    ///   W = [[2.0]], b = [0.0]
    ///   input x = [1.0]
    ///   z = 2.0, a = sigmoid(2.0) ≈ 0.8808
    ///   loss = 0.5 * (a - target)²  with target = 0.0
    ///   dL/da = a - target = 0.8808
    ///   dL/dz = dL/da * sigmoid'(z) = 0.8808 * 0.8808 * (1 - 0.8808) ≈ 0.1050
    ///   dL/dW = dL/dz * x = 0.1050
    ///   dL/db = dL/dz         = 0.1050
    ///
    /// After one update with lr=1.0:
    ///   W_new = 2.0 - 0.1050 ≈ 1.8950
    ///   b_new = 0.0 - 0.1050 ≈ -0.1050
    #[test]
    fn backward_updates_weights_correctly() {
        let mut net = Network {
            layers: vec![Layer {
                weights: Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                biases: array![0.0],
            }],
            activation: ActivationFunction::Sigmoid,
        };

        let input = array![1.0];
        let target = array![0.0];

        let (output, caches) = net.forward_with_cache(&input);

        // MSE gradient: dL/da = output - target
        let d_output = &output - &target;

        net.backward(&caches, &d_output, 1.0);

        let a = 1.0_f64 / (1.0 + (-2.0_f64).exp()); // sigmoid(2)
        let dz = a * (1.0 - a);                       // sigmoid'(2)
        let delta = a * dz;                            // (a - 0) * sigmoid'(2)

        let w_expected = 2.0 - delta;   // W -= lr * delta * x,  x=1
        let b_expected = 0.0 - delta;   // b -= lr * delta

        assert!((net.layers[0].weights[[0, 0]] - w_expected).abs() < 1e-10);
        assert!((net.layers[0].biases[0] - b_expected).abs() < 1e-10);
    }

    /// Verify that running backward for many steps drives the loss down on a
    /// trivial 1-sample, 1-output problem where the answer is known.
    #[test]
    fn backward_decreases_loss() {
        let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        let input = array![0.0, 1.0];
        let target = array![1.0];
        let lr = 0.5;

        let initial_loss = {
            let out = net.forward(&input);
            let diff = &out - &target;
            diff.mapv(|v| v * v).sum()
        };

        for _ in 0..500 {
            let (output, caches) = net.forward_with_cache(&input);
            let d_output = &output - &target;
            net.backward(&caches, &d_output, lr);
        }

        let final_loss = {
            let out = net.forward(&input);
            let diff = &out - &target;
            diff.mapv(|v| v * v).sum()
        };

        assert!(final_loss < initial_loss, "loss should decrease after training");
    }

    // ── Dexian: validation and debugging ─────────────────────────────────────

    /// Train on XOR dataset and verify that loss decreases over the full run.
    ///
    /// Numerical note: XOR requires a hidden layer to be linearly separable.
    /// With sigmoid activations and Xavier init, lr=1.0 and 10 000 epochs
    /// reliably converge; lower lr or fewer epochs may not.  Vanishing
    /// gradients are not a practical issue here because the network is shallow
    /// (2 layers) and sigmoid derivatives at the operating point stay above ~0.1.
    #[test]
    fn xor_loss_decreases_after_training() {
        let inputs = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let targets =
            Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);

        let initial_loss: f64 = inputs
            .rows()
            .into_iter()
            .zip(targets.rows())
            .map(|(x, t)| {
                let (loss, _) = mse_loss(&net.forward(&x.to_owned()), &t.to_owned());
                loss
            })
            .sum();

        net.train(&inputs, &targets, 10_000, 1.0, 0);

        let final_loss: f64 = inputs
            .rows()
            .into_iter()
            .zip(targets.rows())
            .map(|(x, t)| {
                let (loss, _) = mse_loss(&net.forward(&x.to_owned()), &t.to_owned());
                loss
            })
            .sum();

        assert!(
            final_loss < initial_loss,
            "XOR loss did not decrease: {final_loss} >= {initial_loss}"
        );
    }

    /// After sufficient training, predict should return values close to XOR targets.
    /// Threshold: outputs for target=0 must be <0.1, outputs for target=1 must be >0.9.
    #[test]
    fn xor_predictions_converge() {
        let xor_cases: &[([f64; 2], f64)] = &[
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];

        let inputs = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let targets =
            Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        net.train(&inputs, &targets, 20_000, 1.0, 0);

        for (input_arr, expected) in xor_cases {
            let input = Array1::from_vec(input_arr.to_vec());
            let out = net.predict(&input)[0];
            if *expected < 0.5 {
                assert!(
                    out < 0.1,
                    "input {:?}: expected ~0 but got {out:.4}",
                    input_arr
                );
            } else {
                assert!(
                    out > 0.9,
                    "input {:?}: expected ~1 but got {out:.4}",
                    input_arr
                );
            }
        }
    }

    /// Verify mse_loss returns correct value and gradient.
    #[test]
    fn mse_loss_value_and_gradient() {
        let output = array![0.8, 0.2];
        let target = array![1.0, 0.0];
        let (loss, grad) = mse_loss(&output, &target);
        // loss = 0.5 * ((0.8-1)^2 + (0.2-0)^2) = 0.5 * (0.04 + 0.04) = 0.04
        assert!((loss - 0.04).abs() < 1e-10, "loss = {loss}");
        assert!((grad[0] - (-0.2)).abs() < 1e-10);
        assert!((grad[1] - 0.2).abs() < 1e-10);
    }
}
