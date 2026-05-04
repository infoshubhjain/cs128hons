//! The feedforward neural network: forward pass, backpropagation, and training loop.
//!
//! `Network` owns a stack of `Layer`s and an `ActivationFunction`. Training runs
//! stochastic gradient descent one sample at a time: for each epoch every sample
//! is forward-passed, loss is accumulated, and weights are updated via `backward`.

use ndarray::{Array1, Array2};

use crate::activation::ActivationFunction;
use crate::layer::Layer;

/// Mean squared error loss: 0.5 · Σ(output − target)².
///
/// Returns `(loss, dL/d_output)` where the gradient is simply `output − target`.
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
    /// Input to the layer (activation from the previous layer, or the network input).
    pub input: Array1<f64>,
    /// Pre-activation values z = W·input + b.
    pub z: Array1<f64>,
}

impl Network {
    /// Create a new network with the given layer sizes and activation function.
    ///
    /// `layer_sizes` must have at least two elements (input and output size).
    /// Hidden layers are specified as intermediate elements, e.g. `&[2, 4, 1]`.
    pub fn new(layer_sizes: &[usize], activation: ActivationFunction) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input and output sizes");
        let layers = layer_sizes
            .windows(2)
            .map(|pair| Layer::new(pair[0], pair[1]))
            .collect();
        Network { layers, activation }
    }

    /// Run a forward pass and return the final output (used for inference and testing).
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut current = input.clone();
        for layer in &self.layers {
            let z = layer.forward_z(&current);
            current = z.mapv(|v| self.activation.apply(v));
        }
        current
    }

    /// Forward pass that also records per-layer caches needed for backprop.
    ///
    /// Returns `(output, caches)` where `caches[i]` corresponds to `layers[i]`.
    pub fn forward_with_cache(&self, input: &Array1<f64>) -> (Array1<f64>, Vec<LayerCache>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut current = input.clone();

        for layer in &self.layers {
            let z = layer.forward_z(&current);
            let a = z.mapv(|v| self.activation.apply(v));
            caches.push(LayerCache { input: current, z });
            current = a;
        }

        (current, caches)
    }

    /// Train the network for `epochs` iterations, printing loss and accuracy every
    /// `print_every` epochs. Set `print_every` to `0` to suppress all output.
    ///
    /// - `inputs`  — shape `(n_samples, input_size)`
    /// - `targets` — shape `(n_samples, output_size)`, one row per sample
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,
        targets: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
        print_every: usize,
    ) {
        let n = inputs.nrows();
        let n_f64 = n as f64;

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
                // Compute accuracy: threshold output at 0.5 for binary classification.
                let correct: usize = inputs
                    .rows()
                    .into_iter()
                    .zip(targets.rows())
                    .filter(|(x, t)| {
                        let out = self.predict(&x.to_owned());
                        (out[0].round() - t[0].round()).abs() < 1e-9
                    })
                    .count();
                println!(
                    "{:<8} {:<16.6} {}/{} ({:.0}%)",
                    epoch + 1,
                    total_loss / n_f64,
                    correct,
                    n,
                    correct as f64 / n_f64 * 100.0,
                );
            }
        }
    }

    /// Backward pass: compute gradients via the chain rule and update weights in place.
    ///
    /// Given the per-layer caches from `forward_with_cache` and the loss gradient
    /// w.r.t. the network output (`d_output`), works right-to-left through the layers:
    ///
    /// ```text
    ///   delta          = upstream ⊙ σ'(z)      — error at pre-activation
    ///   dW             = delta ⊗ input          — weight gradient (outer product)
    ///   db             = delta                  — bias gradient
    ///   next upstream  = Wᵀ · delta             — propagate to the layer below
    /// ```
    pub fn backward(
        &mut self,
        caches: &[LayerCache],
        d_output: &Array1<f64>,
        learning_rate: f64,
    ) {
        let mut upstream = d_output.clone();

        for (layer, cache) in self.layers.iter_mut().zip(caches.iter()).rev() {
            // delta = upstream ⊙ σ'(z): element-wise product of upstream gradient
            // and the activation derivative evaluated at the pre-activation z.
            let delta: Array1<f64> =
                upstream * cache.z.mapv(|v| activation_deriv(&self.activation, v));

            // Pass the gradient to the layer below before we update weights,
            // because we still need the current weight matrix for this multiply.
            upstream = layer.weights.t().dot(&delta);

            // W -= lr · (delta ⊗ input)  — outer product row by row
            for (mut row, &d) in layer.weights.rows_mut().into_iter().zip(delta.iter()) {
                row.scaled_add(-learning_rate * d, &cache.input);
            }

            // b -= lr · delta
            layer.biases.scaled_add(-learning_rate, &delta);
        }
    }
}

/// Return the activation derivative σ'(z). Factored out of `backward` to avoid
/// a simultaneous mutable borrow of `self.layers` and shared borrow of `self.activation`.
fn activation_deriv(activation: &ActivationFunction, z: f64) -> f64 {
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
        let output = net.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn output_is_bounded_for_sigmoid() {
        let net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        for &(a, b) in &[(0.0_f64, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
            let input = Array1::from_vec(vec![a, b]);
            let output = net.predict(&input);
            assert!(output[0] > 0.0 && output[0] < 1.0, "sigmoid output must be in (0,1)");
        }
    }

    #[test]
    fn forward_with_cache_matches_predict() {
        let net = Network::new(&[2, 3, 1], ActivationFunction::Sigmoid);
        let input = array![0.5, -0.5];
        let out1 = net.predict(&input);
        let (out2, caches) = net.forward_with_cache(&input);
        assert!((out1[0] - out2[0]).abs() < 1e-12);
        assert_eq!(caches.len(), 2);
        assert_eq!(caches[0].z.len(), 3);
        assert_eq!(caches[1].z.len(), 1);
    }

    /// Manually verify backward on a 1-layer network (1 input → 1 output, sigmoid).
    ///
    /// Setup:
    ///   W = [[2.0]], b = [0.0], input x = [1.0], target = 0.0
    ///   z = 2.0,  a = σ(2) ≈ 0.8808
    ///   dL/da = a − 0 = 0.8808
    ///   dL/dz = dL/da · σ'(2) = a · (a − 0) · (1 − a) ≈ 0.1050
    ///   dL/dW = dL/dz · x = 0.1050,  dL/db = 0.1050
    ///   After lr=1: W_new ≈ 1.8950,  b_new ≈ −0.1050
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
        let d_output = &output - &target;
        net.backward(&caches, &d_output, 1.0);

        let a = 1.0_f64 / (1.0 + (-2.0_f64).exp()); // σ(2)
        let dz = a * (1.0 - a);                       // σ'(2)
        let delta = a * dz;                            // (a − 0) · σ'(2)

        assert!((net.layers[0].weights[[0, 0]] - (2.0 - delta)).abs() < 1e-10);
        assert!((net.layers[0].biases[0] - (0.0 - delta)).abs() < 1e-10);
    }

    #[test]
    fn backward_decreases_loss() {
        let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        let input = array![0.0, 1.0];
        let target = array![1.0];
        let lr = 0.5;

        let initial_loss = {
            let out = net.predict(&input);
            let diff = &out - &target;
            diff.mapv(|v| v * v).sum()
        };

        for _ in 0..500 {
            let (output, caches) = net.forward_with_cache(&input);
            let d_output = &output - &target;
            net.backward(&caches, &d_output, lr);
        }

        let final_loss = {
            let out = net.predict(&input);
            let diff = &out - &target;
            diff.mapv(|v| v * v).sum()
        };

        assert!(final_loss < initial_loss, "loss should decrease after training");
    }

    /// Train on XOR and verify loss decreases over 10 000 epochs.
    ///
    /// XOR is not linearly separable, so the hidden layer is load-bearing.
    /// Xavier init + sigmoid + lr=1.0 converges reliably at this epoch count.
    #[test]
    fn xor_loss_decreases_after_training() {
        let inputs = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let targets = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);

        let initial_loss: f64 = inputs
            .rows()
            .into_iter()
            .zip(targets.rows())
            .map(|(x, t)| {
                let (loss, _) = mse_loss(&net.predict(&x.to_owned()), &t.to_owned());
                loss
            })
            .sum();

        net.train(&inputs, &targets, 10_000, 1.0, 0);

        let final_loss: f64 = inputs
            .rows()
            .into_iter()
            .zip(targets.rows())
            .map(|(x, t)| {
                let (loss, _) = mse_loss(&net.predict(&x.to_owned()), &t.to_owned());
                loss
            })
            .sum();

        assert!(
            final_loss < initial_loss,
            "XOR loss did not decrease: {final_loss} >= {initial_loss}"
        );
    }

    /// After 20 000 epochs, outputs must be within 0.1 of the XOR targets.
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
        let targets = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        net.train(&inputs, &targets, 20_000, 1.0, 0);

        for (input_arr, expected) in xor_cases {
            let input = Array1::from_vec(input_arr.to_vec());
            let out = net.predict(&input)[0];
            if *expected < 0.5 {
                assert!(out < 0.1, "input {input_arr:?}: expected ~0 but got {out:.4}");
            } else {
                assert!(out > 0.9, "input {input_arr:?}: expected ~1 but got {out:.4}");
            }
        }
    }

    /// Verify mse_loss returns correct value and gradient.
    #[test]
    fn mse_loss_value_and_gradient() {
        let output = array![0.8, 0.2];
        let target = array![1.0, 0.0];
        let (loss, grad) = mse_loss(&output, &target);
        // loss = 0.5 * ((0.8−1)² + (0.2−0)²) = 0.5 * 0.08 = 0.04
        assert!((loss - 0.04).abs() < 1e-10, "loss = {loss}");
        assert!((grad[0] - (-0.2)).abs() < 1e-10);
        assert!((grad[1] - 0.2).abs() < 1e-10);
    }
}
