use ndarray::Array1;

use crate::activation::ActivationFunction;
use crate::layer::Layer;

/// A feedforward neural network composed of fully-connected layers.
pub struct Network {
    pub layers: Vec<Layer>,
    pub activation: ActivationFunction,
}

/// Per-layer cache produced by a forward pass, needed for backpropagation.
#[allow(dead_code)]
pub struct LayerCache {
    /// Input to the layer (activation from previous layer, or network input).
    pub input: Array1<f64>,
    /// Pre-activation values z = W·input + b.
    pub z: Array1<f64>,
    /// Post-activation values a = activation(z).
    pub a: Array1<f64>,
}

impl Network {
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
    #[allow(dead_code)]
    pub fn forward_with_cache(&self, input: &Array1<f64>) -> (Array1<f64>, Vec<LayerCache>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut current = input.clone();

        for layer in &self.layers {
            let z = layer.forward_z(&current);
            let a = z.mapv(|v| self.activation.apply(v));
            caches.push(LayerCache {
                input: current,
                z: z.clone(),
                a: a.clone(),
            });
            current = a;
        }

        (current, caches)
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
    #[allow(dead_code)]
    pub fn backward(
        &mut self,
        caches: &[LayerCache],
        d_output: &Array1<f64>,
        learning_rate: f64,
    ) {
        let mut upstream = d_output.clone();

        for (layer, cache) in self.layers.iter_mut().zip(caches.iter()).rev() {
            // delta = upstream ⊙ activation'(z)
            let delta: Array1<f64> = upstream * cache.z.mapv(|v| layer_activation_deriv(&layer, v, &self.activation));

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

/// Helper: activation derivative at pre-activation value z, dispatched on the
/// network's activation function. Kept outside Network to avoid borrow conflicts
/// inside the backward loop.
#[allow(dead_code)]
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
        assert_eq!(caches[0].a.len(), 3);
        assert_eq!(caches[1].a.len(), 1);
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
}
