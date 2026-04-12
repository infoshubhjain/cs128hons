use ndarray::Array1;

use crate::activation::ActivationFunction;
use crate::layer::Layer;

/// A feedforward neural network composed of fully-connected layers.
pub struct Network {
    pub layers: Vec<Layer>,
    pub activation: ActivationFunction,
}

impl Network {
    /// Build a network from a slice of layer sizes.
    ///
    /// `layer_sizes[0]` is the input dimension; the last entry is the output
    /// dimension.  Every consecutive pair creates one `Layer`.
    ///
    /// # Example
    /// ```
    /// // 2-input, 4-hidden, 1-output network
    /// let net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
    /// ```
    pub fn new(layer_sizes: &[usize], activation: ActivationFunction) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input and output sizes");

        let layers = layer_sizes
            .windows(2)
            .map(|pair| Layer::new(pair[0], pair[1]))
            .collect();

        Network { layers, activation }
    }

    /// Run a forward pass through the network and return the output vector.
    ///
    /// Activation is applied after every layer including the output layer.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut current = input.clone();
        for layer in &self.layers {
            let z = layer.forward_z(&current);
            current = z.mapv(|v| self.activation.apply(v));
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
