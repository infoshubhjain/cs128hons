use ndarray::{Array1, Array2};
use rand::Rng;

/// A single fully-connected layer: weights (out × in) and biases (out).
#[derive(Clone, Debug)]
pub struct Layer {
    /// Weight matrix, shape [output_size, input_size].
    pub weights: Array2<f64>,
    /// Bias vector, shape [output_size].
    pub biases: Array1<f64>,
}

impl Layer {
    /// Create a new layer with Xavier-initialised weights and zero biases.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier initialisation: uniform in [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
        let limit = (6.0_f64 / (input_size + output_size) as f64).sqrt();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.random_range(-limit..limit)
        });
        let biases = Array1::zeros(output_size);

        Layer { weights, biases }
    }

    /// Compute the pre-activation (z = W·x + b) for an input vector.
    pub fn forward_z(&self, input: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(input) + &self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn forward_z_output_shape() {
        let layer = Layer::new(3, 2);
        let input = Array1::from_vec(vec![1.0, 0.5, -1.0]);
        let z = layer.forward_z(&input);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn forward_z_known_weights_and_bias() {
        // Manually construct a layer with known weights and biases.
        // weights = [[1, 2], [3, 4]], biases = [0.5, -0.5]
        // input = [1, 1]
        // expected z = [1*1 + 2*1 + 0.5, 3*1 + 4*1 + (-0.5)] = [3.5, 6.5]
        let layer = Layer {
            weights: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            biases: array![0.5, -0.5],
        };
        let input = array![1.0, 1.0];
        let z = layer.forward_z(&input);
        assert!((z[0] - 3.5).abs() < 1e-10);
        assert!((z[1] - 6.5).abs() < 1e-10);
    }
}
