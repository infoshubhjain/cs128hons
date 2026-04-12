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

        // Xavier initialisation: uniform in [-limit, limit]
        // where limit = sqrt(6 / (fan_in + fan_out))
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
    use ndarray::array;

    // Output shape

    #[test]
    fn forward_z_output_shape() {
        let layer = Layer::new(3, 2);
        let input = Array1::from_vec(vec![1.0, 0.5, -1.0]);
        let z = layer.forward_z(&input);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn weight_matrix_has_correct_shape() {
        let layer = Layer::new(4, 3);
        assert_eq!(layer.weights.shape(), &[3, 4]);
        assert_eq!(layer.biases.len(), 3);
    }

    // Known-value tests

    /// Identity weights + constant bias: W = I, b = [1, 2], x = [3, 4]
    /// Expected z = [3+1, 4+2] = [4, 6]
    #[test]
    fn forward_z_known_weights_and_bias() {
        let mut layer = Layer::new(2, 2);
        layer.weights = array![[1.0, 0.0], [0.0, 1.0]];
        layer.biases = array![1.0, 2.0];

        let z = layer.forward_z(&array![3.0, 4.0]);
        assert!((z[0] - 4.0).abs() < 1e-10, "expected z[0]=4.0, got {}", z[0]);
        assert!((z[1] - 6.0).abs() < 1e-10, "expected z[1]=6.0, got {}", z[1]);
    }

    /// Single output neuron: w = [2, -1], b = 0.5, x = [1, 3]
    /// Expected z = 2*1 + (-1)*3 + 0.5 = -0.5
    #[test]
    fn forward_z_single_output_neuron() {
        let mut layer = Layer::new(2, 1);
        layer.weights = array![[2.0, -1.0]];
        layer.biases = array![0.5];

        let z = layer.forward_z(&array![1.0, 3.0]);
        assert_eq!(z.len(), 1);
        assert!((z[0] - (-0.5)).abs() < 1e-10, "expected -0.5, got {}", z[0]);
    }

    /// Zero input must return exactly the bias vector.
    #[test]
    fn forward_z_zero_input_returns_bias() {
        let mut layer = Layer::new(3, 2);
        layer.biases = array![7.0, -3.0];

        let z = layer.forward_z(&Array1::zeros(3));
        assert!((z[0] - 7.0).abs() < 1e-10);
        assert!((z[1] - (-3.0)).abs() < 1e-10);
    }

    //Xavier initialisation
    #[test]
    fn xavier_weights_are_within_limit() {
        let input_size = 4;
        let output_size = 6;
        let limit = (6.0_f64 / (input_size + output_size) as f64).sqrt();
        let layer = Layer::new(input_size, output_size);

        for &w in layer.weights.iter() {
            assert!(
                w >= -limit && w < limit,
                "weight {} outside Xavier limit ±{:.4}",
                w, limit
            );
        }
    }

    #[test]
    fn biases_initialised_to_zero() {
        let layer = Layer::new(4, 6);
        for &b in layer.biases.iter() {
            assert_eq!(b, 0.0);
        }
    }
}