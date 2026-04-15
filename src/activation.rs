/// Supported activation functions for network layers.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
}

impl ActivationFunction {
    /// Apply the activation function element-wise to a scalar value.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::ReLU => x.max(0.0),
        }
    }

    /// Compute the derivative of the activation function at x.
    /// For Sigmoid, x is the pre-activation (z); for ReLU, x is the pre-activation.
    #[allow(dead_code)]
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            ActivationFunction::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        let s = ActivationFunction::Sigmoid.apply(0.0);
        assert!((s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn relu_clips_negatives() {
        assert_eq!(ActivationFunction::ReLU.apply(-3.0), 0.0);
        assert_eq!(ActivationFunction::ReLU.apply(2.5), 2.5);
    }

    #[test]
    fn sigmoid_derivative_at_zero() {
        let d = ActivationFunction::Sigmoid.derivative(0.0);
        assert!((d - 0.25).abs() < 1e-10);
    }
}
