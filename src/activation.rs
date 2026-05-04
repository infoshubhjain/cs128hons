//! Activation functions used by network layers during forward and backward passes.
//!
//! Each variant implements both the function itself (`apply`) and its derivative
//! (`derivative`), which is required by backpropagation to compute gradients.

/// Supported activation functions for network layers.
///
/// Add new variants here and implement the corresponding arms in `apply` and
/// `derivative` to extend the network with additional activation functions.
#[derive(Clone, Debug)]
pub enum ActivationFunction {
    /// Classic sigmoid: output is always in (0, 1). Good default for XOR.
    Sigmoid,
    /// Rectified linear unit: max(0, x). Faster to compute, no vanishing gradient for x > 0.
    #[allow(dead_code)] // available in the API; the XOR demo uses Sigmoid only
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

    /// Compute the derivative of the activation function at pre-activation value `x`.
    ///
    /// For Sigmoid: σ'(x) = σ(x) · (1 − σ(x))
    /// For ReLU:   f'(x) = 1 if x > 0, else 0
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
