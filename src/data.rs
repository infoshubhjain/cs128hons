use ndarray::{Array1, Array2};

/// Returns the 4 XOR input/output pairs as ndarray types.
///
/// Inputs shape: (4, 2) — each row is one sample.
/// Targets shape: (4,)  — each entry is the expected XOR output.
pub fn xor_data() -> (Array2<f64>, Array1<f64>) {
    let inputs = Array2::from_shape_vec(
        (4, 2),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    )
    .expect("xor input shape is fixed");

    let targets = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

    (inputs, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_data_shape() {
        let (inputs, targets) = xor_data();
        assert_eq!(inputs.shape(), &[4, 2]);
        assert_eq!(targets.len(), 4);
    }

    #[test]
    fn xor_data_values() {
        let (inputs, targets) = xor_data();
        // [0,0]=>0, [0,1]=>1, [1,0]=>1, [1,1]=>0
        let expected_inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let expected_targets = vec![0.0, 1.0, 1.0, 0.0];
        for (i, (ei, et)) in expected_inputs.iter().zip(expected_targets.iter()).enumerate() {
            assert_eq!(inputs[[i, 0]], ei[0]);
            assert_eq!(inputs[[i, 1]], ei[1]);
            assert_eq!(targets[i], *et);
        }
    }
}
