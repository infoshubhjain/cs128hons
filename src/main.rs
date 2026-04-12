mod activation;
mod layer;
mod network;

use ndarray::Array1;

use activation::ActivationFunction;
use network::Network;

fn main() {
    // 2-input → 4-hidden → 1-output network with sigmoid activation.
    let net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);

    // XOR truth table: [input] => expected output
    let xor_cases: &[([f64; 2], f64)] = &[
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    println!("Forward pass on XOR inputs (untrained — random weights):");
    println!("{:<20} {:<10} {:<10}", "Input", "Expected", "Output");
    println!("{}", "-".repeat(42));

    for (inputs, expected) in xor_cases {
        let input = Array1::from_vec(inputs.to_vec());
        let output = net.forward(&input);
        println!(
            "[{:.0}, {:.0}]{:<12} {:<10.4} {:.4}",
            inputs[0], inputs[1], "", expected, output[0]
        );
    }
}
