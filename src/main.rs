mod activation;
mod data;
mod layer;
mod network;

use ndarray::{Array1, Array2};

use activation::ActivationFunction;
use data::xor_data;
use network::Network;

fn main() {
    let (inputs_1d, targets_1d) = xor_data();

    // Reshape targets to (4, 1) for train()
    let targets_2d = Array2::from_shape_vec(
        (4, 1),
        targets_1d.to_vec(),
    )
    .unwrap();

    let mut net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);

    println!("Training on XOR dataset (2→4→1, sigmoid, lr=1.0, 10 000 epochs)...");
    net.train(&inputs_1d, &targets_2d, 10_000, 1.0, 1_000);

    println!("\nPredictions after training:");
    println!("{:<20} {:<10} {:<10}", "Input", "Expected", "Predicted");
    println!("{}", "-".repeat(42));

    let xor_cases: &[([f64; 2], f64)] = &[
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    for (input_arr, expected) in xor_cases {
        let input = Array1::from_vec(input_arr.to_vec());
        let out = net.predict(&input)[0];
        println!(
            "[{:.0}, {:.0}]{:<12} {:<10.4} {:.4}",
            input_arr[0], input_arr[1], "", expected, out
        );
    }
}
