mod activation;
mod data;
mod layer;
mod network;

use ndarray::Array1;

use activation::ActivationFunction;
use data::xor_data;
use network::Network;

fn main() {
    // 2-input → 4-hidden → 1-output network with sigmoid activation.
    let net = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);

    let (inputs, targets) = xor_data();

    println!("Forward pass on XOR inputs (untrained — random weights):");
    println!("{:<20} {:<10} {:<10}", "Input", "Expected", "Output");
    println!("{}", "-".repeat(42));

    for (i, (input_row, &expected)) in inputs.rows().into_iter().zip(targets.iter()).enumerate() {
        let input: Array1<f64> = input_row.to_owned();
        let output = net.forward(&input);
        println!(
            "[{:.0}, {:.0}]{:<12} {:<10.4} {:.4}",
            inputs[[i, 0]], inputs[[i, 1]], "", expected, output[0]
        );
    }
}
