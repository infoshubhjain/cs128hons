//! Entry point: interactive CLI for training and predicting with neuro-rust.
//!
//! Presents a menu that lets the user configure learning rate and epoch count,
//! trigger training with live loss and accuracy output, run predictions on the
//! four XOR inputs, or quit. All neural network logic lives in the other modules;
//! this file only handles I/O and wires the pieces together.

mod activation;
mod data;
mod layer;
mod network;

use std::io::{self, Write};

use ndarray::{Array1, Array2};

use activation::ActivationFunction;
use data::xor_data;
use network::Network;

fn main() {
    let mut learning_rate: f64 = 1.0;
    let mut epochs: usize = 10_000;
    let mut net: Option<Network> = None;

    println!("=== neuro-rust: XOR neural network ===");

    loop {
        println!();
        println!("  [1] Train   (lr={learning_rate:.2}, epochs={epochs})");
        println!("  [2] Predict (requires a trained network)");
        println!("  [3] Set learning rate  (current: {learning_rate:.2})");
        println!("  [4] Set epochs         (current: {epochs})");
        println!("  [5] Quit");
        print!("Choice: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => {
                let (inputs, targets_1d) = xor_data();
                let targets = Array2::from_shape_vec((4, 1), targets_1d.to_vec()).unwrap();

                let mut n = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
                println!();
                println!(
                    "Training (2→4→1, sigmoid, lr={learning_rate:.2}, {epochs} epochs)..."
                );
                println!("{:<8} {:<16} Accuracy", "Epoch", "MSE Loss");
                println!("{}", "-".repeat(36));

                let print_every = (epochs / 10).max(1);
                n.train(&inputs, &targets, epochs, learning_rate, print_every);

                net = Some(n);
                println!("Training complete.");
            }
            "2" => {
                let Some(ref n) = net else {
                    println!("No trained network yet — run option 1 first.");
                    continue;
                };

                let xor_cases: &[([f64; 2], f64)] = &[
                    ([0.0, 0.0], 0.0),
                    ([0.0, 1.0], 1.0),
                    ([1.0, 0.0], 1.0),
                    ([1.0, 1.0], 0.0),
                ];

                println!();
                println!("{:<14} {:<12} {:<12} Correct?", "Input", "Expected", "Predicted");
                println!("{}", "-".repeat(48));

                let mut correct = 0usize;
                for (input_arr, expected) in xor_cases {
                    let inp = Array1::from_vec(input_arr.to_vec());
                    let out = n.predict(&inp)[0];
                    let predicted_class = if out >= 0.5 { 1.0 } else { 0.0 };
                    let is_correct = (predicted_class - expected).abs() < 1e-9;
                    if is_correct {
                        correct += 1;
                    }
                    println!(
                        "{:<14} {:<12.4} {:<12.4} {}",
                        format!("[{:.0}, {:.0}]", input_arr[0], input_arr[1]),
                        expected,
                        out,
                        if is_correct { "✓" } else { "✗" },
                    );
                }
                println!();
                println!(
                    "Accuracy: {}/{} ({:.0}%)",
                    correct,
                    xor_cases.len(),
                    correct as f64 / xor_cases.len() as f64 * 100.0
                );
            }
            "3" => {
                print!("New learning rate: ");
                io::stdout().flush().unwrap();
                let mut buf = String::new();
                io::stdin().read_line(&mut buf).unwrap();
                match buf.trim().parse::<f64>() {
                    Ok(v) if v > 0.0 => {
                        learning_rate = v;
                        println!("Learning rate set to {v:.2}");
                    }
                    _ => println!("Invalid value — must be a positive number."),
                }
            }
            "4" => {
                print!("New epoch count: ");
                io::stdout().flush().unwrap();
                let mut buf = String::new();
                io::stdin().read_line(&mut buf).unwrap();
                match buf.trim().parse::<usize>() {
                    Ok(v) if v > 0 => {
                        epochs = v;
                        println!("Epochs set to {v}");
                    }
                    _ => println!("Invalid value — must be a positive integer."),
                }
            }
            "5" | "q" | "quit" => {
                println!("Goodbye.");
                break;
            }
            _ => println!("Unknown option — enter 1–5."),
        }
    }
}
