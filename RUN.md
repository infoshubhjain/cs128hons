# Running neuro-rust

## Prerequisites

You need Rust and Cargo installed. If you don't have them:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Tested with rustc 1.93.0 / cargo 1.93.0. Any stable release from 1.70+ should work.

## Clone and run

```
git clone https://github.com/cs128-2025c/neuro-rust-shubhj3-jl321-dexian2.git
cd neuro-rust-shubhj3-jl321-dexian2
cargo run
```

Cargo will download dependencies on the first run. Expected output:

```
Training on XOR dataset (2→4→1, sigmoid, lr=1.0, 10 000 epochs)...
Epoch  1000 — MSE loss: ...
Epoch  2000 — MSE loss: ...
...
Epoch 10000 — MSE loss: ...

Predictions after training:
Input                Expected   Predicted
------------------------------------------
[0, 0]              0.0000     ~0.0
[0, 1]              1.0000     ~1.0
[1, 0]              1.0000     ~1.0
[1, 1]              0.0000     ~0.0
```

Loss should decrease each interval. Final predictions should be close to 0 or 1.

## Run tests

```
cargo test
```

All 15 tests should pass. The slowest test (`xor_predictions_converge`) trains for 20 000 epochs and takes a few seconds.

## Run lints

```
cargo clippy
```

Should produce no warnings.
