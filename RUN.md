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

Cargo will download dependencies on the first run. You will see an interactive menu:

```
=== neuro-rust: XOR neural network ===

  [1] Train   (lr=1, epochs=10000)
  [2] Predict (requires a trained network)
  [3] Set learning rate  (current: 1)
  [4] Set epochs         (current: 10000)
  [5] Quit
Choice:
```

**Typical session:**

1. Press `1` then Enter to train with the defaults. Output during training:

```
Training (2→4→1, sigmoid, lr=1, 10000 epochs)...
Epoch    MSE Loss         Accuracy
------------------------------------
1000     0.123456         2/4 (50%)
2000     0.054321         3/4 (75%)
...
10000    0.002345         4/4 (100%)
Training complete.
```

Loss should decrease each interval. Accuracy should reach 4/4 (100%) by the end.

2. Press `2` then Enter to see predictions:

```
Input           Expected     Predicted    Correct?
--------------------------------------------------
[0, 0]          0.0000       0.0123       ✓
[0, 1]          1.0000       0.9876       ✓
[1, 0]          1.0000       0.9854       ✓
[1, 1]          0.0000       0.0145       ✓

Accuracy: 4/4 (100%)
```

3. Press `5` to quit.

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
