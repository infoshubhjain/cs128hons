# Running neuro-rust

## Prerequisites

You need Rust and Cargo installed. If you don't have them:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Tested with rustc 1.93.0 / cargo 1.93.0. Any stable release from 1.70+ should work.

## Clone and run

```
git clone https://github.com/infoshubhjain/cs128hons.git
cd cs128hons
cargo run
```

Cargo will download dependencies on the first run. You will see an interactive menu:

```
=== neuro-rust: XOR neural network ===

  [1] Train   (lr=1.00, epochs=10000)
  [2] Predict (requires a trained network)
  [3] Set learning rate  (current: 1.00)
  [4] Set epochs         (current: 10000)
  [5] Quit
Choice:
```

**Typical session:**

1. Press `1` then Enter to train with the defaults. Output during training:

```
Training (2→4→1, sigmoid, lr=1.00, 10000 epochs)...
Epoch    MSE Loss         Accuracy
------------------------------------
1000     0.002079         4/4 (100%)
2000     0.000530         4/4 (100%)
...
10000    0.000063         4/4 (100%)
Training complete.
```

Loss decreases each interval. Accuracy reaches 4/4 (100%) within the first few hundred epochs
and stays there. Exact loss values vary between runs due to random weight initialisation.

2. Press `2` then Enter to see predictions:

```
Input          Expected     Predicted    Correct?
------------------------------------------------
[0, 0]         0.0000       0.0099       ✓
[0, 1]         1.0000       0.9872       ✓
[1, 0]         1.0000       0.9923       ✓
[1, 1]         0.0000       0.0136       ✓

Accuracy: 4/4 (100%)
```

Predicted values will differ slightly each run but should always be within 0.1 of the target.

3. Press `5` to quit.

You can also press `3` to change the learning rate or `4` to change the epoch count before
training. The menu always shows the current values.

## Run tests

```
cargo test
```

All 15 tests should pass. The slowest test (`xor_predictions_converge`) trains for 20 000 epochs
and takes a few seconds.

## Run lints

```
cargo clippy
```

Should produce no warnings.
