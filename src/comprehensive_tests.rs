mod comprehensive_tests {
    use crate::activation::ActivationFunction;
    use crate::data::xor_data;
    use crate::layer::Layer;
    use crate::network::{mse_loss, Network};
    use ndarray::{array, Array1, Array2};




    // ACTIVATION FUNCTION TESTS
   

  
    fn test_sigmoid_boundary_cases() {
        let sigmoid = ActivationFunction::Sigmoid;

        // Test: Sigmoid at zero should be 0.5
        let s_zero = sigmoid.apply(0.0);
        assert!((s_zero - 0.5).abs() < 1e-10, "sigmoid(0) should be 0.5");

        // Test: Sigmoid at large positive value approaches 1
        let s_positive = sigmoid.apply(10.0);
        assert!(s_positive > 0.99, "sigmoid(10) should approach 1");

        // Test: Sigmoid at large negative value approaches 0
        let s_negative = sigmoid.apply(-10.0);
        assert!(s_negative < 0.01, "sigmoid(-10) should approach 0");

        // Test: Sigmoid is symmetric around 0.5
        let s_pos = sigmoid.apply(2.0);
        let s_neg = sigmoid.apply(-2.0);
        assert!((s_pos + s_neg - 1.0).abs() < 1e-10, "sigmoid(x) + sigmoid(-x) should equal 1");
    }

  
    fn test_sigmoid_derivative() {
        let sigmoid = ActivationFunction::Sigmoid;

        // Test: Derivative at zero should be 0.25 (0.5 * 0.5)
        let d_zero = sigmoid.derivative(0.0);
        assert!((d_zero - 0.25).abs() < 1e-10, "sigmoid derivative at 0 should be 0.25");

        // Test: Derivative is always non-negative
        for x in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
            let d = sigmoid.derivative(*x);
            assert!(d >= 0.0, "sigmoid derivative should always be non-negative");
        }

        // Test: Derivative approaches 0 at extremes
        let d_extreme_pos = sigmoid.derivative(20.0);
        let d_extreme_neg = sigmoid.derivative(-20.0);
        assert!(d_extreme_pos < 1e-8, "sigmoid derivative at large positive should be near 0");
        assert!(d_extreme_neg < 1e-8, "sigmoid derivative at large negative should be near 0");
    }

  
    fn test_relu_basic_behavior() {
        let relu = ActivationFunction::ReLU;

        // Test: ReLU clips negative values to 0
        assert_eq!(relu.apply(-5.0), 0.0);
        assert_eq!(relu.apply(-0.5), 0.0);
        assert_eq!(relu.apply(0.0), 0.0);

        // Test: ReLU passes positive values unchanged
        assert_eq!(relu.apply(0.5), 0.5);
        assert_eq!(relu.apply(3.7), 3.7);
        assert_eq!(relu.apply(100.0), 100.0);
    }

  
    fn test_relu_derivative() {
        let relu = ActivationFunction::ReLU;

        // Test: Derivative is 0 for negative values
        assert_eq!(relu.derivative(-5.0), 0.0);
        assert_eq!(relu.derivative(-0.1), 0.0);

        // Test: Derivative is 1 for positive values
        assert_eq!(relu.derivative(0.1), 1.0);
        assert_eq!(relu.derivative(5.0), 1.0);
        assert_eq!(relu.derivative(100.0), 1.0);

        // Test: Derivative at zero is 0 (convention)
        assert_eq!(relu.derivative(0.0), 0.0);
    }

   
    // LAYER TESTS
   

  
    fn test_layer_construction_and_shape() {
        // Test: Layer with input size 5, output size 3
        let layer = Layer::new(5, 3);
        assert_eq!(layer.weights.shape(), &[3, 5], "weights shape should be [output, input]");
        assert_eq!(layer.biases.shape(), &[3], "biases shape should be [output]");

        // Test: Xavier initialization produces reasonable values
        let max_weight = layer.weights.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let limit = (6.0 / (5 + 3) as f64).sqrt();
        assert!(
            max_weight <= limit,
            "Xavier initialization should bound weights by limit"
        );

        // Test: Biases are initialized to zero
        for b in &layer.biases {
            assert_eq!(*b, 0.0, "biases should be initialized to 0");
        }
    }

  
    fn test_layer_forward_z_with_known_values() {
        // Test: forward_z with manually specified weights and biases
        let layer = Layer {
            weights: Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
            biases: array![0.5, -0.5],
        };
        let input = array![1.0, 2.0, 3.0];
        let z = layer.forward_z(&input);

        // Expected: [1*1 + 2*2 + 3*3 + 0.5, 4*1 + 5*2 + 6*3 - 0.5]
        //         = [1 + 4 + 9 + 0.5, 4 + 10 + 18 - 0.5]
        //         = [14.5, 31.5]
        assert!((z[0] - 14.5).abs() < 1e-10);
        assert!((z[1] - 31.5).abs() < 1e-10);
    }

  
    fn test_layer_forward_z_output_shape() {
        // Test: Output shape matches layer output size
        let layer = Layer::new(10, 4);
        let input = Array1::ones(10);
        let z = layer.forward_z(&input);
        assert_eq!(z.len(), 4, "output size should match layer output size");
    }

  
    fn test_layer_forward_z_with_different_sizes() {
        // Test: 1x1 layer
        let layer_1x1 = Layer::new(1, 1);
        let input_1 = array![2.0];
        let z_1 = layer_1x1.forward_z(&input_1);
        assert_eq!(z_1.len(), 1);

        // Test: 100x10 layer
        let layer_large = Layer::new(100, 10);
        let input_large = Array1::ones(100);
        let z_large = layer_large.forward_z(&input_large);
        assert_eq!(z_large.len(), 10);

        // Test: 5x2 layer
        let layer_5x2 = Layer::new(5, 2);
        let input_5 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let z_5 = layer_5x2.forward_z(&input_5);
        assert_eq!(z_5.len(), 2);
    }

   
    // MSE LOSS FUNCTION TESTS
   

  
    fn test_mse_loss_perfect_prediction() {
        // Test: MSE loss is 0 when output matches target
        let output = array![0.5, 0.3, 0.8];
        let target = array![0.5, 0.3, 0.8];
        let (loss, _) = mse_loss(&output, &target);
        assert!(loss.abs() < 1e-10, "loss should be 0 for perfect predictions");
    }

  
    fn test_mse_loss_computation() {
        // Test: MSE loss is correctly computed
        // Loss = 0.5 * [(1-0)^2 + (0-1)^2] = 0.5 * [1 + 1] = 1.0
        let output = array![1.0, 0.0];
        let target = array![0.0, 1.0];
        let (loss, _) = mse_loss(&output, &target);
        assert!((loss - 1.0).abs() < 1e-10, "loss should be 1.0 for [1,0] vs [0,1]");
    }

  
    fn test_mse_loss_gradient() {
        // Test: MSE loss gradient is output - target
        let output = array![0.7, 0.3];
        let target = array![0.5, 0.6];
        let (_, grad) = mse_loss(&output, &target);
        assert!((grad[0] - 0.2).abs() < 1e-10, "gradient should be output - target");
        assert!((grad[1] - (-0.3)).abs() < 1e-10);
    }

  
    fn test_mse_loss_single_output() {
        // Test: MSE loss with single output (binary classification)
        let output = array![0.8];
        let target = array![1.0];
        let (loss, grad) = mse_loss(&output, &target);
        assert!((loss - 0.02).abs() < 1e-10, "0.5 * (0.8-1.0)^2 = 0.02");
        assert!((grad[0] - (-0.2)).abs() < 1e-10, "gradient should be 0.8 - 1.0 = -0.2");
    }

   
    // NETWORK CONSTRUCTION TESTS
   

  
    fn test_network_single_layer() {
        // Test: Network with single layer (no hidden layers)
        let network = Network::new(&[2, 1], ActivationFunction::Sigmoid);
        assert_eq!(network.layers.len(), 1, "single layer network should have 1 layer");
        assert_eq!(network.layers[0].weights.shape(), &[1, 2]);
    }

  
    fn test_network_multiple_layers() {
        // Test: Network with multiple hidden layers
        let network = Network::new(&[2, 4, 3, 1], ActivationFunction::Sigmoid);
        assert_eq!(network.layers.len(), 3, "3-layer network should have 3 layer structs");

        // Verify each layer's shape
        assert_eq!(network.layers[0].weights.shape(), &[4, 2], "first layer: 2->4");
        assert_eq!(network.layers[1].weights.shape(), &[3, 4], "second layer: 4->3");
        assert_eq!(network.layers[2].weights.shape(), &[1, 3], "third layer: 3->1");
    }

  
    fn test_network_with_relu_activation() {
        // Test: Network can be constructed with ReLU activation
        let network = Network::new(&[5, 10, 1], ActivationFunction::ReLU);
        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].weights.shape(), &[10, 5]);
    }

   
    // NETWORK FORWARD PASS TESTS
   

  
    fn test_network_predict_output_shape() {
        // Test: predict returns correct output shape
        let network = Network::new(&[3, 5, 2], ActivationFunction::Sigmoid);
        let input = Array1::ones(3);
        let output = network.predict(&input);
        assert_eq!(output.len(), 2, "output size should match final layer size");
    }

  
    fn test_network_predict_bounds_sigmoid() {
        // Test: With sigmoid activation, outputs are bounded in (0, 1)
        let network = Network::new(&[4, 8, 1], ActivationFunction::Sigmoid);
        let input = Array1::from_vec(vec![10.0, -10.0, 0.0, 5.0]);
        let output = network.predict(&input);
        assert!(output[0] > 0.0 && output[0] < 1.0, "sigmoid output should be in (0,1)");
    }

  
    fn test_network_forward_with_cache_returns_caches() {
        // Test: forward_with_cache returns per-layer caches
        let network = Network::new(&[2, 3, 1], ActivationFunction::Sigmoid);
        let input = array![0.5, 0.5];
        let (output, caches) = network.forward_with_cache(&input);

        assert_eq!(output.len(), 1, "output should have size 1");
        assert_eq!(caches.len(), 2, "should have 2 caches for 2 layers");

        // Check first cache
        assert_eq!(caches[0].input.len(), 2, "first cache input is network input");
        assert_eq!(caches[0].z.len(), 3, "first cache z matches first layer output");

        // Check second cache
        assert_eq!(caches[1].input.len(), 3, "second cache input comes from first layer");
        assert_eq!(caches[1].z.len(), 1, "second cache z matches second layer output");
    }

   
    // TRAINING LOOP TESTS
   

  
    fn test_network_train_single_epoch() {
        // Test: Network can run training for a single epoch without errors
        let mut network = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        let inputs = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();

        // Should not panic
        network.train(&inputs, &targets, 1, 0.5, 0);
    }

  
    fn test_network_train_multiple_epochs() {
        // Test: Training for multiple epochs
        let mut network = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        let inputs = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();

        // Should not panic
        network.train(&inputs, &targets, 100, 0.5, 0);
    }

  
    fn test_network_train_with_different_learning_rates() {
        // Test: Training with various learning rates
        let learning_rates = vec![0.01, 0.1, 0.5, 1.0, 2.0];
        let inputs = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();

        for lr in learning_rates {
            let mut network = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
            // Should not panic with any reasonable learning rate
            network.train(&inputs, &targets, 10, lr, 0);
        }
    }


    // XOR PROBLEM TESTS



    fn test_xor_data_loading() {
        // Test: XOR data has correct shape and values
        let (inputs, targets) = xor_data();
        assert_eq!(inputs.shape(), &[4, 2], "XOR inputs should be 4x2");
        assert_eq!(targets.len(), 4, "XOR targets should have 4 elements");
    }

    fn test_xor_data_correctness() {
        // Test: XOR data contains correct input-output pairs
        let (inputs, targets) = xor_data();
        // [0,0]=>0, [0,1]=>1, [1,0]=>1, [1,1]=>0
        assert_eq!(inputs[[0, 0]], 0.0);
        assert_eq!(inputs[[0, 1]], 0.0);
        assert_eq!(targets[0], 0.0, "[0,0] should map to 0");

        assert_eq!(inputs[[1, 0]], 0.0);
        assert_eq!(inputs[[1, 1]], 1.0);
        assert_eq!(targets[1], 1.0, "[0,1] should map to 1");

        assert_eq!(inputs[[2, 0]], 1.0);
        assert_eq!(inputs[[2, 1]], 0.0);
        assert_eq!(targets[2], 1.0, "[1,0] should map to 1");

        assert_eq!(inputs[[3, 0]], 1.0);
        assert_eq!(inputs[[3, 1]], 1.0);
        assert_eq!(targets[3], 0.0, "[1,1] should map to 0");
    }

    fn test_network_learns_xor_with_hidden_layer() {
        // Test: Network can learn XOR after training

        let (inputs, targets_1d) = xor_data();
        let targets = Array2::from_shape_vec((4, 1), targets_1d.to_vec()).unwrap();

        // Create network: 2 inputs -> 4 hidden -> 1 output
        let mut network = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);

        network.train(&inputs, &targets, 5000, 1.0, 0);

        // After training, check if predictions are reasonable
        // We use a threshold of 0.5 for binary classification
        let mut correct = 0;
        for i in 0..4 {
            let input = inputs.row(i).to_owned();
            let output = network.predict(&input);
            let pred = if output[0] > 0.5 { 1.0 } else { 0.0 };
            let target = targets[[i, 0]];
            if (pred - target).abs() < 1e-9 {
                correct += 1;
            }
        }

        // After 5000 epochs with reasonable learning rate, should solve XOR
        assert!(
            correct >= 3,
            "Network should correctly predict at least 3/4 XOR cases after training"
        );
    }


    fn test_network_xor_all_cases_individually() {
        // Test: Check network's predictions on each XOR case separately
        let (inputs, targets_1d) = xor_data();
        let targets = Array2::from_shape_vec((4, 1), targets_1d.to_vec()).unwrap();

        let mut network = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        network.train(&inputs, &targets, 5000, 1.0, 0);

        // Test case: [0, 0] => 0
        let pred_00 = network.predict(&inputs.row(0).to_owned());
        let is_pred_00_correct = pred_00[0] < 0.5; // Should be close to 0
        println!(
            "Network predicts [0,0] as {:.3} (expected ~0.0)",
            pred_00[0]
        );

        // Test case: [0, 1] => 1
        let pred_01 = network.predict(&inputs.row(1).to_owned());
        let is_pred_01_correct = pred_01[0] > 0.5; // Should be close to 1
        println!(
            "Network predicts [0,1] as {:.3} (expected ~1.0)",
            pred_01[0]
        );

        // Test case: [1, 0] => 1
        let pred_10 = network.predict(&inputs.row(2).to_owned());
        let is_pred_10_correct = pred_10[0] > 0.5; // Should be close to 1
        println!(
            "Network predicts [1,0] as {:.3} (expected ~1.0)",
            pred_10[0]
        );

        // Test case: [1, 1] => 0
        let pred_11 = network.predict(&inputs.row(3).to_owned());
        let is_pred_11_correct = pred_11[0] < 0.5; // Should be close to 0
        println!(
            "Network predicts [1,1] as {:.3} (expected ~0.0)",
            pred_11[0]
        );

        let all_correct = is_pred_00_correct && is_pred_01_correct && is_pred_10_correct && is_pred_11_correct;
        assert!(
            all_correct,
            "Network should correctly predict all 4 XOR cases"
        );
    }


    fn test_shallow_network_cannot_learn_xor() {
        // Test: A single-layer network cannot learn XOR (demonstrating why hidden layers are needed)
        let (inputs, targets_1d) = xor_data();
        let targets = Array2::from_shape_vec((4, 1), targets_1d.to_vec()).unwrap();

        // Single layer: 2 inputs -> 1 output (no hidden layer)
        let mut network = Network::new(&[2, 1], ActivationFunction::Sigmoid);

        // Train extensively
        network.train(&inputs, &targets, 10000, 1.0, 0);

        // Count correct predictions
        let mut correct = 0;
        for i in 0..4 {
            let input = inputs.row(i).to_owned();
            let output = network.predict(&input);
            let pred = if output[0] > 0.5 { 1.0 } else { 0.0 };
            let target = targets[[i, 0]];
            if (pred - target).abs() < 1e-9 {
                correct += 1;
            }
        }

        // Single-layer network should fail to learn XOR perfectly
        assert!(
            correct < 4,
            "Single-layer network should not perfectly learn XOR (should fail: linear separator cannot solve XOR)"
        );
    }


    // EDGE CASES AND ROBUSTNESS TESTS
  

   
    fn test_network_with_large_input_values() {
        // Test: Network handles large input values without numerical issues
        let network = Network::new(&[2, 3, 1], ActivationFunction::Sigmoid);
        let input = array![1000.0, -1000.0];
        let output = network.predict(&input);
        assert!(output[0].is_finite(), "output should not be NaN or Inf");
        assert!(output[0] >= 0.0 && output[0] <= 1.0, "sigmoid output should be in [0,1]");
    }


    fn test_network_with_zero_inputs() {
        // Test: Network handles all-zero inputs
        let network = Network::new(&[4, 3, 1], ActivationFunction::Sigmoid);
        let input = Array1::zeros(4);
        let output = network.predict(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }


    fn test_network_training_with_identical_samples() {
        // Test: Network can train on duplicate samples
        let mut network = Network::new(&[2, 3, 1], ActivationFunction::Sigmoid);
        let inputs = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).unwrap();
        let targets = Array2::from_shape_vec((3, 1), vec![0.0, 0.0, 0.0]).unwrap();

        network.train(&inputs, &targets, 100, 0.5, 0);
        let output = network.predict(&inputs.row(0).to_owned());
        assert!(output[0].is_finite());
    }

   
    fn test_different_network_architectures() {
        // Test: Various network architectures can be created and trained
        let (inputs, targets_1d) = xor_data();
        let targets = Array2::from_shape_vec((4, 1), targets_1d.to_vec()).unwrap();

        let architectures = vec![
            vec![2, 2, 1],  // Minimal hidden layer
            vec![2, 8, 1],  // Wider hidden layer
            vec![2, 3, 3, 1], // Two hidden layers
            vec![2, 5, 5, 5, 1], // Three hidden layers
        ];

        for arch in architectures {
            let mut network = Network::new(&arch, ActivationFunction::Sigmoid);
            network.train(&inputs, &targets, 1000, 0.5, 0);
            let output = network.predict(&inputs.row(0).to_owned());
            assert!(output[0].is_finite(), "architecture {:?} failed", arch);
        }
    }

 
    fn test_sigmoid_vs_relu_on_xor() {
        // Test: Compare sigmoid and relu performance on XOR
        let (inputs, targets_1d) = xor_data();
        let targets = Array2::from_shape_vec((4, 1), targets_1d.to_vec()).unwrap();

        // Network with sigmoid
        let mut net_sigmoid = Network::new(&[2, 4, 1], ActivationFunction::Sigmoid);
        net_sigmoid.train(&inputs, &targets, 5000, 1.0, 0);

        // Network with relu
        let mut net_relu = Network::new(&[2, 4, 1], ActivationFunction::ReLU);
        net_relu.train(&inputs, &targets, 5000, 1.0, 0);

        // Both should produce finite outputs
        let out_sigmoid = net_sigmoid.predict(&inputs.row(0).to_owned());
        let out_relu = net_relu.predict(&inputs.row(0).to_owned());

        assert!(out_sigmoid[0].is_finite());
        assert!(out_relu[0].is_finite());
    }
}
