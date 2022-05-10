use crate::node::Node;
use rand::{seq::SliceRandom, thread_rng};
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::iter::Iterator;
use std::{fs::File, io::Error};
use rayon::prelude::*;
use log::debug;
use serde::{Deserialize, Serialize};

pub type Layer = Vec<Node>;
pub type Network = Vec<Layer>;

#[derive(Serialize, Deserialize, Clone)]
pub enum ActivationFunc {
    Sigmoid,
    Heaviside,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct KochNET {
    layers: Vec<Layer>,
    learn_rate: f64,
    activation_func: ActivationFunc,
}

impl KochNET {
    const SAVE_FILE: &'static str = "network_params.json";
    pub fn new(layer_config: Vec<usize>, learn_rate: f64) -> KochNET {
        let mut nn = Self::empty();
        nn.set_learn_rate(learn_rate);
        nn.reconfigure(layer_config);
        return nn;
    }

    pub fn empty() -> KochNET {
        let nn = KochNET {
            layers: Vec::new(),
            learn_rate: 0.0,
            activation_func: ActivationFunc::Heaviside,
        };
        return nn;
    }

    pub fn set_activation_func(&mut self, func: ActivationFunc) {
        self.activation_func = func;
    }

    pub fn get_activation_func(&self) -> &'static dyn Fn(f64) -> f64 {
        match self.activation_func {
            ActivationFunc::Heaviside => return &Self::heaviside,
            ActivationFunc::Sigmoid => return &Self::sigmoid,
        }
    }

    pub fn set_learn_rate(&mut self, learn_rate: f64) {
        self.learn_rate = learn_rate;
    }

    /* Reconfigures the layers entirely (drops the old weights) */
    pub fn reconfigure(&mut self, layer_config: Vec<usize>) {
        self.layers = Vec::new();
        // for i in 0..layer_config.last().unwrap().clone() {
        //     self.biases.push();//rand::thread_rng().gen_range(-1.0..1.0));
        // }

        // Add layers inbetween and their weights one
        let mut prev_layer_size = &0;
        for layer_index in 0..layer_config.len() {
            let layer_size = layer_config.get(layer_index).unwrap();
            let mut new_layer = Vec::new();

            for _i in 0..*layer_size {
                // Number of weights should be equal to number of nodes in
                // previous layer
                new_layer.push(Node::new(prev_layer_size));
            }
            prev_layer_size = layer_size;
            self.layers.push(new_layer);
        }

        // set biases of first layer = 0
        for node in self.layers.first_mut().unwrap() {
            node.set_bias(0.0);
        }
    }

    fn train_output_layer(
        &self,
        activations: &Vec<Vec<f64>>,
        expected: &Vec<f64>,
        errors: &mut Vec<f64>,
        new_layers: &mut Vec<Layer>,
    ) {
        let second_last_activations = activations
            .get(activations.len() - 2)
            .expect("network should have at least 2 layers");
        let output_activations = activations.last().unwrap();

        let output_layer: &Layer = self.layers.last().unwrap();
        let node_iter = output_layer.iter().zip(output_activations.iter());

        let mut layer_changes: Layer = Vec::new();
        for ((node, activation), expected) in node_iter.zip(expected.iter()) {
            let (error, node_change) = node.node_change_output(
                self.learn_rate,
                activation,
                expected,
                &second_last_activations,
            );

            layer_changes.push(node_change);
            errors.push(error);
        }
        new_layers.push(layer_changes); // we can push because there is no changes yet, otherwise we need to add to front
    }

    // https://github.com/jackm321/RustNN/blob/master/src/lib.rs
    /* Returns the squared error of each output and the output itself */
    pub fn train_iter(
        &self,
        input: &Vec<f64>,
        expected_output: &Vec<f64>,
    ) -> (f64, Vec<Layer>, Vec<f64>) {
        let (mut activations, output) = self.evaluate(input);

        // Work backwards starting from the last layer as the "input"
        let mut new_layers: Vec<Layer> = vec![];
        let mut errors = Vec::new();

        // initialize the errors to do backpropogation using the output layer and the target layer
        debug!("-------------------------------- OUTPUT TRAINING --------------------------------");
        self.train_output_layer(&activations, &expected_output, &mut errors, &mut new_layers);

        // Do backpropagation on the remaining layers
        debug!(
            "-------------------------------- BACKPROP TRAINING --------------------------------"
        );

        for L in (1..self.layers.len() - 1).rev() {
            let (current_layer, downstream_layer) = (&self.layers[L], &self.layers[L + 1]);
            let (upstream_activations, current_activations) =
                (&activations[L - 1], &activations[L]);

            let mut layer_errors = Vec::new();
            let mut layer_changes: Layer = Vec::new();

            // Repeat the process of updating weights for each node in the output layer (since the weights
            // are stored on the output layer nodes)

            for (curr_node_index, (curr_node, curr_node_activ)) in
                current_layer.iter().zip(current_activations).enumerate()
            {
                let (error, node_change) = curr_node.node_change_hidden_layer(
                    &self.learn_rate,
                    curr_node_activ,
                    curr_node_index,
                    &errors,
                    downstream_layer,
                    upstream_activations,
                );
                layer_errors.push(error);
                layer_changes.push(node_change);
            }
            errors = layer_errors;
            new_layers.insert(0, layer_changes);
        }

        // Add the input layer to the front of the new_layers since combine_layers() needs equal sizing
        new_layers.insert(0, self.layers[0].clone());
        return (self.rmse(&output, expected_output), new_layers, output);
    }

    fn rmse(&self, output: &Vec<f64>, expected: &Vec<f64>) -> f64 {
        let mut total = 0.0;
        for (out, exp) in output.iter().zip(expected.iter()) {
            total += (out - exp).powf(2.0);
        }
        return total;
    }


    /// Returns an uninitialized set of nodes with the proper number of weights
    fn clone_layer_shape(&self) -> Vec<Vec<Node>> {
        let mut layers = Vec::new();
        for l in self.layers.iter() {
            let mut copied_nodes = Vec::new();

            for node in l.iter() {
                copied_nodes.push(Node::empty(&node.weights().len()));
            }
            layers.push(copied_nodes);
        }
        return layers;
    }

    pub fn train(
        &mut self,
        epochs: usize,
        examples: &mut Vec<(Vec<f64>, Vec<f64>)>,
        is_running: Arc<AtomicBool>,
        handle_data: &dyn Fn(usize, f64)
    ) {
        let mut avg_err = 0f64;
        for _e in 0..epochs {
            let mut combined_changes: Network = self.layers.clone();
            let mut epoch_rmse = 0f64;
            examples.shuffle(&mut thread_rng());

            for (input, expected) in examples.iter() {
                if !is_running.load(Ordering::Relaxed) {
                    println!("Saving the network.......");
                    self.save();
                    std::process::exit(0);
                }
                let (rmse, weight_changes, output) = self.train_iter(input, expected);
                epoch_rmse += rmse;

                debug!(
                    "Combined changes ---- \n{:?} \n\n Weight changes: ---- {:?}",
                    combined_changes, weight_changes
                );

                combined_changes = Self::combine_layers(combined_changes, weight_changes);
                // println!("{:?}", output);
            }
            avg_err = epoch_rmse / (examples.len() as f64);
            (handle_data)(_e, avg_err);
            self.layers = combined_changes;
            debug!(
                "Avg RMSE {}: : : :\nCombined ----- {:?} ",
                avg_err, self.layers
            );
            // print!("{}, ", avg_err);
        }
        println!("Saving the network.......");
        self.save();
    }


    fn combine_layers(n1: Vec<Layer>, n2: Vec<Layer>) -> Vec<Layer> {
        if n1.len() != n2.len() {
            panic!(
                "Cannot combine networks of different length! n1: {}, n2: {}",
                n1.len(),
                n2.len()
            );
        }
        let mut new_network: Vec<Layer> = n1;

        // the network_changes does not include the input layer since it does not have any weights
        for layer in 0..new_network.len() {
            for node in 0..new_network[layer].len() {
                let node_a = new_network[layer].get_mut(node).unwrap();
                let node_b = n2[layer].get(node).unwrap();
                node_a.apply_changes(node_b);
            }
        }
        return new_network;
    }

    fn heaviside(input: f64) -> f64 {
        if input < 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    pub fn sigmoid(x: f64) -> f64 {
        let e: f64 = std::f64::consts::E;
        return (1.0 + e.powf(-x)).powf(-1.0);
    }

    fn dot_prod(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
        // Computes the dot product of <a_1, a_2, ..., a_n> dot <b_1, b_2, ..., b_n> + biases
        let mut total = 0.0;
        for (v1_val, v2_val) in v1.iter().zip(v2.iter()) {
            total += v1_val * v2_val;
        }
        return total;
    }

    /// Returns (activations of network, output of network)
    /// This skips the input layer for activations
    fn evaluate(&self, inputs: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut current_input = inputs.to_vec();
        let mut saved_activations = vec![current_input.clone()];

        // We skip the first layer since that is really the input
        for layer_index in 1..self.layers.len() {
            let layer = self.layers.get(layer_index).unwrap();
            let mut output = Vec::new();

            // For each node, compute the dot product of the input and that node's weights
            for (node_idx, node) in layer.iter().enumerate() {
                let mut node_output = Self::dot_prod(&current_input, &node.weights());

                node_output += node.bias();

                output.push((self.get_activation_func())(node_output));
            }

            // We save the activations for when we are training the neural network
            saved_activations.push(output.clone());
            current_input = output;
        }
        return (saved_activations, current_input);
    }

    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        let (activations, output) = self.evaluate(inputs);
        return output;
    }

    pub fn save(&self) -> Result<(), Error> {
        let network_string = serde_json::to_string(&self).unwrap();
        let mut output = File::create(Self::SAVE_FILE)?;
        write!(output, "{}", network_string).expect("There was an error saving the network");
        Ok(())
    }

    pub fn load() -> Result<KochNET, Error> {
        let contents = fs::read_to_string(Self::SAVE_FILE).expect("Loading the file went wrong");
        let network: KochNET = serde_json::from_str(contents.as_str())?;
        println!("Sucessfully loaded the network");

        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicBool, Arc};

    use crate::KochNET::{KochNET, ActivationFunc};
    use log::LevelFilter;

    #[test]
    fn test_intialization() {
        //---------- Test with large neural network
        let layer_sizes = vec![2, 3, 4, 5, 6];
        let nn = KochNET::new(layer_sizes.clone(), 0.05);

        assert_eq!(nn.layers.len(), layer_sizes.len());

        for node in nn.layers.first().unwrap() {
            assert_eq!(node.weights().len(), 0);
        }

        let mut prev_layer_size = 0;
        for (layer, expected_size) in nn.layers.iter().zip(layer_sizes) {
            assert_eq!(layer.len(), expected_size);

            for node in layer.iter() {
                assert_eq!(node.weights().len(), prev_layer_size);
                for w in node.weights().iter() {
                    assert_ne!(*w, 0.0);
                }
            }
            prev_layer_size = layer.len();
        }
    }

    #[test]
    fn test_OR_eval() {
        fn ann_OR(A: u32, B: u32) -> f64 {
            let mut nn = KochNET::new(vec![2, 1], 0.01);
            nn.layers[1][0].weights_mut()[0] = 1.0;
            nn.layers[1][0].weights_mut()[1] = 1.0;
            nn.layers[1][0].set_bias(-0.5);

            return nn.run(&[A as f64, B as f64])[0];
        }

        assert_eq!(ann_OR(0, 0), 0.0);
        assert_eq!(ann_OR(0, 1), 1.0);
        assert_eq!(ann_OR(1, 0), 1.0);
        assert_eq!(ann_OR(1, 1), 1.0);
    }

    #[test]
    fn test_XOR_eval() {
        fn ann_XOR(A: u32, B: u32) -> f64 {
            let mut nn = KochNET::new(vec![2, 2, 1], 0.01);

            nn.layers[1][0].weights_mut()[0] = 20.0;
            nn.layers[1][0].weights_mut()[1] = 20.0;
            nn.layers[1][0].set_bias(-10.0);

            nn.layers[1][1].weights_mut()[0] = -20.0;
            nn.layers[1][1].weights_mut()[1] = -20.0;
            nn.layers[1][1].set_bias(30.0);

            nn.layers[2][0].weights_mut()[0] = 20.0;
            nn.layers[2][0].weights_mut()[1] = 20.0;
            nn.layers[2][0].set_bias(-30.0);

            return nn.run(&[A as f64, B as f64])[0];
        }

        assert_eq!(ann_XOR(0, 0), 0.0);
        assert_eq!(ann_XOR(0, 1), 1.0);
        assert_eq!(ann_XOR(1, 0), 1.0);
        assert_eq!(ann_XOR(1, 1), 0.0);
    }

    #[test]
    fn test_AND_eval() {
        fn ann_AND(A: u32, B: u32) -> f64 {
            let mut nn = KochNET::new(vec![2, 1], 0.05);
            nn.layers[1][0].weights_mut()[0] = 0.4;
            nn.layers[1][0].weights_mut()[1] = 0.4;
            nn.layers[1][0].set_bias(-0.5);

            return nn.run(&[A as f64, B as f64])[0];
        }

        assert_eq!(ann_AND(0, 0), 0.0);
        assert_eq!(ann_AND(0, 1), 0.0);
        assert_eq!(ann_AND(1, 0), 0.0);
        assert_eq!(ann_AND(1, 1), 1.0);
    }

    #[test]
    fn test_apply_changes() {
        let mut nn1 = KochNET::new(vec![2, 1], 0.01);
        nn1.layers[1][0].weights_mut()[0] = 1.0;
        nn1.layers[1][0].weights_mut()[1] = 1.0;
        nn1.layers[1][0].set_bias(-0.5);

        let mut nn2 = KochNET::new(vec![2, 1], 0.01);
        nn2.layers[1][0].weights_mut()[0] = -1.0;
        nn2.layers[1][0].weights_mut()[1] = 1.0;
        nn2.layers[1][0].set_bias(-1.5);

        let mut expected_nn = KochNET::new(vec![2, 1], 0.01);
        expected_nn.layers[1][0].weights_mut()[0] = 0.0;
        expected_nn.layers[1][0].weights_mut()[1] = 2.0;
        expected_nn.layers[1][0].set_bias(-2.0);

        let combined = KochNET::combine_layers(nn1.layers, nn2.layers);

        assert_eq!(combined, expected_nn.layers)
    }

    #[test]
    fn test_AND_train() {
        // simple_logging::log_to_file("and_train.log", LevelFilter::Debug);

        let mut AND_nn = KochNET::new(vec![2, 1], 0.05);
        AND_nn.set_activation_func(ActivationFunc::Sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        AND_nn.train(1000, &mut examples, Arc::new(AtomicBool::new(true)), &|a: usize, b:f64|{});

        assert_eq!(AND_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![0.0, 1.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![1.0, 0.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![1.0, 1.0])[0].round(), 1.0);
    }

    #[test]
    fn test_AND_train_3_layers() {
        // simple_logging::log_to_file("and_train.log", LevelFilter::Debug);

        let mut AND_nn = KochNET::new(vec![2, 2, 1], 0.05);
        AND_nn.set_activation_func(ActivationFunc::Sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        AND_nn.train(10000, &mut examples, Arc::new(AtomicBool::new(true)), &|a: usize, b:f64|{});

        assert_eq!(AND_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![0.0, 1.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![1.0, 0.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![1.0, 1.0])[0].round(), 1.0);
    }

    #[test]
    fn test_OR_train_3_layers() {
        // simple_logging::log_to_file("or_train.log", LevelFilter::Debug);

        let mut OR_nn = KochNET::new(vec![2, 2, 1], 0.05);
        OR_nn.set_activation_func(ActivationFunc::Sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        OR_nn.train(10000, &mut examples, Arc::new(AtomicBool::new(true)), &|a: usize, b:f64|{});

        assert_eq!(OR_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
        assert_eq!(OR_nn.run(&vec![0.0, 1.0])[0].round(), 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 0.0])[0].round(), 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 1.0])[0].round(), 1.0);
    }

    #[test]
    fn test_OR_train() {
        // simple_logging::log_to_file("or_train.log", LevelFilter::Debug);

        let mut OR_nn = KochNET::new(vec![2, 1], 0.05);
        OR_nn.set_activation_func(ActivationFunc::Sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        OR_nn.train(1000, &mut examples, Arc::new(AtomicBool::new(true)), &|a: usize, b:f64|{});

        assert_eq!(OR_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
        assert_eq!(OR_nn.run(&vec![0.0, 1.0])[0].round(), 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 0.0])[0].round(), 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 1.0])[0].round(), 1.0);
    }

    #[test]
    fn test_XOR_train() {
        // simple_logging::log_to_file("xor_train.log", LevelFilter::Debug);
        let mut XOR_nn = KochNET::new(vec![2, 2, 1], 0.005);
        XOR_nn.set_activation_func(ActivationFunc::Sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        XOR_nn.train(10000, &mut examples, Arc::new(AtomicBool::new(true)), &|a: usize, b:f64|{});

        assert_eq!(XOR_nn.run(&vec![1.0, 1.0])[0].round(), 0.0);
        assert_eq!(XOR_nn.run(&vec![1.0, 0.0])[0].round(), 1.0);
        assert_eq!(XOR_nn.run(&vec![0.0, 1.0])[0].round(), 1.0);
        assert_eq!(XOR_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
    }
}
