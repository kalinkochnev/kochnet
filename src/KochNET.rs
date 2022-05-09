use crate::node::Node;
use rand::{seq::SliceRandom, thread_rng};

use log::debug;


pub type Layer = Vec<Node>;
pub type Network = Vec<Layer>;

pub struct KochNET {
    layers: Vec<Layer>,
    learn_rate: f32,
    activation_func: Box<dyn Fn(f32) -> f32>,
}

impl KochNET {
    pub fn new(layer_config: Vec<usize>, learn_rate: f32) -> KochNET {
        let mut nn = Self::empty();
        nn.set_learn_rate(learn_rate);
        nn.reconfigure(layer_config);
        return nn;
    }

    pub fn empty() -> KochNET {
        let nn = KochNET {
            layers: Vec::new(),
            learn_rate: 0.0,
            activation_func: Box::new(Self::heaviside),
        };
        return nn;
    }

    pub fn set_activation_func(&mut self, activation_func: &'static dyn Fn(f32) -> f32) {
        self.activation_func = Box::new(activation_func);
    }

    pub fn set_learn_rate(&mut self, learn_rate: f32) {
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
        activations: &Vec<Vec<f32>>,
        expected: &Vec<f32>,
        errors: &mut Vec<f32>,
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
            let (error, node_change) =
                node.delta_node_output_layer(self.learn_rate, activation, expected, &second_last_activations);

            layer_changes.push(node_change);
            errors.push(error);
        }
        new_layers.push(layer_changes); // we can push because there is no changes yet, otherwise we need to add to front
    }

    // https://github.com/jackm321/RustNN/blob/master/src/lib.rs
    /* Returns the squared error of each output and the output itself */
    pub fn train_iter(&mut self, input: &Vec<f32>, expected_output: &Vec<f32>) -> (f32, Vec<Layer>, Vec<f32>) {
        let (activations, output) = self.evaluate(input);
        // let mut prev_error_terms: Vec<f32> = Vec::new(); // keeps track of error terms
        // let mut new_error_terms: Vec<f32> = Vec::new();

        // Work backwards starting from the last layer as the "input"
        let mut new_layers: Vec<Layer> = vec![];
        let mut prev_errors = Vec::new();

        // initialize the errors to do backpropogation using the output layer and the target layer
        self.train_output_layer(&activations, &expected_output, &mut prev_errors, &mut new_layers);

        // // Do backpropagation on the remaining layers
        // for L in (0..self.layers.len() - 2).rev() {
        //     let (input_layer, output_layer) = (&self.layers[L], &self.layers[L + 1]);
        //     let (input_activations, output_activations) = (&activations[L], &activations[L + 1]);
        //     let temp_errors = Vec::new();

        //     // Repeat the process of updating weights for each node in the output layer (since the weights
        //     // are stored on the output layer nodes)
        //     for (output_node, output_activ) in output_layer.iter().zip(output_activations) {
                
        //     }

        // }

        // Add the input layer to the front of the new_layers since combine_layers() needs equal sizing
        new_layers.insert(0, self.layers[0].clone());
        return (self.rmse(&output, expected_output), new_layers, output);
    }

    /// Returns (error_term, delta_weight) for a particular neuron (based on backpropogation)
    /// * `input_layer_index` - What layer is currently being treated as the input layer
    /// * `start_index` (j)- The input layer node index to calculate the weight change for
    /// * `end_index` (i)- The output layer node index to calculate the weight change for
    /// * `layer_activations` - This all the saved activations for a particular input
    /// * `errors` - These are the error terms (see wikipedia) for the layer ahead
    /// https://en.wikipedia.org/wiki/Backpropagation#Derivation
    /// 1. Calculate the δ(Error)/δ(output_j)
    ///     You need to find the contributions of the activated node to the error (since it
    ///     affects the neurons in the next layer too)
    /// ```
    ///     Layer j     Layer l
    ///     (A)         |---(0.2)
    ///     (B)---------|---(0.3)
    ///     (C)         |---(0.1)
    /// ```
    /// 2. Calculate δ(activation_j)/δ(output_j)
    ///     If using the sigmoid as the activation function,
    ///     this is just activation_j * (1 - activation_j)
    fn weight_change_hidden(
        &self,
        start_index: usize,
        end_index: usize,
        input_layer_index: usize,
        layer_activations: &Vec<Vec<f32>>,
        errors: &Vec<f32>,
    ) -> (f32, f32) {
        let output_layer = &self.layers[input_layer_index + 1];
        let mut err_rspct_input_actv = 0.0; // δ(Error)/δ(output)j where j is the "input" layer (since we're working backwards)

        for (_output_ind, output_neuron) in output_layer.iter().enumerate() {
            let input_output_weight = output_neuron.weights()[start_index];
            err_rspct_input_actv += input_output_weight * &errors[end_index];
        }
        // debug!("Error term (where i is a hidden neuron):\nΣ[err_term_j * ] * {} * (1.0 - {})", expected, output_activ, output_activ, output_activ);

        // Calculate the δ(activation input)/δ(weighted sum to input)j
        let input_activ = &layer_activations[input_layer_index][start_index];
        let activation_rspct_wghted_sum = input_activ * (1.0 - input_activ);

        let error_term = err_rspct_input_actv * activation_rspct_wghted_sum;

        // Calculate the gradient of the error function with respect to weight
        let output_activation = &layer_activations[input_layer_index][end_index];
        let delta_error_resp_weight = error_term * output_activation;

        return (error_term, delta_error_resp_weight);
    }

    fn rmse(&self, output: &Vec<f32>, expected: &Vec<f32>) -> f32 {
        let mut total = 0.0;
        for (out, exp) in output.iter().zip(expected.iter()) {
            total += (out - exp).powf(2.0);
        }
        return total;
    }

    pub fn train(&mut self, epochs: usize, examples: &mut Vec<(Vec<f32>, Vec<f32>)>) {

        let mut avg_err = 0f32;
        for _e in 0..epochs {
            let mut combined_changes: Network = self.layers.clone();
            let mut epoch_rmse = 0f32;
            examples.shuffle(&mut thread_rng());

            for (input, expected) in examples.iter() {
                let (rmse, weight_changes, output) = self.train_iter(input, expected);
                epoch_rmse += rmse;

                combined_changes = Self::combine_layers(combined_changes, weight_changes);
                // println!("{:?}", combined_changes);
                // println!("{:?}", output);
            }
            avg_err = epoch_rmse / (examples.len() as f32);
            self.layers = combined_changes;
            debug!("Avg RMSE {}: : : :\nCombined ----- {:?} ", avg_err, self.layers);
            // print!("{}, ", avg_err);
        }
    }

    fn combine_layers(n1: Vec<Layer>, n2: Vec<Layer>) -> Vec<Layer> {
        if n1.len() != n2.len() {
            panic!("Cannot combine networks of different length!");
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

    fn heaviside(input: f32) -> f32 {
        if input < 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    pub fn sigmoid(x: f32) -> f32 {
        let e: f32 = std::f32::consts::E;
        return (1.0 + e.powf(-x)).powf(-1.0);
    }

    fn dot_prod(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
        // Computes the dot product of <a_1, a_2, ..., a_n> dot <b_1, b_2, ..., b_n> + biases
        let mut total = 0.0;
        for (v1_val, v2_val) in v1.iter().zip(v2.iter()) {
            total += v1_val * v2_val;
        }
        return total;
    }

    /// Returns (activations of network, output of network)
    /// This skips the input layer for activations
    fn evaluate(&self, inputs: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
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

                output.push((self.activation_func)(node_output));
            }

            // We save the activations for when we are training the neural network
            saved_activations.push(output.clone());
            current_input = output;
        }
        return (saved_activations, current_input);
    }

    pub fn run(&self, inputs: &[f32]) -> Vec<f32> {
        let (activations, output) = self.evaluate(inputs);
        return output;
    }
}

#[cfg(test)]
mod tests {
    use crate::KochNET::KochNET;
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
        fn ann_OR(A: u32, B: u32) -> f32 {
            let mut nn = KochNET::new(vec![2, 1], 0.01);
            nn.layers[1][0].weights_mut()[0] = 1.0;
            nn.layers[1][0].weights_mut()[1] = 1.0;
            nn.layers[1][0].set_bias(-0.5);

            return nn.run(&[A as f32, B as f32])[0];
        }

        assert_eq!(ann_OR(0, 0), 0.0);
        assert_eq!(ann_OR(0, 1), 1.0);
        assert_eq!(ann_OR(1, 0), 1.0);
        assert_eq!(ann_OR(1, 1), 1.0);
    }

    #[test]
    fn test_AND_eval() {
        fn ann_AND(A: u32, B: u32) -> f32 {
            let mut nn = KochNET::new(vec![2, 1], 0.05);
            nn.layers[1][0].weights_mut()[0] = 0.4;
            nn.layers[1][0].weights_mut()[1] = 0.4;
            nn.layers[1][0].set_bias(-0.5);

            return nn.run(&[A as f32, B as f32])[0];
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
        AND_nn.set_activation_func(&KochNET::sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        AND_nn.train(1000, &mut examples);

        assert_eq!(AND_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![0.0, 1.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![1.0, 0.0])[0].round(), 0.0);
        assert_eq!(AND_nn.run(&vec![1.0, 1.0])[0].round(), 1.0);
    }


    #[test]
    fn test_OR_train() {
        // simple_logging::log_to_file("or_train.log", LevelFilter::Debug);

        let mut OR_nn = KochNET::new(vec![2, 1], 0.05);
        OR_nn.set_activation_func(&KochNET::sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        OR_nn.train(1000, &mut examples);

        assert_eq!(OR_nn.run(&vec![0.0, 0.0])[0].round(), 0.0);
        assert_eq!(OR_nn.run(&vec![0.0, 1.0])[0].round(), 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 0.0])[0].round(), 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 1.0])[0].round(), 1.0);
    }

    #[test]
    fn test_XOR_train() {
        let mut XOR_nn = KochNET::new(vec![2, 2, 1], 0.005);
        XOR_nn.set_activation_func(&KochNET::sigmoid);

        let mut examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        XOR_nn.train(500, &mut examples);

        assert_eq!(XOR_nn.run(&vec![0.0, 0.0])[0], 0.0);
        assert_eq!(XOR_nn.run(&vec![0.0, 1.0])[0], 1.0);
        assert_eq!(XOR_nn.run(&vec![1.0, 0.0])[0], 1.0);
        assert_eq!(XOR_nn.run(&vec![1.0, 1.0])[0], 0.0);
    }
}
