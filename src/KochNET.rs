use crate::node::Node;
pub struct KochNET {
    layers: Vec<Vec<Node>>,
    biases: Vec<f32>,
    learn_rate: f32,
    activation_func: Box<dyn Fn(f32) -> f32>
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
            biases: Vec::new(),
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
        self.biases = vec![0.0; layer_config.last().unwrap().clone()];

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
    }

    /// Returns an uninitialized set of nodes with the proper number of weights
    fn clone_node_shape(&self) -> Vec<Vec<Node>> {
        let mut layers = Vec::new();
        for l in self.layers.iter() {
            let mut copied_nodes = Vec::new();

            for node in l.iter() {
                copied_nodes.push(Node::new(&node.weights().len()));
            }
            layers.push(copied_nodes);
        }
        return layers;
    }

    // https://github.com/jackm321/RustNN/blob/master/src/lib.rs
    /* Returns the squared error of each output and the output itself */
    pub fn train_iter(&mut self, input: &Vec<f32>, expected_output: &Vec<f32>) -> (f32, Vec<f32>) {
        let learn_rate = 0.05;
        let (activations, output) = self.evaluate(input, );
        let output_layer = self.layers.last().unwrap(); // this isn't used until the initial errors are determined

        let mut prev_error_terms: Vec<f32> = Vec::new(); // keeps track of error terms
        let mut new_error_terms: Vec<f32> = Vec::new();

        // Work backwards starting from the last layer as the "input"
        let mut new_layers: Vec<Vec<Node>> = self.layers.clone();

        for (layer_index, input_layer) in self.layers[1..].iter().enumerate().rev() {
            
            // Repeat the process of updating weights for each node in the output layer (since the weights
            // are stored on the output layer nodes)
            for (i, output_node) in output_layer.iter().enumerate() {
                // Stores the output_nodes new weights to be applied later
                // layer_index + 1 because we want the output layer which contains the weights
                let new_output_node = &mut new_layers[layer_index + 1][i];

                // iterate through each input node
                for (j, input_node) in input_layer.iter().enumerate() {
                    let (err_term, change_in_weight): (f32, f32);

                    // Calculate the change in weight (and error term) based on if you are in the input or output layer
                    if input_layer == self.layers.last().unwrap() {
                        (err_term, change_in_weight) = self.weight_change_output(i, &expected_output[j], &activations);
                    } else {
                        (err_term, change_in_weight) = self.weight_change_hidden(j, i, layer_index, &activations, &prev_error_terms)
                    }      
                   
                    new_error_terms.push(err_term);  // We keep the error terms to be used in the next iteration of backprop
                    new_output_node.weights_mut()[j] = change_in_weight + output_node.weights()[j];
                }
            }
            // When we are training the next layer down, we need to swap
            prev_error_terms = new_error_terms;
            new_error_terms = Vec::new();
        }

        self.layers = new_layers;

        return (self.rmse(&output, expected_output), output);
    }

    /// Returns (error_term, delta_weight)
    fn weight_change_output(
        &self,
        start_index: usize,
        expected: &f32,
        layer_activations: &Vec<Vec<f32>>,
    ) -> (f32, f32) {
        // Calculate the error term of the neuron in the input layer
        let input_activ = &layer_activations.last().unwrap()[start_index];
        let error_term = (input_activ - expected) * input_activ * (1.0 - input_activ);

        // Calculate the gradient of the error function with respect to weight
        let delta_error_resp_weight = error_term * expected;

        return (error_term, delta_error_resp_weight);
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

        for (output_ind, output_neuron) in output_layer.iter().enumerate() {
            let input_output_weight = output_neuron.weights()[start_index];
            err_rspct_input_actv += input_output_weight * &errors[output_ind];
        }

        // Calculate the δ(activation input)/δ(weighted sum to input)j
        let input_activ = &layer_activations[input_layer_index][start_index];
        let activation_rspct_wghted_sum = input_activ * (1.0 - input_activ);

        let error_term = err_rspct_input_actv * activation_rspct_wghted_sum;

        // Calculate the gradient of the error function with respect to weight
        let output_activation = &layer_activations[input_layer_index + 1][end_index];
        let delta_error_resp_weight = error_term * output_activation;

        return (error_term, delta_error_resp_weight);
    }

    pub fn sigmoid(x: f32) -> f32 {
        let e: f32 = std::f32::consts::E;
        return (1.0 + e.powf(-x)).powf(-1.0);
    }

    fn rmse(&self, output: &Vec<f32>, expected: &Vec<f32>) -> f32 {
        output
            .iter()
            .zip(expected.iter())
            .fold(0.0, |acc, (output, expected)| (output - expected).powf(2.0))
    }

    pub fn train(&mut self, epochs: usize, examples: &Vec<(Vec<f32>, Vec<f32>)>) {
        for e in 0..epochs {
            let (input, expected_output) = &examples[e % examples.len()];
            self.train_iter(input, expected_output);
        }
    }

    fn heaviside(input: f32) -> f32 {
        if input < 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
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
        let mut saved_activations = Vec::new();

        // We skip the first layer since that is really the input
        for layer_index in 1..self.layers.len() {
            let layer = self.layers.get(layer_index).unwrap();
            let mut output = Vec::new();

            // For each node, compute the dot product of the input and that node's weights
            for (node_idx, node) in layer.iter().enumerate() {
                let mut node_output = Self::dot_prod(&current_input, &node.weights());

                // If the last layer, add the bias
                if layer == self.layers.last().unwrap() {
                    node_output += self.biases[node_idx];
                }
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

    #[test]
    fn test_intialization() {
        //---------- Test with large neural network
        let layer_sizes = vec![2, 3, 4, 5, 6];
        let nn = KochNET::new(layer_sizes.clone(), 0.05);

        assert_eq!(nn.layers.len(), layer_sizes.len());
        assert_eq!(nn.biases.len(), 6);

        for node in nn.layers.first().unwrap() {
            assert_eq!(node.weights().len(), 0);
        }

        let mut prev_layer_size = 0;
        for (layer, expected_size) in nn.layers.iter().zip(layer_sizes) {
            assert_eq!(layer.len(), expected_size);

            for node in layer.iter() {
                assert_eq!(node.weights().len(), prev_layer_size);
                for w in node.weights().iter() {
                    assert_eq!(*w, 0.0);
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
            nn.biases[0] = -0.5;

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
            nn.biases[0] = -0.5;

            return nn.run(&[A as f32, B as f32])[0];
        }

        assert_eq!(ann_AND(0, 0), 0.0);
        assert_eq!(ann_AND(0, 1), 0.0);
        assert_eq!(ann_AND(1, 0), 0.0);
        assert_eq!(ann_AND(1, 1), 1.0);
    }

    #[test]
    fn test_OR_train() {
        let mut OR_nn = KochNET::new(vec![2, 1], 0.05);
        OR_nn.set_activation_func(&KochNET::sigmoid);

        let examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        OR_nn.train(50, &examples);

        assert_eq!(OR_nn.run(&vec![0.0, 0.0])[0], 0.0);
        assert_eq!(OR_nn.run(&vec![0.0, 1.0])[0], 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 0.0])[0], 1.0);
        assert_eq!(OR_nn.run(&vec![1.0, 1.0])[0], 1.0);
    }

    #[test]
    fn test_XOR_train() {
        let mut XOR_nn = KochNET::new(vec![2, 2, 1], 0.05);
        XOR_nn.set_activation_func(&KochNET::sigmoid);

        let examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        XOR_nn.train(50, &examples);

        assert_eq!(XOR_nn.run(&vec![0.0, 0.0])[0], 0.0);
        assert_eq!(XOR_nn.run(&vec![0.0, 1.0])[0], 1.0);
        assert_eq!(XOR_nn.run(&vec![1.0, 0.0])[0], 1.0);
        assert_eq!(XOR_nn.run(&vec![1.0, 1.0])[0], 0.0);
    }
}
