use crate::node::{Node};
pub struct KochNET {
    layers: Vec<Vec<Node>>,
    biases: Vec<f32>,
}

impl KochNET {

    pub fn new(layer_config: Vec<usize>) -> KochNET {
        let mut nn = Self::empty();
        nn.reconfigure(layer_config);
        return nn;
    }
    pub fn empty() -> KochNET {
        let nn = KochNET {
            layers: Vec::new(),
            biases: Vec::new(),
        };
        return nn;
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

    // https://github.com/jackm321/RustNN/blob/master/src/lib.rs
    /* Return the error of each output */
    pub fn train_iter(&mut self, input: &Vec<f32>, expected_output: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        todo!();
        // let errors = Vec::new();

    }

    pub fn train(&mut self, epochs: usize, examples: &Vec<(Vec<f32>, Vec<f32>)>) {
        for e in 0..epochs {
            let (input, expected_output) = &examples[e % examples.len()];
            self.train_iter(&input, &expected_output);
        }
    }

    fn heaviside(&self, input: f32) -> f32 {
        if input < 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    pub fn evaluate(&self, inputs: &[f32]) -> Vec<f32> {
        fn dot_prod(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
            // Computes the dot product of <a_1, a_2, ..., a_n> dot <b_1, b_2, ..., b_n> + biases
            let mut total = 0.0;
            for (v1_val, v2_val) in v1.iter().zip(v2.iter()) {
                total += v1_val * v2_val;
            }
            return total;
        }
        let mut current_input = inputs.to_vec();
        
        // We skip the first layer since that is really the input
        for layer_index in 1..self.layers.len() {
            let layer = self.layers.get(layer_index).unwrap();
            let mut output = Vec::new();

            // For each node, compute the dot product of the input and that node's weights
            for (node_idx, node) in layer.iter().enumerate() {   
                let mut node_output = dot_prod(&current_input, &node.weights());
                
                // If the last layer, add the bias
                if layer == self.layers.last().unwrap() {
                    node_output += self.biases[node_idx];
                }
                output.push(self.heaviside(node_output));
            }
            
            current_input = output;
        }
        return current_input;
    }

}


#[cfg(test)]
mod tests {
    use crate::KochNET::KochNET;

    #[test]
    fn test_intialization() {
        //---------- Test with large neural network
        let layer_sizes = vec![2, 3, 4, 5, 6];
        let nn = KochNET::new(layer_sizes.clone());

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
            let mut nn = KochNET::new(vec![2, 1]);
            nn.layers[1][0].weights_mut()[0] = 1.0;
            nn.layers[1][0].weights_mut()[1] = 1.0;
            nn.biases[0] = -0.5;

            return nn.evaluate(&[A as f32, B as f32])[0];
        }

        assert_eq!(ann_OR(0, 0), 0.0);
        assert_eq!(ann_OR(0, 1), 1.0);
        assert_eq!(ann_OR(1, 0), 1.0);
        assert_eq!(ann_OR(1, 1), 1.0);
    }

    #[test]
    fn test_AND_eval() {
        fn ann_AND(A: u32, B: u32) -> f32 {
            let mut nn = KochNET::new(vec![2, 1]);
            nn.layers[1][0].weights_mut()[0] = 0.4;
            nn.layers[1][0].weights_mut()[1] = 0.4;
            nn.biases[0] = -0.5;

            return nn.evaluate(&[A as f32, B as f32])[0];
        }

        assert_eq!(ann_AND(0, 0), 0.0);
        assert_eq!(ann_AND(0, 1), 0.0);
        assert_eq!(ann_AND(1, 0), 0.0);
        assert_eq!(ann_AND(1, 1), 1.0);
    }

    
    #[test]
    fn test_OR_train() {
        let mut OR_nn = KochNET::new(vec![2, 1]);

        let examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        OR_nn.train(50, &examples);

        assert_eq!(OR_nn.evaluate(&vec![0.0, 0.0])[0], 0.0);
        assert_eq!(OR_nn.evaluate(&vec![0.0, 1.0])[0], 1.0);
        assert_eq!(OR_nn.evaluate(&vec![1.0, 0.0])[0], 1.0);
        assert_eq!(OR_nn.evaluate(&vec![1.0, 1.0])[0], 1.0);
    }

    #[test]
    fn test_XOR_train() {
        let mut XOR_nn = KochNET::new(vec![2, 2, 1]);

        let examples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        XOR_nn.train(50, &examples);

        assert_eq!(XOR_nn.evaluate(&vec![0.0, 0.0])[0], 0.0);
        assert_eq!(XOR_nn.evaluate(&vec![0.0, 1.0])[0], 1.0);
        assert_eq!(XOR_nn.evaluate(&vec![1.0, 0.0])[0], 1.0);
        assert_eq!(XOR_nn.evaluate(&vec![1.0, 1.0])[0], 0.0);
    }
}