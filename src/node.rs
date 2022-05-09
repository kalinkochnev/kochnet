use log::debug;
use rand::Rng;

use crate::KochNET::KochNET;



#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    weights: Vec<f32>,
    bias: f32
}
impl Node {
    pub fn new(num_weights: &usize) -> Node {
        let mut weights: Vec<f32> = vec![];
        for i in 0..*num_weights as u32 {
            weights.push(1.0);//rand::thread_rng().gen_range(-1.0..1.0));
        }
        Node {
            weights: weights,
            bias: 0.0
        }
    }

    pub fn weights(&self) -> &Vec<f32> {
        return &self.weights;
    }

    pub fn weights_mut(&mut self) -> &mut Vec<f32> {
        return &mut self.weights;
    }

    /// Returns a `Node` with the the weights attribute representing how much to change the weights
    /// as well as the bias representing how much to change the bias.
    /// Returns (f32, Node) -> (error term for node, change of the node)
    pub fn delta_node_output_layer(&self, learn_rate: f32, actual_activ: &f32, target_activ: &f32, prev_activations: &Vec<f32>) -> (f32, Node) {
        fn error_term(target: &f32, node_output: &f32) -> f32 {
            debug!("expected: {} actual: {}", target, node_output);
            let err = (target - node_output ) * node_output * (1.0 - node_output);
            debug!("change: ({} - {} ) * {} * (1.0 - {}) = {}", target, node_output, node_output, node_output, err);

            return err;
        }

        // Node error is only dependent on the activation of the node and what it actual output should be
        let node_error = error_term(target_activ, actual_activ);

        // Calculate how much to change the weights based on the activations of the nodes prior to this one
        let mut weight_changes = Vec::new();
        for y_k in prev_activations.iter() { 
            weight_changes.push(node_error * y_k * learn_rate);
        }
        debug!("prev activations -- \n{:?}", prev_activations);
        debug!("weight changes -- \n{:?}", weight_changes);

        // Calculate the change in bias (since it is treated as a neuron that is always activated)
        let bias_change = node_error  * learn_rate;
        debug!("biase change = {} * {} * {} = {}", node_error, self.bias, learn_rate, bias_change);

        return (node_error, Node {weights: weight_changes, bias: bias_change})
    }

    pub fn delta_node_hidden_layer(&self) -> (f32, Node) {
        
    }

    pub fn apply_changes(&mut self, delta: &Node) {
        for w in 0..self.weights.len() {
            self.weights[w] += delta.weights()[w];
        }
        self.bias += delta.bias;
    }

    pub fn bias(&self) -> &f32 {
        return &self.bias;
    }

    pub fn set_bias(&mut self, bias: f32)  {
        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use crate::node::Node;

    #[test]
    fn test_apply_changes() {
        let mut n1 = Node {weights: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], bias: 0.2};
        let delta = Node {weights: vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0], bias: 0.8};
        n1.apply_changes(&delta);

        assert_eq!(n1, Node {weights: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], bias: 1.0});
    }
}