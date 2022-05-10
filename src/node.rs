use csv::Error;
use log::debug;
use rand::Rng;
use serde::{Serialize, Deserialize};

use crate::KochNET::KochNET;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Node {
    weights: Vec<f64>,
    bias: f64,
}
impl Node {
    pub fn new(num_weights: &usize) -> Node {
        let mut weights: Vec<f64> = vec![];
        for i in 0..*num_weights as u32 {
            weights.push(rand::thread_rng().gen_range(-2.95..2.95));
        }
        Node {
            weights: weights,
            bias: 0.0,
        }
    }

    pub fn empty(num_weights: &usize) -> Node {
        let mut weights: Vec<f64> = vec![];
        for i in 0..*num_weights as u32 {
            weights.push(0.0);
        }
        Node {
            weights: weights,
            bias: 0.0,
        }
            
    }

    pub fn weights(&self) -> &Vec<f64> {
        return &self.weights;
    }

    pub fn weights_mut(&mut self) -> &mut Vec<f64> {
        return &mut self.weights;
    }

    /// Returns a `Node` with the the weights attribute representing how much to change the weights
    /// if the node is located at the output layer
    /// as well as the bias representing how much to change the bias.
    /// Returns (f64, Node) -> (error term for node, change of the node)
    pub fn node_change_output(
        &self,
        learn_rate: f64,
        node_activ: &f64,
        target_activ: &f64,
        upstream_activ: &Vec<f64>,
    ) -> (f64, Node) {
        fn error_term(target: &f64, node_output: &f64) -> f64 {
            debug!("expected: {} actual: {}", target, node_output);
            let err = (target - node_output) * node_output * (1.0 - node_output);
            debug!(
                "change: ({} - {} ) * {} * (1.0 - {}) = {}",
                target, node_output, node_output, node_output, err
            );

            return err;
        }

        // Node error is only dependent on the activation of the node and what it actual output should be
        let node_error = error_term(target_activ, node_activ);

        // Calculate how much to change the weights based on the activations of the nodes prior to this one
        let mut weight_changes = Vec::new();
        for y_k in upstream_activ.iter() { // y_k is the activation of the upstream node
            weight_changes.push(node_error * y_k * learn_rate);
        }
        debug!("prev activations -- \n{:?}", upstream_activ);
        debug!("weight changes -- \n{:?}", weight_changes);

        // Calculate the change in bias (since it is treated as a neuron that is always activated)
        let bias_change = node_error * learn_rate;
        debug!(
            "biase change = {} * {} * {} = {}",
            node_error, self.bias, learn_rate, bias_change
        );

        return (
            node_error,
            Node {
                weights: weight_changes,
                bias: bias_change,
            },
        );
    }

    /// Returns (error_term, change in weights) for a particular neuron (based on backpropogation)
    /// * `learn_rate` - a constant 0<a<1 for rate of learning
    /// * `node_activ` - the activation of the current node
    /// * `node_index`- the index of the current node within the current layer
    /// * `errors` - the error terms from the previous iteration of backprop
    /// * `downstream` - The layer that is downstream/ahead of the layer the current node is in
    /// * `upstream_activ` - These are the activations of the nodes upstream/behind the current node (where behind is the direction of backprop)
    /// 
    /// https://en.wikipedia.org/wiki/Backpropagation#Derivation
    /// 1. Calculate the δ(Error)/δ(output_j)
    ///     You need to find the contributions of the activated node to the error (since it
    ///     affects the neurons in the next layer too)
    /// 
    ///     Layer j     Layer l
    ///     (A)         |---(0.2)
    ///     (B)---------|---(0.3)
    ///     (C)         |---(0.1)
    /// 
    /// 2. Calculate δ(activation_j)/δ(output_j)
    ///     If using the sigmoid as the activation function,
    ///     this is just activation_j * (1 - activation_j)
    pub fn node_change_hidden_layer(
        &self,
        learn_rate: &f64,
        node_activ: &f64,
        node_index: usize,
        errors: &Vec<f64>,
        downstream: &Vec<Node>,
        upstream_activ: &Vec<f64>,
    ) -> (f64, Node) {
        fn error_term(node_activ: &f64, node_index: usize, downstream: &Vec<Node>, errors: &Vec<f64>) -> f64 {
            let change_in_activation = node_activ * (1.0 - node_activ); // with respect to the net input
            debug!(
                "change in activation: {} * (1.0 - {}) = {}",
                node_activ, node_activ, change_in_activation
            );
            let downstream_weights = downstream.iter().map(|node| node.weights()[node_index]);
            let change_in_error: f64 = downstream_weights
                .zip(errors.iter())
                .map(|(weight, error)| weight * error)
                .sum(); // contributed to by all nodes that recieve input from the current node
    
            return change_in_activation * change_in_error;
        }
        
        
        let mut new_weights = Vec::new();
        let node_error = error_term(node_activ, node_index, downstream, errors);

        for y_k in upstream_activ.iter() { // y_k is the activation of the upstream node
            new_weights.push(learn_rate * node_error * y_k);
        }

        let bias_change = learn_rate * node_error;

        return (node_error, 
            Node {
                weights: new_weights,
                bias: bias_change
            }
        )
    }

    pub fn apply_changes(&mut self, delta: &Node) {
        for w in 0..self.weights.len() {
            self.weights[w] += delta.weights()[w];
        }
        self.bias += delta.bias;
    }

    pub fn bias(&self) -> &f64 {
        return &self.bias;
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use crate::node::Node;

    #[test]
    fn test_apply_changes() {
        let mut n1 = Node {
            weights: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.2,
        };
        let delta = Node {
            weights: vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            bias: 0.8,
        };
        n1.apply_changes(&delta);

        assert_eq!(
            n1,
            Node {
                weights: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                bias: 1.0
            }
        );
    }
}
