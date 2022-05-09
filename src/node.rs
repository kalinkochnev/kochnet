use rand::Rng;


#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    weights: Vec<f32>,
    bias: f32
}
impl Node {
    pub fn new(num_weights: &usize) -> Node {
        let mut weights: Vec<f32> = vec![];
        for i in 0..*num_weights as u32 {
            weights.push(rand::thread_rng().gen_range(-1.0..1.0));
        }
        Node {
            weights: weights,
            bias: rand::thread_rng().gen_range(-1.0..1.0)
        }
    }

    pub fn weights(&self) -> &Vec<f32> {
        return &self.weights;
    }

    pub fn weights_mut(&mut self) -> &mut Vec<f32> {
        return &mut self.weights;
    }

    

    /* Calculates the change in weights and returns a node
    object with how much to change the weights and bias.
    Accepts the previous layers activations */
    // pub fn delta_weights_output(&self, activations: Vec<f32> ) -> Node {
    //     let weights_iter = self.weights.iter().chain(vec![self.bias].iter());

    //     let new_weights = Vec::new();
    //     for  in self.weights.iter().enumerate() {
            
    //     }
    // }

    pub fn bias(&self) -> &f32 {
        return &self.bias;
    }

    pub fn set_bias(&mut self, bias: f32)  {
        self.bias = bias;
    }
}