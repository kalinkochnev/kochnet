
#[derive(Debug, PartialEq)]
pub struct Node {
    weights: Vec<f32>,
}
impl Node {
    pub fn new(num_weights: &usize) -> Node {
        Node {
            weights: vec![0.0; *num_weights],
        }
    }

    pub fn weights(&self) -> &Vec<f32> {
        return &self.weights;
    }
}