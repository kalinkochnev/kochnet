pub mod KochNET;
pub mod node;
pub mod plotting;
pub mod weather_data;
pub mod weather_nn;
pub mod weather_record;

use std::error::Error;

use plotting::plot_line;
use weather_nn::WeatherNeuralNetwork;

use crate::weather_data::{RecordsSetting};

fn average(errors: &Vec<f32>) -> f32 {
    let mut sum = 0.0;
    for err in errors.iter() {
        sum += err;
    }
    return sum / (errors.len() as f32);
}

fn main() -> Result<(), Box<dyn Error>> {
    let weather_net = WeatherNeuralNetwork::new(
        vec![5, 5],
        "training_data/Niskayuna NY 1970-01-02 to 2022-05-18.csv",
        RecordsSetting::FirstAmount(20),
    )?;

    let error_over_epochs = Vec::new();
    let epochs: Vec<f32> = (0..error_over_epochs.len()).map(|x| x as f32).collect();
    let training_data = weather_net.data;

    for (epoch, example) in training_data.records().iter().enumerate() {
        let input = weather_net.record_to_neural_input(example);
        let expected_output = None;

        let (error, outputs) = weather_net.train_iter();
        error_over_epochs.push(error);
    }
    plot_line(&epochs, &error_over_epochs, "error-eval.png");
    Ok(())
}
