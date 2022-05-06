pub mod KochNET;
pub mod node;
pub mod plotting;
pub mod weather_data;
pub mod weather_nn;
pub mod weather_record;

use std::error::Error;

use plotting::plot_line;
use weather_data::WeatherData;
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
    let mut weather_data = WeatherData::new("training_data/Niskayuna NY 1970-01-02 to 2022-05-18.csv").unwrap();
    weather_data.load_data(RecordsSetting::FirstAmount(20))?;

    let weather_net = WeatherNeuralNetwork::new(vec![5, 5], weather_data.records())?;

    let mut error_over_epochs = Vec::new();
    let epochs: Vec<f32> = (0..error_over_epochs.len()).map(|x| x as f32).collect();

    // Train the network while keeping track of its error
    for (epoch, example) in weather_data.records().iter().enumerate() {
        let input = weather_net.record_to_neural_input(example);
        let expected_output: Vec<f32> = Vec::new(); // TODO set what the expect output is

        let (error, outputs) = weather_net.train_iter(&input, &expected_output);
        error_over_epochs.push(error);
    }
    plot_line(&epochs, &error_over_epochs, "error-eval.png");
    Ok(())
}
