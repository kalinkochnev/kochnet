pub mod KochNET;
pub mod node;
pub mod plotting;
pub mod weather_data;
pub mod weather_nn;
pub mod weather_record;

use core::num;
use std::error::Error;

use log::LevelFilter;
use plotting::plot_line;
use weather_data::WeatherData;
use weather_nn::WeatherNeuralNetwork;

use crate::weather_data::{RecordsSetting};


fn print_progress(epoch: usize, perc_completion: f32, current_error: f32) {
    let num_ticks = 30;
    let times_repeat = (num_ticks as f32 * perc_completion - 1.0).round() as usize;

    if epoch % num_ticks == 0 {
        // clear the screen and then print the completion progress
        if std::process::Command::new("clear").status().unwrap().success() {
            let progess = "=".repeat(times_repeat) + ">";

            let completion_str = format!("[{:<30}] {:.3}% Epoch {} ----------- RMSE {}", progess, 100.0*perc_completion, epoch, current_error);
            println!("{}", completion_str);
        }
    }
    
}

fn main() -> Result<(), Box<dyn Error>> {
    // simple_logging::log_to_file("weather_train.log", LevelFilter::Debug);

    let mut weather_data = WeatherData::new("training_data/Niskayuna NY 1970-01-02 to 2022-05-18.csv").unwrap();
    weather_data.load_data(RecordsSetting::All);
    weather_data.shuffle_data();

    let mut weather_net = WeatherNeuralNetwork::new(vec![7, 5], weather_data.records())?;

    // Train the network while keeping track of its error
    let examples = weather_net.create_training_examples();
    let mut error_over_epochs = Vec::new();


    for (epoch, example) in examples.iter().enumerate() {
        let (input, expected_output) =  example;

        let (error, outputs) = weather_net.train_iter(&input, &expected_output);
        error_over_epochs.push((epoch as f32, error.clone()));
        print_progress(epoch, (epoch as f32)/ (examples.len() as f32), error);
    }

    plot_line(&error_over_epochs, "error-eval.png");
    Ok(())
}
