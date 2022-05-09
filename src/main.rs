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

fn train_network() -> Result<(), Box<dyn Error>> {
    // simple_logging::log_to_file("weather_train.log", LevelFilter::Debug);

    // TODO have a very small change in weight when doing batches (so that it approximates continuous)
    let mut weather_data = WeatherData::new("training_data/Niskayuna NY 1970-01-02 to 2022-05-18.csv").unwrap();
    weather_data.load_data(RecordsSetting::All);
    weather_data.shuffle_data();

    let mut weather_net = WeatherNeuralNetwork::new(vec![15, 12, 10], weather_data.records(), 0.5)?;

    // Train the network while keeping track of its error
    let examples = weather_net.create_training_examples();
    let mut error_over_epochs = Vec::new();


    // TODO run the network multiple times on the same dataset (many different patterns)
    for (epoch, example) in examples.iter().enumerate() {
        let (input, expected_output) =  example;

        let (error, outputs) = weather_net.train_iter(&input, &expected_output);
        println!("Predicted next day conditions {:?}, Actual next day {:?}", weather_net.neural_output_to_conditions(&outputs), weather_net.neural_output_to_conditions(&expected_output));
        println!("output {:?}, expected {:?}", &outputs, &expected_output);
        println!("\n\nnn input \n{:?}", input);

        let b1 = std::io::stdin().read_line(&mut String::from(""));
        error_over_epochs.push((epoch as f32, error.clone()));
        print_progress(epoch, (epoch as f32)/ (examples.len() as f32), error);
    }

    // keep track of the total squared error over the patterns
    // TODO compute the error over all the patterns (do batch updates instead of pattern update)
    // TODO apply the weight deltas at the end
    plot_line(&error_over_epochs, "error-eval.png");
    Ok(())
}

fn cross_validate() {

}

fn main() {
    train_network();
}
