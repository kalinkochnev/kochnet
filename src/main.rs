pub mod KochNET;
pub mod node;
pub mod plotting;
pub mod weather_data;
pub mod weather_nn;
pub mod weather_record;

use std::error::Error;

use log::{debug, LevelFilter};
use plotting::plot_line;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use weather_data::WeatherData;
use weather_nn::WeatherNeuralNetwork;

use crate::weather_data::RecordsSetting;

fn print_progress(epoch: usize, perc_completion: f64, current_error: f64) {
    let num_ticks = 30;
    let times_repeat = (num_ticks as f64 * perc_completion - 1.0).round() as usize;

    if epoch % 1 == 0 {
        // clear the screen and then print the completion progress
        if std::process::Command::new("clear")
            .status()
            .unwrap()
            .success()
        {
            let progess = "=".repeat(times_repeat) + ">";

            let completion_str = format!(
                "[{:<30}] {:.3}% Epoch {} ----------- RMSE {}",
                progess,
                100.0 * perc_completion,
                epoch,
                current_error
            );
            println!("{}", completion_str);
            debug!("{}", completion_str);
        }
    }
}

fn train_network(is_running: Arc<AtomicBool>) -> Result<(), Box<dyn Error>> {
    // simple_logging::log_to_file("weather_train.log", LevelFilter::Debug);

    // TODO have a very small change in weight when doing batches (so that it approximates continuous)
    let mut weather_data =
        WeatherData::new("training_data/Niskayuna NY 1970-01-02 to 2022-05-18.csv").unwrap();
    weather_data.load_data(RecordsSetting::All);
    let mut weather_net = WeatherNeuralNetwork::new(vec![15, 12, 10], weather_data.records(), 0.00001)?;

    // Train the network while keeping track of its error
    let mut examples = weather_net.create_training_examples();
    let num_epochs = 10000;
    let error_over_epochs = Arc::new(Mutex::new(Vec::new()));

    weather_net.train(num_epochs, &mut examples, is_running, &|epoch, rmse| {
        error_over_epochs.lock().unwrap().push((epoch as f64, rmse.clone()));
        print_progress(epoch, (epoch as f64) / (num_epochs as f64), rmse);
    });
    plot_line(&error_over_epochs.lock().unwrap(), "error-eval.png");
    Ok(())
}

fn handle_ctrl_c() -> Arc<AtomicBool> {
    let running = Arc::new(AtomicBool::new(true));
    let r1 = running.clone();
    let r2 = running.clone();

    ctrlc::set_handler(move || {
        r1.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // println!("Waiting for Ctrl-C...");
    // while running.load(Ordering::SeqCst) {}
    // println!("Got it! Exiting...");

    return r2;
}

fn main() {
    let running_status = handle_ctrl_c();
    train_network(running_status);
}
