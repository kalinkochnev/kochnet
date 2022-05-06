use chrono::Datelike;

use crate::{
    weather_data::{RecordsSetting, WeatherData},
    weather_record::WeatherRecord,
    KochNET::KochNET,
};
use std::error::Error;

pub struct WeatherNeuralNetwork {
    pub data: WeatherData,
    conditions: Vec<String>, // the index of the weather condition corresponds to the equivalent neural network input
    pub network: KochNET,
}

impl WeatherNeuralNetwork {
    pub fn new(
        hidden_sizes: Vec<usize>,
        data_path: &str,
        data_settings: RecordsSetting,
    ) -> Result<Self, Box<dyn Error>> {
        // Load in the weather data
        let mut weather_data = WeatherData::new(data_path).unwrap();
        weather_data.load_data(data_settings)?;

        let weather_net = Self {
            data: weather_data,
            network: KochNET::empty(),
            conditions: Vec::new(),
        };

        // initialize weather types so layer size
        weather_net.init_weather_conditions();

        // set layer sizes
        weather_net.network.reconfigure(vec![]);
        return Ok(weather_net);
    }

    fn init_weather_conditions(&self) {
        for weather in self.conditions.iter() {
            self.condition_to_input(weather);
        }
    }

    fn output_size(&self) -> usize {
        let first_record = self
            .data
            .records()
            .first()
            .expect("Weather neural network needs at least one data sample");
        return self.record_to_neural_output(first_record).len();
    }

    fn input_size(&self) -> usize {
        let first_record = self
            .data
            .records()
            .first()
            .expect("Weather neural network needs at least one data sample");
        return self.record_to_neural_input(first_record).len();
    }

    /* Returns a float representation of a weather condition to be used for training in the neural network */
    fn condition_to_input(&self, weather_condition: &String) -> f32 {
        match self
            .conditions
            .iter()
            .position(|item| item == weather_condition)
        {
            Some(item_index) => return item_index as f32,
            None => {
                self.conditions.push(weather_condition.clone());
                return (self.conditions.len() - 1) as f32;
            }
        }
    }

    /* Outputs the errors of the output and the output of the network*/
    pub fn train_iter(&self, input: &Vec<f32>, expected_output: &Vec<f32>) -> (f32, &String) {
        let (errors, outputs) = self.network.train_iter(input, expected_output);
    }

    pub fn record_to_neural_input(&self, record: &WeatherRecord) -> Vec<f32> {
        vec![
            record.datetime.ordinal0() as f32,
            record.tempmax,
            record.tempmin,
            record.temp,
            record.dew,
            record.humidity,
            record.precip,
            record.windspeed,
            record.winddir,
            record.sealevelpressure,
            record.cloudcover,
            record.visibility,
            self.condition_to_input(&record.conditions),
        ]
    }

    pub fn record_to_neural_output(&self, record: &WeatherRecord) -> Vec<f32> {
        vec![self.condition_to_input(&record.conditions)]
    }

    pub fn neural_output_to_type(&self, output: f32) -> &String {
        return self.conditions.get(output as usize).unwrap();
    }
}
