use chrono::Datelike;

use crate::{
    weather_data::{RecordsSetting, WeatherData},
    weather_record::WeatherRecord,
    KochNET::KochNET,
};
use std::{error::Error, collections::HashMap, cmp::Ordering};

pub struct WeatherNeuralNetwork<'a> {
    pub data: &'a Vec<WeatherRecord>,
    conditions: Vec<String>, // the index of the weather condition corresponds to the equivalent neural network input
    pub network: KochNET,
}

impl<'a> WeatherNeuralNetwork<'a> {
    pub fn new(
        hidden_layer_sizes: Vec<usize>,
        weather_data: &'a Vec<WeatherRecord>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut weather_net = Self {
            data: weather_data,
            network: KochNET::empty(),
            conditions: Vec::new(),
        };
        weather_net.network.set_learn_rate(0.01);

        // initialize weather types so layer size
        weather_net.init_weather_conditions();
        weather_net.network.set_activation_func(&KochNET::sigmoid);

        // set layer sizes
        let mut layers = vec![weather_net.input_size()];
        layers.extend(hidden_layer_sizes);
        layers.push(weather_net.conditions.len());
        weather_net.network.reconfigure(layers);

        return Ok(weather_net);
    }

    pub fn conditions(&self) -> &Vec<String> {
        return &self.conditions;
    }

    fn init_weather_conditions(&mut self) {
        for record in self.data.iter() {
            let weather_condition = &record.conditions;
            if !self.conditions.contains(weather_condition) {
                self.conditions.push(weather_condition.clone());
            }
        }
    }

    fn output_size(&self) -> usize {
        let first_record = self
            .data
            .first()
            .expect("Weather neural network needs at least one data sample");
        return self.record_to_neural_output(first_record).len();
    }

    fn input_size(&self) -> usize {
        let first_record = self
            .data
            .first()
            .expect("Weather neural network needs at least one data sample");
        return self.record_to_neural_input(first_record).len();
    }

    /* Outputs the errors of the output and the output of the network*/
    pub fn train_iter(&mut self, input: &Vec<f32>, expected_output: &Vec<f32>) -> (f32, Vec<f32>) {
        return self.network.train_iter(input, expected_output);
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

    /* Returns a float representation of a weather condition to be used for training in the neural network */
    fn condition_to_input(&self, weather_condition: &String) -> f32 {
        return self.conditions.iter().position(|item| item == weather_condition).unwrap() as f32;
    }

    pub fn record_to_neural_output(&self, record: &WeatherRecord) -> Vec<f32> {
        let condition_index = self.conditions.iter().position(|item| record.conditions == *item).unwrap();
        let mut neural_output_equiv = vec![0.0; condition_index];
        neural_output_equiv.push(1.0);
        neural_output_equiv.extend(vec![0.0; self.conditions().len() - neural_output_equiv.len()]);

        return neural_output_equiv;
    }

    pub fn neural_output_to_type(&self, output: Vec<f32>) -> &String {
        let index_of_max: Option<usize> = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index);

        return self.conditions.get(index_of_max.unwrap()).unwrap();
    }

    pub fn create_training_examples(&self) -> Vec<(Vec<f32>, Vec<f32>)>{
        let mut examples = Vec::new();
        for record_index in 0..self.data.len() - 1 {
            let curr_record = self.data.get(record_index).unwrap();
            let next_record = self.data.get(record_index + 1).unwrap();
            examples.push((self.record_to_neural_input(curr_record), self.record_to_neural_output(next_record)));
        }
        return examples;
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use crate::{weather_record::WeatherRecord, weather_nn::WeatherNeuralNetwork};


    fn new_rec() -> WeatherRecord {
        WeatherRecord {
            name: String::from("some town"), 
            datetime: NaiveDate::from_ymd(2022, 5, 3),
            tempmax: 0.0,
            tempmin: 0.0,
            temp: 0.0,
            dew: 0.0,
            humidity: 0.0,
            precip: 0.0,
            windspeed: 0.0,
            winddir: 0.0,
            sealevelpressure: 0.0,
            cloudcover: 0.0,
            visibility: 0.0,
            conditions: String::from("fog"),
            description: String::from("some rain and stuff"),
            icon: String::from("sun-icon"),
        }
    }

    fn record_fixture() -> Vec<WeatherRecord> {
        let mut r1 = new_rec();
        r1.conditions = String::from("sunny and rainy");
        let r1_dup = r1.clone();

        let mut r2 = new_rec();
        r2.conditions = String::from("thunderstorm");

        let mut r3 = new_rec();
        r3.conditions = String::from("tornado");

        vec![r1, r1_dup, r2, r3]
    }

    #[test]
    fn test_input_output_conversion() {
        let records = record_fixture();
        let weather_nn = WeatherNeuralNetwork::new(vec![3, 2, 1], &records).unwrap();
    
        assert_eq!(weather_nn.conditions().len(), 3); // there should only be 3 unique conditions based on the data
        
        assert_eq!(weather_nn.condition_to_input(&String::from("thunderstorm")), 1.0);
        assert_eq!(weather_nn.neural_output_to_type(vec![0.0, 1.0, 0.0]), "thunderstorm");
    }

}