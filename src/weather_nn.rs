use chrono::Datelike;
use lazy_static::lazy_static;
use log::debug;
use crate::{
    weather_data::{RecordsSetting, WeatherData},
    weather_record::WeatherRecord,
    KochNET::{KochNET, Layer}, node::Node,
};
use std::{error::Error, collections::HashMap, cmp::Ordering};
use regex::Regex;

pub struct WeatherNeuralNetwork<'a> {
    pub data: &'a Vec<WeatherRecord>,
    conditions: Vec<String>, // the index of the weather condition corresponds to the equivalent neural network input
    pub network: KochNET,
}

impl<'a> WeatherNeuralNetwork<'a> {
    pub fn new(
        hidden_layer_sizes: Vec<usize>,
        weather_data: &'a Vec<WeatherRecord>,
        learn_rate: f64,
    ) -> Result<Self, Box<dyn Error>> {
        let mut weather_net = Self {
            data: weather_data,
            network: KochNET::empty(),
            conditions: Vec::new(),
        };
        weather_net.network.set_learn_rate(learn_rate);

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
            let weather_conditions = self.parse_conditions(&record.conditions);
            
            for cond in weather_conditions.iter() {
                if !self.conditions.contains(cond) {
                    self.conditions.push(cond.clone());
                }
            }

        }
        debug!("{:?}", self.conditions());
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
    pub fn train_iter(&mut self, input: &Vec<f64>, expected_output: &Vec<f64>) -> (f64, Vec<Layer>, Vec<f64>) {
        return self.network.train_iter(input, expected_output);
    }

    pub fn record_to_neural_input(&self, record: &WeatherRecord) -> Vec<f64> {
        let mut neural_input = vec![
            record.datetime.ordinal0() as f64,
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
        ];
        neural_input.extend(self.conditions_to_input(&record.conditions));
        return neural_input;
    }

    /* Returns a float representation of a weather condition to be used for training in the neural network */
    fn conditions_to_input(&self, weather_condition: &String) -> Vec<f64> {
        let conditions = self.parse_conditions(weather_condition);
        let mut output = vec![0.0; self.conditions.len()];

        for (i, poss_condition) in self.conditions().iter().enumerate() {
            if conditions.contains(poss_condition) {
                output[i] = 1.0;
            }
        }
        return output;
    }

    pub fn record_to_neural_output(&self, record: &WeatherRecord) -> Vec<f64> {
        return self.conditions_to_input(&record.conditions);
    }

    fn parse_conditions(&self, condition: &str) -> Vec<String> {
        fn remove_trailing_whitespace(word: &str) -> String {
            lazy_static! {
                // matches any comma separated values (and ignores leading/trailing whitespaces)
                static ref RE: Regex = Regex::new(r"^[ \t]+|[ \t]+$").unwrap();
            };

            let result = RE.replace_all(word, "");
            return result.into_owned();
        }
        
        let mut conditions = Vec::new();

        for cond in condition.split(",") {
            conditions.push(remove_trailing_whitespace(cond));
        }
        
        return conditions;
    }

    pub fn neural_output_to_conditions(&self, output: &Vec<f64>) -> Vec<String> {
        const THRESHOLD: f64 = 0.50;

        let  mut conditions = Vec::new();
        for (index, activation) in output.iter().enumerate() {
            if activation >= &THRESHOLD {
                conditions.push(self.conditions.get(index).unwrap().clone());
            }
        }

        return conditions;
    }

    pub fn create_training_examples(&self) -> Vec<(Vec<f64>, Vec<f64>)>{
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
        r1.conditions = String::from("sunny, rainy");
        let r1_dup = r1.clone();

        let mut r2 = new_rec();
        r2.conditions = String::from("thunderstorm, clouds");

        let mut r3 = new_rec();
        r3.conditions = String::from("tornado");

        vec![r1, r1_dup, r2, r3]
    }

    #[test]
    fn test_parse_conditions() {
        let records = record_fixture();
        let mut weather_nn = WeatherNeuralNetwork::new(vec![3, 2, 1], &records, 0.05).unwrap();
        let result = weather_nn.parse_conditions("rain, snow, chance of tornados");
        assert_eq!(result, vec!["rain", "snow", "chance of tornados"])
    }

    #[test]
    fn test_input_output_conversion() {
        let records = record_fixture();
        let mut weather_nn = WeatherNeuralNetwork::new(vec![3, 2, 1], &records, 0.05).unwrap();
        weather_nn.init_weather_conditions();

        assert_eq!(weather_nn.conditions().len(), 5); // there should only be 3 unique conditions based on the data
        
        assert_eq!(weather_nn.conditions_to_input(&String::from("thunderstorm, tornado, sunny")), vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(weather_nn.neural_output_to_conditions(&vec![1.0, 0.0, 1.0, 0.0, 1.0]), vec![String::from("sunny"), String::from("thunderstorm"), String::from("tornado")]);
    }

    #[test]
    fn test_training_examples() {
        let records = record_fixture();

        let mut weather_nn = WeatherNeuralNetwork::new(vec![3, 2, 1], &records, 0.05).unwrap();
        weather_nn.init_weather_conditions();
    
        let examples = weather_nn.create_training_examples();
        
        let expected = vec![
            (weather_nn.record_to_neural_input(&records[0]), weather_nn.record_to_neural_output(&records[1])),
            (weather_nn.record_to_neural_input(&records[1]), weather_nn.record_to_neural_output(&records[2])),
        
        ];

        for (ex, expected) in examples.iter().zip(expected.iter()) {
            assert_eq!(ex, expected);
        }
    }
}