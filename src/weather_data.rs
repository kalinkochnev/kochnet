use std::error::Error;
use std::fs::File;
use crate::weather_record::WeatherRecord;
use rand::thread_rng;
use rand::seq::SliceRandom;

pub enum RecordsSetting {
    All, // Retrieves all records
    FirstAmount(usize), // First number of records regardless of cleaned status
    Clean, // Retrieves all clean records
    MaxCleanAmount(usize), // Maximum number of cleaned records to attempt to retrieve
    RandomFirst(usize)
}
pub struct WeatherData {
    reader: csv::Reader<File>,
    records: Vec<WeatherRecord>,
}

impl WeatherData {
    pub fn new(filepath: &str) -> Result<Self, Box<dyn Error>> { 
        let file = File::open(filepath).expect("File could not be opened");
        Ok(Self {
            reader: csv::Reader::from_reader(file),
            records: Vec::new(),
        })
    }

    pub fn records(&self) -> &Vec<WeatherRecord> {
        return &self.records;
    }

    fn load_first_amount(&mut self, rows: usize) {
        let mut count = 0;

        for result in self.reader.deserialize() {
            if count == rows {
                break;
            }
            count += 1;
            match result {
                Ok(record) => self.records.push(record),
                Error => {}
            };
        }
    }

    fn load_all(&mut self) {
        for result in self.reader.deserialize() {
            match result {
                Ok(record) => self.records.push(record),
                Error => {}
            };
        }
    }

    pub fn shuffle_data(&mut self) {
        self.records.shuffle(&mut thread_rng());
    }

    pub fn load_data(&mut self, setting: RecordsSetting) {
        match setting {
            RecordsSetting::FirstAmount(rows) => self.load_first_amount(rows),
            RecordsSetting::All => self.load_all(), 
            _ => {todo!()}
        }
    }
}

#[cfg(test)]
mod tests {

}