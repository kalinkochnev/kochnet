use std::error::Error;
use std::fs::File;
use crate::weather_record::WeatherRecord;



pub enum RecordsSetting {
    All, // Retrieves all records
    FirstAmount(usize), // First number of records regardless of cleaned status
    Clean, // Retrieves all clean records
    MaxCleanAmount(usize), // Maximum number of cleaned records to attempt to retrieve
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

    fn load_first_amount(&self, rows: usize) -> Result<(), Box<dyn Error>> {
        let mut count = 0;

        for result in self.reader.deserialize() {
            if count == rows {
                break;
            }
            count += 1;
            let record: WeatherRecord = result?;
            println!("{:?}", record)
        }
        return Ok(());
    }

    pub fn load_data(&mut self, setting: RecordsSetting) -> Result<(), Box<dyn Error>>{
        match setting {
            RecordsSetting::FirstAmount(rows) => return self.load_first_amount(rows),
            _ => {todo!()}
        }
    }
}

#[cfg(test)]
mod tests {

}