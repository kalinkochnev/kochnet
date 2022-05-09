// This lets us write `#[derive(Deserialize)]`.
use serde::Deserialize;

use chrono::{NaiveDate};

// https://github.com/chronotope/chrono
// https://serde.rs/custom-date-format.html
// https://docs.rs/csv/1.1.6/csv/tutorial/index.html#reading-csv

#[derive(Debug, Deserialize, Clone)]
pub struct WeatherRecord {
    pub name: String,
    #[serde(with = "YYYY_MM_DDFormat")]
    pub datetime: NaiveDate,
    pub tempmax: f64,
    pub tempmin: f64,
    pub temp: f64,
    // feelslikemax,
    // feelslikemin,
    // feelslike,
    pub dew: f64,
    pub humidity: f64,
    pub precip: f64,
    // precipprob,
    // precipcover,
    // preciptype,
    // snow,
    // snowdepth,
    // windgust,
    pub windspeed: f64,
    pub winddir: f64,
    pub sealevelpressure: f64,
    pub cloudcover: f64,
    pub visibility: f64,
    // solarradiation,
    // solarenergy,
    // uvindex,
    // severerisk,
    // sunrise,
    // sunset,
    // moonphase,
    pub conditions: String,
    pub description: String,
    pub icon: String,
    // stations
}

mod YYYY_MM_DDFormat {
    use chrono::{NaiveDate};
    use serde::{self, Deserialize, Serializer, Deserializer};

    const FORMAT: &'static str = "%Y-%m-%d";
    // The signature of a serialize_with function must follow the pattern:
    //
    //    fn serialize<S>(&T, S) -> Result<S::Ok, S::Error>
    //    where
    //        S: Serializer
    //
    // although it may also be generic over the input types T.
    pub fn serialize<S>(date: &NaiveDate, serializer: S,) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", date.format(FORMAT));
        serializer.serialize_str(&s)
    }

    // The signature of a deserialize_with function must follow the pattern:
    //
    //    fn deserialize<'de, D>(D) -> Result<T, D::Error>
    //    where
    //        D: Deserializer<'de>
    //
    // although it may also be generic over the output types T.
    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<NaiveDate, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        NaiveDate::parse_from_str(&s, FORMAT).map_err(serde::de::Error::custom)
    }
}