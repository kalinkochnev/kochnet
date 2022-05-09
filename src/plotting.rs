use plotters::prelude::*;

pub fn plot_line(data: &Vec<(f64, f64)>, filename: &str)-> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Weather Condition Prediction Error", ("sans-serif", (5).percent_height()))
        .x_label_area_size(20f64)
        .y_label_area_size(40f64)// .margin((1).percent())
        .build_cartesian_2d(0f64..data.len() as f64, 0f64..1f64)?;

    chart
        .configure_mesh()
        .x_desc("Epoch #")
        .y_desc("Squared Error")
        .draw()?;


    chart.draw_series(LineSeries::new(
        data.iter().map(|(x, y)| (*x, *y)),
        &BLUE,
    ))?;


    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", filename);

    Ok(())
}