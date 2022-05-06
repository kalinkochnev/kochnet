use plotters::prelude::*;

pub fn plot_line(x: &Vec<f32>, y: &Vec<f32>, filename: &str)-> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(&filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Weather Condition Prediction Error", ("sans-serif", (5).percent_height()))
        .set_label_area_size(LabelAreaPosition::Left, (8).percent())
        .set_label_area_size(LabelAreaPosition::Bottom, (4).percent())
        .margin((1).percent())
        .build_cartesian_2d(0f32..10000f32, 0f32..100f32)?;

    chart
        .configure_mesh()
        .x_desc("Epoch #")
        .y_desc("Squared Error")
        .draw()?;


    let color = Palette99::pick(0).mix(0.9);
    chart.draw_series(LineSeries::new(
        x.iter().zip(y).map(|(x, y)| (*x, *y)),
        &BLUE,
    ))?;


    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", filename);

    Ok(())
}