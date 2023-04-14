use clap::Parser;
use cluster_event::{clust_analysis, clust_analysis_cutoff, load_hdf5, write_hdf5, Event};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
#[derive(Parser, Debug)]
#[command(author, version, about = "Evaluates Cluster from Electron Microskopy", long_about = None)]
struct Args {
    /// HDF5 File
    #[arg(
        short,
        long,
        default_value_t = ("./example_measurement.hdf5").to_string()
    )]
    file: String,

    /// Run benchmark
    #[arg(short, long, default_value_t = false)]
    bench: bool,

    /// Maximum Pixel distance
    #[arg(short = 'p', long, default_value_t = 5)]
    eps_pixel: u16,

    /// Maximum Time distance [s]
    #[arg(short = 't', long, default_value_t = 500e-9)]
    eps_time: f64,

    /// Length of the Buffer
    #[arg(short, long, default_value_t = 0)]
    cutoff: usize,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    println!("File: {}", args.file);
    if args.bench {
        calc_sum(&args.file, args.eps_pixel, args.eps_time, args.cutoff)?;
        return Ok(());
    }
    let hits: Vec<Event> = load_hdf5(&args.file).unwrap();
    if args.cutoff == 0 {
        let clusters = clust_analysis(&hits, args.eps_pixel, args.eps_time);
        println!("Found {} clusters", clusters.len());
        write_hdf5("clusters.hdf5", &clusters);
    } else {
        let clusters = clust_analysis_cutoff(&hits, args.eps_pixel, args.eps_time, args.cutoff);
        println!("Found {} clusters", clusters.len());
        write_hdf5("clusters.hdf5", &clusters);
    }
    return Ok(());
}

fn calc_sum(path: &str, eps_pixel: u16, eps_time: f64, cutoff: usize) -> std::io::Result<()> {
    println!("Run time measurment");
    let hits: Vec<Event> = load_hdf5(path).unwrap();
    let num_average = 5;
    let mut file: File = File::create("speed.csv")?;
    write!(file, "num_hits freq[MHz] time[microsec]\n")?;
    for len in 1..=100 {
        let length = len * hits.len() / 100;
        let sliced_hits = &hits[1..length];

        let duration = if cutoff > 0 {
            println!("Calculating with {} as cutoff", cutoff);
            let start = Instant::now();
            for _ in 0..num_average {
                let _dat = clust_analysis_cutoff(sliced_hits, eps_pixel, eps_time, cutoff);
            }
            start.elapsed() / num_average
        } else {
            let start = Instant::now();
            for _ in 0..num_average {
                let _dat = clust_analysis(sliced_hits, eps_pixel, eps_time);
            }
            start.elapsed() / num_average
        };
        let freq = (sliced_hits.len() as f64) / (duration.as_micros() as f64);
        write!(file, "{} {:?} {:?}\n", length, freq, duration.as_micros())?;
    }
    return Ok(());
}
