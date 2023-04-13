use cluster_event::{
    clust_analysis, clust_analysis_cutoff, load_hdf5,
    slow_ref::{clust_analysis_skip, clust_analysis_with_ring},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn criterion_benchmark(c: &mut Criterion) {
    //let hits = load_hdf5("Measurement_Mar_16_2023_13h01m57s.hdf5").unwrap();
    let hits = load_hdf5("./example_measurement.hdf5").unwrap();

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(hits.len() as u64));
    group.bench_function("base_line", |b| b.iter(|| clust_analysis(&hits, 5, 500e-9)));
    
    group.bench_function("cutoff_10", |b| {
        b.iter(|| clust_analysis_cutoff(&hits, 5, 500e-9, 10))
    });
    
    group.bench_function("cutoff_5", |b| {
        b.iter(|| clust_analysis_cutoff(&hits, 5, 500e-9, 5))
    });
    
    group.bench_function("cutoff_1", |b| {
        b.iter(|| clust_analysis_cutoff(&hits, 5, 500e-9, 1))
    });

    group.bench_function("ring", |b| {
        b.iter(|| clust_analysis_with_ring(&hits, 5, 500e-9))
    });
    group.bench_function("skip", |b| b.iter(|| clust_analysis_skip(&hits, 5, 500e-9)));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
//criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[allow(dead_code)]
fn bench_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation");
    let hits = load_hdf5("example_measurement.hdf5").unwrap();
    for i in 1..=10 {
        let test: usize = (i.to_owned() * hits.len() / 10) as usize;
        group.throughput(Throughput::Elements(test as u64));
        group.bench_with_input(BenchmarkId::new("Linear", i), &i, |b, _i| {
            b.iter(|| clust_analysis(&hits[0..test], 5, 500e-9))
        });
        group.bench_with_input(BenchmarkId::new("Iterative", i), &i, |b, _i| {
            b.iter(|| clust_analysis_with_ring(&hits[0..test], 5, 500e-9))
        });
    }
    group.finish();
}
