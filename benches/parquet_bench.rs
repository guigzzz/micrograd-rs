use std::path::Path;

use micrograd_rs::data::Mnist;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("from_parquet", |b| {
        b.iter(|| Mnist::from_parquet(black_box(Path::new("mnist.parquet"))))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
