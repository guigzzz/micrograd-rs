use std::fs::File;

use nn::MultiLayerPerceptron;
use rand::{seq::SliceRandom, thread_rng};

use crate::nn::Mean;

mod engine;
mod nn;

fn main() {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    let mut mlp = MultiLayerPerceptron::new(Vec::from([2, 2]));

    let xy = &vec![
        (vec![1., 0.], 1.),
        (vec![0., 1.], 1.),
        (vec![1., 1.], 0.),
        (vec![0., 0.], 0.),
    ];

    let epochs = 25000;
    for i in 0..epochs {
        let mut xy = xy.clone();
        xy.shuffle(&mut thread_rng());

        let (acc, loss): (Vec<f64>, Vec<f64>) = xy
            .iter()
            .map(|(x, y)| {
                let y_pred = mlp.forward(x)[0];

                let loss = 0.5 * (y_pred - y).powf(2.);

                let d_loss = y_pred - y;

                mlp.zero_grads();
                mlp.backward(d_loss);
                mlp.update_weights(0.1);

                let acc = if (y_pred > 0.5) == (*y > 0.5) {
                    1.0
                } else {
                    0.0
                };

                (acc, loss)
            })
            .unzip();

        if i % 100 == 0 {
            println!(
                "Epoch {i} - Acc={:?}, Loss={:?}",
                acc.iter().mean(),
                loss.iter().mean()
            );
        }
    }

    let acc = xy
        .iter()
        .map(|(x, y)| {
            let y_pred = mlp.forward(x)[0];

            let acc = if (y_pred > 0.5) == (*y > 0.5) {
                1.0
            } else {
                0.0
            };

            acc
        })
        .mean();

    println!("{:?}", acc);

    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
    };
}
