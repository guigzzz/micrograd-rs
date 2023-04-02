use std::{fs::File, path::Path};

use nn::MultiLayerPerceptron;
use rand::{seq::SliceRandom, thread_rng};

use crate::util::{Mean, Util};
use data::Mnist;

mod data;
mod engine;
mod nn;
mod util;

fn main() {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    let mnist = Mnist::from_parquet(Path::new("mnist.parquet"));

    let mut mlp = MultiLayerPerceptron::new(vec![mnist.x_dim, 32, mnist.y_dim]);

    let epochs = 100;
    for i in 0..epochs {
        let mut xy = mnist.as_xy().clone();
        xy.shuffle(&mut thread_rng());

        let (acc, loss): (Vec<f64>, Vec<f64>) = xy
            .iter()
            .map(|(x, y)| {
                let y_preds = mlp.forward(x);

                let max = y_preds.iter().max_by(|l, r| l.total_cmp(r)).unwrap();
                let sum_exp = y_preds.iter().map(|y| (y - max).exp()).sum::<f64>();
                let softmax: Vec<_> = y_preds.iter().map(|y| (y - max).exp() / sum_exp).collect();

                // https://deepnotes.io/softmax-crossentropy
                let grads: Vec<f64> = softmax
                    .iter()
                    .enumerate()
                    .map(|(i, y_pred)| {
                        let y = if (*y as usize) == i { 1. } else { 0. };
                        y_pred - y
                    })
                    .collect();

                mlp.zero_grads();
                mlp.backward(grads);
                mlp.update_weights(0.001);

                let acc = if Util::argmax(&y_preds) == *y as usize {
                    1.0
                } else {
                    0.0
                };

                let loss = -softmax
                    .iter()
                    .enumerate()
                    .map(|(i, sm)| {
                        let y = if (*y as usize) == i { 1. } else { 0. };
                        y * sm.log10()
                    })
                    .sum::<f64>();

                (acc, loss)
            })
            .unzip();

        if i % 10 == 0 {
            println!(
                "Epoch {i} - Acc={:?}, Loss={:?}",
                acc.iter().mean(),
                loss.iter().mean()
            );
        }
    }

    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
    };
}
