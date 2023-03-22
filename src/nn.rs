use std::{cell::RefCell, cmp::min_by, rc::Rc};

use rand::Rng;
use std::cmp::max_by;

use crate::engine::{GraphBuilder, IdGenerator, InputNode, NodeId, RunnableGraph};

pub struct Neuron<'a> {
    op: GraphBuilder<'a>,
}

impl<'a> Neuron<'a> {
    fn max(left: f64, right: f64) -> f64 {
        max_by(left, right, |l, r| l.partial_cmp(r).unwrap())
    }

    fn min(left: f64, right: f64) -> f64 {
        min_by(left, right, |l, r| l.partial_cmp(r).unwrap())
    }

    fn new(inputs: Vec<GraphBuilder<'a>>) -> Neuron<'a> {
        let factory = inputs.first().unwrap();

        let mut rng = rand::thread_rng();

        let weights: Vec<GraphBuilder> = inputs
            .iter()
            .map(|i| {
                let v = Self::max(Self::min(rng.gen(), 1.), -1.);
                // let v = 0.1;
                &factory.create_immediate(v) * i
            })
            .collect();

        let mut first = weights[0].clone();
        let tail = &weights[1..];

        for g in tail {
            first = first + g.clone();
        }

        let v = Self::max(Self::min(rng.gen(), 1.), -1.);
        // let v = 0.1;
        let bias = factory.create_immediate(v);

        Neuron {
            op: (first + bias).relu(),
        }
    }
}

#[derive(Debug)]
pub struct MultiLayerPerceptron {
    inputs: Vec<NodeId>,
    output: RunnableGraph,
}

impl MultiLayerPerceptron {
    fn new(sizes: Vec<u32>) -> MultiLayerPerceptron {
        let ids = &mut IdGenerator::new();
        let ids = Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);

        let num_inputs = sizes[0];
        let inputs: Vec<InputNode> = (0..num_inputs).map(|_| graph.create_input()).collect();

        let builders: Vec<GraphBuilder> = inputs.into_iter().map(|i| i.into()).collect();

        let outputs = sizes.iter().skip(1).fold(builders.clone(), |b, s| {
            (0..*s).map(|_| Neuron::new(b.clone()).op).collect()
        });

        let output = Neuron::new(outputs).op;

        MultiLayerPerceptron {
            inputs: builders.iter().map(|i| i.root.into()).collect(),
            output: output.make(),
        }
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> f64 {
        if inputs.len() != self.inputs.len() {
            panic!(
                "Expected {} inputs, but got {}",
                self.inputs.len(),
                inputs.len()
            )
        }
        self.inputs
            .iter()
            .zip(inputs.iter())
            .for_each(|(input, value)| self.output.set_input(*input, *value));

        self.output.forward()
    }

    fn backward(&mut self, out_grad: f64) {
        self.output.backwards(out_grad);
    }
}

pub trait Mean {
    fn mean(self) -> f64;
}

impl<F, T> Mean for T
where
    T: Iterator<Item = F>,
    F: std::borrow::Borrow<f64>,
{
    fn mean(self) -> f64 {
        self.zip(1..).fold(0., |s, (e, i)| {
            (*e.borrow() + s * (i - 1) as f64) / i as f64
        })
    }
}

#[cfg(test)]
mod tests {

    use rand::{seq::SliceRandom, thread_rng};

    use crate::nn::*;

    #[test]
    fn test_mlp() {
        let mut mlp = MultiLayerPerceptron::new(Vec::from([2, 2]));

        dbg!(&mlp);

        let xy = &vec![
            (vec![1., 0.], 1.),
            (vec![0., 1.], 1.),
            (vec![1., 1.], 0.),
            (vec![0., 0.], 0.),
        ];

        let epochs = 1000;
        for i in 0..epochs {
            let mut xy = xy.clone();
            xy.shuffle(&mut thread_rng());

            let (acc, loss): (Vec<f64>, Vec<f64>) = xy
                .iter()
                .map(|(x, y)| {
                    let y_pred = mlp.forward(x);

                    let loss = 0.5 * (y_pred - y).powf(2.);

                    let d_loss = y_pred - y;

                    mlp.output.zero_grads();
                    mlp.backward(d_loss);
                    mlp.output.update_weights(0.1);

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
    }
}
