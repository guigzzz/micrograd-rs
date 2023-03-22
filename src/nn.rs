use std::{cell::RefCell, rc::Rc};

use crate::engine::{GraphBuilder, IdGenerator, InputNode, NodeId, RunnableGraph};

pub struct Neuron<'a> {
    op: GraphBuilder<'a>,
}

impl<'a> Neuron<'a> {
    fn new(inputs: Vec<GraphBuilder<'a>>) -> Neuron<'a> {
        let factory = inputs.first().unwrap();

        let weights: Vec<GraphBuilder> = inputs
            .iter()
            .map(|i| &factory.create_immediate(0.) * i)
            .collect();

        let mut first = weights[0].clone();
        let tail = &weights[1..];

        for g in tail {
            first = first + g.clone();
        }

        let bias = factory.create_immediate(0.);

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

    use crate::nn::*;

    #[test]
    fn test_mlp() {
        let mut mlp = MultiLayerPerceptron::new(Vec::from([2]));

        let x = vec![vec![1., 0.], vec![0., 1.]];

        let y = &vec![-1., 1.];

        let epochs = 100;
        for i in 0..epochs {
            let preds: Vec<f64> = x.iter().map(|x| mlp.forward(x)).collect();

            let acc = (&preds)
                .iter()
                .zip(y)
                .map(|(y_pred, y)| {
                    if (*y_pred > 0.) == (*y > 0.) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .mean();

            let loss = (&preds)
                .iter()
                .zip(y)
                .map(|(y_pred, y)| {
                    let v = 1. - y_pred * y;
                    if v > 0. {
                        v
                    } else {
                        0.
                    }
                })
                .mean();

            println!("Epoch {i} - Acc={acc}, Loss={loss}");

            mlp.backward(loss);

            let learning_rate = 1.0 - 0.9 * (i as f64) / 100.;
            mlp.output.update_weights(learning_rate)
        }
    }
}
