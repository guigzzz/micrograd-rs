use std::{cell::RefCell, rc::Rc};

use rand::Rng;

use crate::{
    engine::{GraphBuilder, IdGenerator, InputNode, NodeId, RunnableGraph},
    optimiser::Optimiser,
};

pub struct Neuron<'a> {
    op: GraphBuilder<'a>,
}

impl<'a> Neuron<'a> {
    fn new(inputs: Vec<GraphBuilder<'a>>, non_linearity: bool) -> Neuron<'a> {
        let factory = inputs.first().unwrap();

        let mut rng = rand::thread_rng();

        let weights: Vec<GraphBuilder> = inputs
            .iter()
            .map(|i| &factory.create_immediate(rng.gen_range(-1.0..1.)) * i)
            .collect();

        let mut first = weights[0].clone();
        let tail = &weights[1..];

        for g in tail {
            first = first + g.clone();
        }

        let bias = factory.create_immediate(rng.gen_range(-1.0..1.));

        let output_value = first + bias;
        let output_value = if non_linearity {
            output_value.relu()
        } else {
            output_value
        };

        Neuron { op: output_value }
    }
}

#[derive(Debug)]
pub struct MultiLayerPerceptron {
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    graph: RunnableGraph,
}

impl MultiLayerPerceptron {
    pub fn new(sizes: Vec<usize>) -> MultiLayerPerceptron {
        let ids = &mut IdGenerator::new();
        let ids = Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);

        let num_inputs = sizes[0];
        let inputs: Vec<InputNode> = (0..num_inputs).map(|_| graph.create_input()).collect();

        let builders: Vec<GraphBuilder> = inputs.into_iter().map(|i| i.into()).collect();

        let outputs = sizes
            .iter()
            .enumerate()
            .skip(1)
            .fold(builders.clone(), |b, (i, s)| {
                let non_linearity = i != sizes.len() - 1;
                (0..*s)
                    .map(|_| Neuron::new(b.clone(), non_linearity).op)
                    .collect()
            });

        MultiLayerPerceptron {
            inputs: builders.iter().map(|i| i.root.into()).collect(),
            outputs: outputs.iter().map(|o| o.root).collect(),
            graph: RunnableGraph::new(outputs.iter().collect()),
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
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
            .for_each(|(input, value)| self.graph.set_input(*input, *value));

        self.graph.evaluate(&self.outputs)
    }

    pub fn backward(&mut self, out_grads: Vec<f64>) {
        let pairs: Vec<(NodeId, f64)> = self
            .outputs
            .clone()
            .into_iter()
            .zip(out_grads.into_iter())
            .collect();
        self.graph.backwards(pairs);
    }

    pub fn zero_grads(&mut self) {
        self.graph.zero_grads();
    }

    pub fn update_weights(&mut self, optimiser: &mut impl Optimiser) {
        self.graph.update_weights(optimiser);
    }

    pub fn num_parameters(&self) -> usize {
        self.graph.num_parameters()
    }
}

#[cfg(test)]
mod tests {

    use rand::{seq::SliceRandom, thread_rng};

    use crate::{
        nn::*,
        optimiser::LearningRateOptimiser,
        util::{Mean, Util},
    };

    #[test]
    fn test_mlp_xor() {
        let xy = &vec![
            (vec![1., 0.], vec![0., 1.]),
            (vec![0., 1.], vec![0., 1.]),
            (vec![1., 1.], vec![1., 0.]),
            (vec![0., 0.], vec![1., 0.]),
        ];

        let mut mlp = MultiLayerPerceptron::new(Vec::from([xy[0].0.len(), 2, xy[0].1.len()]));

        let optimiser = &mut LearningRateOptimiser::new(0.1);

        let epochs = 1000;
        for i in 0..epochs {
            let mut xy = xy.clone();
            xy.shuffle(&mut thread_rng());

            let (acc, loss): (Vec<f64>, Vec<f64>) = xy
                .iter()
                .map(|(x, y)| {
                    let y_preds = mlp.forward(x);

                    let loss = y
                        .iter()
                        .zip(y_preds.iter())
                        .map(|(y, y_pred)| (y_pred - y).powf(2.))
                        .sum::<f64>();

                    let grads: Vec<f64> = y
                        .iter()
                        .zip(y_preds.iter())
                        .map(|(y, y_pred)| (y_pred - y))
                        .collect();

                    mlp.zero_grads();
                    mlp.backward(grads);
                    mlp.update_weights(optimiser);

                    let acc = if Util::argmax(&y_preds) == Util::argmax(&y) {
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
                let y_preds = mlp.forward(x);
                let acc = if Util::argmax(&y_preds) == Util::argmax(&y) {
                    1.0
                } else {
                    0.0
                };

                acc
            })
            .mean();

        assert_eq!(acc, 1.0)
    }
}
