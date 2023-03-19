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

        let outputs = sizes[1..].iter().fold(builders.clone(), |b, s| {
            (0..*s).map(|_| Neuron::new(b.clone()).op).collect()
        });

        let output = Neuron::new(outputs).op;

        MultiLayerPerceptron {
            inputs: builders.iter().map(|i| i.root.into()).collect(),
            output: output.make(),
        }
    }

    fn forward(&mut self, inputs: Vec<f64>) -> f64 {
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
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        engine::{IdGenerator, InputNode},
        nn::*,
    };

    #[test]
    fn test_mlp() {
        let mut mlp = MultiLayerPerceptron::new(Vec::from([4, 3, 2]));

        let value = mlp.forward(Vec::from([1., 2., 3., 4.]));
        assert_eq!(value, 0.);
    }
}
