use crate::engine::Data;

pub trait Optimiser {
    fn optimise(&mut self, data: &mut Vec<Data>);
}

pub struct AdamOptimiser {
    m: Vec<f64>,
    v: Vec<f64>,
    t: f64,
}

impl AdamOptimiser {
    const ALPHA: f64 = 0.001;
    const BETA_1: f64 = 0.9;
    const BETA_2: f64 = 0.999;
    const EPSILON: f64 = 1e-8;

    pub fn new(num_params: usize) -> Self {
        AdamOptimiser {
            m: vec![0.; num_params],
            v: vec![0.; num_params],
            t: 0.,
        }
    }
}

impl Optimiser for AdamOptimiser {
    fn optimise(&mut self, data: &mut Vec<Data>) {
        self.t += 1.;

        self.m
            .iter_mut()
            .zip(self.v.iter_mut())
            .zip(data.iter_mut())
            .for_each(|((m, v), d)| {
                let grad: f64 = d.gradient;

                *m = Self::BETA_1 * *m + (1. - Self::BETA_1) * grad;
                *v = Self::BETA_2 * *v + (1. - Self::BETA_2) * grad.powf(2.);

                let beta1 = Self::BETA_1.powf(self.t);
                let beta2 = Self::BETA_2.powf(self.t);
                let alpha = Self::ALPHA * (1. - beta2).sqrt() / (1. - beta1);

                d.value -= alpha * *m / (v.sqrt() + Self::EPSILON)
            });
    }
}

pub struct LearningRateOptimiser {
    learning_rate: f64,
}

impl LearningRateOptimiser {
    pub fn new(learning_rate: f64) -> LearningRateOptimiser {
        LearningRateOptimiser { learning_rate }
    }
}

impl Optimiser for LearningRateOptimiser {
    fn optimise(&mut self, data: &mut Vec<Data>) {
        data.iter_mut().for_each(|v| {
            v.value -= self.learning_rate * v.gradient;
        });
    }
}
