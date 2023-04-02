use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use std::collections::HashSet;
use std::{fs::File, path::Path};

pub struct Mnist {
    images: Vec<Vec<f64>>,
    labels: Vec<u32>,
    pub x_dim: usize,
    pub y_dim: usize,
}

impl Mnist {
    pub fn from_parquet(path: &Path) -> Mnist {
        if let Ok(file) = File::open(&path) {
            let reader = SerializedFileReader::new(file).unwrap();

            let mut iter = reader.get_row_iter(None).unwrap();

            let mut images: Vec<Vec<f64>> = vec![];
            let mut labels: Vec<u32> = vec![];
            while let Some(record) = iter.next() {
                for (name, field) in record.get_column_iter() {
                    match name.as_str() {
                        "data" => match field {
                            Field::ListInternal(l) => {
                                let vals: Vec<f64> = l
                                    .elements()
                                    .iter()
                                    .map(|f| match f {
                                        Field::Double(f) => *f,
                                        f => panic!("Unexpected array value type: {:?}", f),
                                    })
                                    .collect();
                                images.push(vals);
                            }
                            f => panic!("Unexpcted type for data field: {:?}", f),
                        },
                        "labels" => match field {
                            Field::Long(i) => labels.push(*i as u32),
                            f => panic!("Unexpcted type for labels field: {:?}", f),
                        },
                        n => panic!("Unexpected column: {:?}", n),
                    }
                }
            }

            let x_dim = images[0].len();
            let y_dim = labels.iter().collect::<HashSet<_>>().len();

            return Mnist {
                images,
                labels,
                x_dim,
                y_dim,
            };
        }

        panic!()
    }

    pub fn as_xy(&self) -> Vec<(&Vec<f64>, u32)> {
        self.images
            .iter()
            .zip(self.labels.iter().cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use crate::data::*;

    #[test]
    fn test_mnist() {
        let path = Path::new("mnist.parquet");

        let mnist = Mnist::from_parquet(path);

        assert_eq!(mnist.images.len(), 1797);
        assert_eq!(mnist.labels.len(), 1797);
        assert_eq!(mnist.x_dim, 64);
        assert_eq!(mnist.y_dim, 10)
    }
}
