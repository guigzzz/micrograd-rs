use std::{
    collections::HashMap,
    ops::{Add, Mul},
};

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Mul,
    Add,
    Sub,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

#[derive(Debug, Clone, Copy)]
pub struct GraphBuilderNode {
    operation: Operation,
    left_id: NodeId,
    right_id: NodeId,
}

#[derive(Debug, Clone, Copy)]
pub enum Node {
    Operation(GraphBuilderNode),
    Input(f64),
    Immediate(f64),
}

#[derive(Debug)]
pub struct IdGenerator {
    current_id: usize,
}

impl IdGenerator {
    fn get_id(&mut self) -> NodeId {
        self.current_id += 1;
        NodeId(self.current_id)
    }

    fn new() -> IdGenerator {
        IdGenerator { current_id: 0 }
    }
}

#[derive(Debug)]
pub struct GraphBuilder<'a> {
    root: NodeId,
    nodes: HashMap<NodeId, Node>,
    ids: &'a mut IdGenerator,
}

impl<'a> GraphBuilder<'a> {
    pub fn combine(
        op: Operation,
        left: &'a mut GraphBuilder,
        right: &GraphBuilder,
    ) -> GraphBuilder<'a> {
        let new_root = GraphBuilderNode {
            operation: op,
            left_id: left.root,
            right_id: right.root,
        };

        let mut nodes = left.nodes.clone();
        nodes.extend(right.nodes.clone());

        let id = left.ids.get_id();
        nodes.insert(id, Node::Operation(new_root));

        GraphBuilder {
            root: id,
            nodes,
            ids: left.ids,
        }
    }

    pub fn with_immediate(
        op: Operation,
        left: &'a mut GraphBuilder,
        right_val: f64,
    ) -> GraphBuilder<'a> {
        let mut nodes = left.nodes.clone();

        let id = left.ids.get_id();
        nodes.insert(id, Node::Immediate(right_val));

        let new_root = GraphBuilderNode {
            operation: op,
            left_id: left.root,
            right_id: id,
        };

        let root_id = left.ids.get_id();
        nodes.insert(root_id, Node::Operation(new_root));

        GraphBuilder {
            root: root_id,
            nodes,
            ids: left.ids,
        }
    }

    pub fn new(ids: &mut IdGenerator) -> GraphBuilder {
        GraphBuilder {
            root: NodeId(0),
            nodes: HashMap::new(),
            ids,
        }
    }

    pub fn create_input(&mut self) -> NodeId {
        let id = self.ids.get_id();

        self.nodes.insert(id, Node::Input(f64::default()));
        self.root = id;

        id as NodeId
    }

    pub fn set_input(&mut self, inp: NodeId, val: f64) {
        let node = self.nodes.get_mut(&inp).unwrap();
        match node {
            Node::Input(v) => *v = val,
            _ => panic!("Unexpected node type"),
        }
    }

    fn get_node(&self, id: NodeId) -> &Node {
        self.nodes.get(&id).unwrap()
    }

    pub fn get_value(&self) -> f64 {
        let root = self.get_node(self.root);
        self.get_value_for_node(root)
    }

    pub fn get_value_for_node(&self, node: &Node) -> f64 {
        match node {
            Node::Operation(n) => {
                let left_val = self.get_value_for_node(self.get_node(n.left_id));
                let right_val = self.get_value_for_node(self.get_node(n.right_id));
                match n.operation {
                    Operation::Mul => left_val * right_val,
                    Operation::Add => left_val + right_val,
                    Operation::Sub => left_val - right_val,
                    Operation::Div => left_val / right_val,
                }
            }
            Node::Input(v) => *v,
            Node::Immediate(v) => *v,
        }
    }
}

impl<'a> Add<&mut GraphBuilder<'a>> for &'a mut GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: &mut GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Add, self, rhs)
    }
}

impl<'a> Add<f64> for &'a mut GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self, rhs)
    }
}

impl<'a> Mul<&mut GraphBuilder<'a>> for &'a mut GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: &mut GraphBuilder) -> Self::Output {
        GraphBuilder::combine(Operation::Mul, self, rhs)
    }
}

impl<'a> Mul<f64> for &'a mut GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Mul, self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::*;

    #[test]
    fn test_graph_builder() {
        let ids = &mut IdGenerator::new();

        let graph = &mut GraphBuilder::new(ids);
        let input = graph.create_input();

        let mut g = graph + 1.;

        assert_eq!(g.get_value(), 1.);

        g.set_input(input, 1.);
        assert_eq!(g.get_value(), 2.);

        let mut g = &mut g * 4.;

        g.set_input(input, 2.);
        assert_eq!(g.get_value(), 12.);
    }

    #[test]
    fn test_graph_builder_basic() {
        let ids = &mut IdGenerator::new();
        let graph = &mut GraphBuilder::new(ids);

        let input = graph.create_input();

        graph.set_input(input, 2.5);
        assert_eq!(graph.get_value(), 2.5);

        graph.set_input(input, 3.5);
        assert_eq!(graph.get_value(), 3.5);
    }

    // #[test]
    // fn test_value() {
    //     let mut value = Value { data: 2.5 };
    //     let mut input = Graph::Input(&value);

    //     let first = &input + 1.5;
    //     let second = &first + 3.5;

    //     assert_eq!(second.get_value(), 14.5);

    //     // value.data = 3.5;
    //     // assert_eq!(second.get_value(), 15.5);
    // }
}

// use std::rc::Rc;

// pub struct Value {
//     data: f64,
//     // grad: f64,
//     // children: Option<(Box<Value>, Box<Value>)>,
// }

// #[derive(Debug)]
// pub struct GraphBuilderInput {
//     id: usize,
// }

// #[derive(Debug)]
// pub struct GraphBuilderInputNode {
//     id: usize,
// }

// pub struct OpAdd {
//     left: Graph,
//     right: Graph,
// }

// pub enum Graph<'a> {
//     Operation(Operation, &'a Graph<'a>, &'a Graph<'a>),
//     OperationImmediate(Operation, &'a Graph<'a>, f64),
//     Input(&'a Value),
// }

// impl Graph<'_> {
//     pub fn get_value(&self) -> f64 {
//         match self {
//             Graph::Operation(op, left, right) => {
//                 let left_val = left.get_value();
//                 let right_val = right.get_value();
//                 match op {
//                     Operation::Mul => left_val * right_val,
//                     Operation::Add => left_val + right_val,
//                     Operation::Sub => left_val - right_val,
//                     Operation::Div => left_val / right_val,
//                 }
//             }
//             Graph::OperationImmediate(op, left, right) => {
//                 let left_val = left.get_value();
//                 let right_val = *right;
//                 match op {
//                     Operation::Mul => left_val * right_val,
//                     Operation::Add => left_val + right_val,
//                     Operation::Sub => left_val - right_val,
//                     Operation::Div => left_val / right_val,
//                 }
//             }
//             Graph::Input(v) => v.data,
//         }
//     }

//     pub fn new<'a>(v: &'a Value) -> Graph<'a> {
//         Graph::Input(v)
//     }
// }

// impl Value {
//     fn new(v: f64) -> Self {
//         let v = Value {
//             // data: v,
//             // grad: 0.,
//             children: None,
//         };
//         v
//     }

//     fn new_from_children(v: f64, left: Value, right: Value) -> Self {
//         let children = Some((Box::from(left), Box::from(right)));
//         let v = Value {
//             // data: v,
//             // grad: 0.,
//             children,
//         };
//         v
//     }
// }

// impl<'a> Add<&'a Graph<'a>> for &'a Graph<'a> {
//     type Output = Graph<'a>;

//     fn add(self, rhs: &'a Graph<'a>) -> Self::Output {
//         return Graph::Operation(Operation::Add, self, rhs);
//     }
// }

// impl<'a> Add<f64> for &'a Graph<'a> {
//     type Output = Graph<'a>;

//     fn add(self, rhs: f64) -> Self::Output {
//         return Graph::OperationImmediate(Operation::Add, self, rhs);
//     }
// }

// impl<'a> Mul<Graph<'a>> for Graph<'a> {
//     type Output = Graph<'a>;

//     fn mul(self, rhs: Graph<'a>) -> Self::Output {
//         return Graph::Operation(Operation::Mul, Rc::new(self), Rc::new(rhs));
//     }
// }

// impl<'a> Mul<f64> for Graph<'a> {
//     type Output = Graph<'a>;

//     fn mul(self, rhs: f64) -> Self::Output {
//         return Graph::Operation(
//             Operation::Mul,
//             Rc::new(self),
//             Rc::new(Graph::Immediate(rhs)),
//         );
//     }
// }

// impl Mul<Value> for Value {
//     type Output = Value;

//     fn mul(self, rhs: Value) -> Self::Output {
//         return Value::new_from_children(self.data * rhs.data, self, rhs);
//     }
// }

// impl Mul<f64> for Value {
//     type Output = Value;

//     fn mul(self, rhs: f64) -> Self::Output {
//         return Value::new_from_children(self.data * rhs, self, Value::new(rhs));
//     }
// }
