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
pub enum NodeRef {
    OpNode(NodeId),
    InputNode(NodeId),
}

#[derive(Debug, Clone, Copy)]
pub struct GraphBuilderNode {
    operation: Operation,
    left_id: NodeRef,
    right_id: NodeRef,
}

#[derive(Debug)]
pub struct InputNode<'a> {
    id: NodeId,
    builder: GraphBuilder<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum Node {
    Operation(GraphBuilderNode),
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
    inputs: HashMap<NodeId, f64>,
}

impl<'a> GraphBuilder<'a> {
    pub fn combine(op: Operation, left: GraphBuilder<'a>, right: GraphBuilder) -> GraphBuilder<'a> {
        let new_root = GraphBuilderNode {
            operation: op,
            left_id: NodeRef::OpNode(left.root),
            right_id: NodeRef::OpNode(right.root),
        };

        let mut nodes = left.nodes.clone();
        nodes.extend(right.nodes.clone());

        let id = left.ids.get_id();
        nodes.insert(id, Node::Operation(new_root));

        let mut inputs = left.inputs.clone();
        inputs.extend(right.inputs.clone());

        GraphBuilder {
            root: id,
            nodes,
            ids: left.ids,
            inputs,
        }
    }

    pub fn with_immediate(
        op: Operation,
        left: GraphBuilder<'a>,
        right_val: f64,
        is_input: bool,
    ) -> GraphBuilder<'a> {
        let mut nodes = left.nodes.clone();

        let id = left.ids.get_id();
        nodes.insert(id, Node::Immediate(right_val));

        let left_node = if is_input {
            NodeRef::InputNode(left.root)
        } else {
            NodeRef::OpNode(left.root)
        };
        let new_root = GraphBuilderNode {
            operation: op,
            left_id: left_node,
            right_id: NodeRef::OpNode(id),
        };

        let root_id = left.ids.get_id();
        nodes.insert(root_id, Node::Operation(new_root));

        GraphBuilder {
            root: root_id,
            nodes,
            ids: left.ids,
            inputs: left.inputs.clone(),
        }
    }

    pub fn new(ids: &mut IdGenerator) -> GraphBuilder {
        GraphBuilder {
            root: NodeId(0),
            nodes: HashMap::new(),
            ids,
            inputs: HashMap::new(),
        }
    }

    pub fn new_immediate(&'a mut self, val: f64) -> GraphBuilder<'a> {
        let id = self.ids.get_id();
        GraphBuilder {
            root: id,
            nodes: HashMap::from([(id, Node::Immediate(val))]),
            ids: self.ids,
            inputs: HashMap::new(),
        }
    }

    pub fn create_input(&'a mut self) -> InputNode<'a> {
        let id = self.ids.get_id();

        let mut inputs = self.inputs.clone();
        inputs.insert(id, f64::default());

        InputNode {
            id,
            builder: GraphBuilder {
                root: id,
                nodes: self.nodes.clone(),
                ids: self.ids,
                inputs,
            },
        }
    }

    pub fn set_input(&mut self, inp: NodeId, val: f64) {
        let node = self.inputs.get_mut(&inp).unwrap();
        *node = val;
    }

    fn get_node_value(&self, id: NodeRef) -> f64 {
        match id {
            NodeRef::InputNode(id) => {
                match self.inputs.get(&id) {
                    None => panic!("Failed to fetch input node with id {:?}", id),
                    Some(v) => *v,
                }
                // *self.inputs.get(&id).unwrap()
            }
            NodeRef::OpNode(id) => self.get_value_for_node(self.nodes.get(&id).unwrap()),
        }
    }

    pub fn get_value(&self) -> f64 {
        self.get_node_value(NodeRef::OpNode(self.root))
    }

    fn get_value_for_node(&self, node: &Node) -> f64 {
        match node {
            Node::Operation(n) => {
                let left_val = self.get_node_value(n.left_id);
                let right_val = self.get_node_value(n.right_id);
                match n.operation {
                    Operation::Mul => left_val * right_val,
                    Operation::Add => left_val + right_val,
                    Operation::Sub => left_val - right_val,
                    Operation::Div => left_val / right_val,
                }
            }
            Node::Immediate(v) => *v,
        }
    }
}

impl<'a> Add<GraphBuilder<'a>> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Add, self, rhs)
    }
}

impl<'a> Add<f64> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self, rhs, false)
    }
}

impl<'a> Add<f64> for InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self.builder, rhs, true)
    }
}

impl<'a> Mul<GraphBuilder<'a>> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: GraphBuilder) -> Self::Output {
        GraphBuilder::combine(Operation::Mul, self, rhs)
    }
}

impl<'a> Mul<f64> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Mul, self, rhs, false)
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::*;

    #[test]
    fn test_graph_builder() {
        let ids = &mut IdGenerator::new();

        let mut graph = GraphBuilder::new(ids);
        let input = graph.create_input();
        let id = input.id;

        let mut g = input + 1.;

        assert_eq!(g.get_value(), 1.);

        g.set_input(id, 1.);
        assert_eq!(g.get_value(), 2.);

        let mut g = g * 4.;

        g.set_input(id, 2.);
        assert_eq!(g.get_value(), 12.);
    }

    // #[test]
    // fn test_graph_builder_multiple_inputs() {
    //     let ids = &mut IdGenerator::new();

    //     let mut graph = GraphBuilder::new(ids);
    //     let input1 = graph.create_input();
    //     let input2 = graph.create_input();
    // }

    // #[test]
    // fn test_graph_builder_basic() {
    //     let ids = &mut IdGenerator::new();
    //     let graph = &mut GraphBuilder::new(ids);

    //     let input = graph.create_input();

    //     graph.set_input(input.id, 2.5);
    //     assert_eq!(graph.get_value(), 2.5);

    //     graph.set_input(input.id, 3.5);
    //     assert_eq!(graph.get_value(), 3.5);
    // }

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
