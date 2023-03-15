use std::{
    cell::RefCell,
    collections::HashMap,
    ops::{Add, Mul},
    rc::Rc,
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct GraphBuilder<'a> {
    root: NodeRef,
    nodes: HashMap<NodeId, Node>,
    ids: &'a Rc<RefCell<&'a mut IdGenerator>>,
    inputs: HashMap<NodeId, f64>,
}

impl<'a> GraphBuilder<'a> {
    pub fn combine(op: Operation, left: GraphBuilder<'a>, right: GraphBuilder) -> GraphBuilder<'a> {
        let new_root = GraphBuilderNode {
            operation: op,
            left_id: left.root,
            right_id: right.root,
        };

        let mut nodes = left.nodes.clone();
        nodes.extend(right.nodes.clone());

        let id = left.ids.borrow_mut().get_id();
        nodes.insert(id, Node::Operation(new_root));

        let mut inputs = left.inputs.clone();
        inputs.extend(right.inputs.clone());

        GraphBuilder {
            root: NodeRef::OpNode(id),
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

        let mut ids = left.ids.borrow_mut();

        let id = ids.get_id();
        nodes.insert(id, Node::Immediate(right_val));

        let new_root = GraphBuilderNode {
            operation: op,
            left_id: left.root,
            right_id: NodeRef::OpNode(id),
        };

        let root_id = ids.get_id();
        nodes.insert(root_id, Node::Operation(new_root));

        GraphBuilder {
            root: NodeRef::OpNode(root_id),
            nodes,
            ids: left.ids,
            inputs: left.inputs.clone(),
        }
    }

    pub fn new(ids: &'a Rc<RefCell<&'a mut IdGenerator>>) -> GraphBuilder<'a> {
        GraphBuilder {
            root: NodeRef::OpNode(NodeId(0)),
            nodes: HashMap::new(),
            ids,
            inputs: HashMap::new(),
        }
    }

    pub fn new_immediate(&self, val: f64) -> GraphBuilder<'a> {
        let id = self.ids.borrow_mut().get_id();
        GraphBuilder {
            root: NodeRef::OpNode(id),
            nodes: HashMap::from([(id, Node::Immediate(val))]),
            ids: self.ids,
            inputs: HashMap::new(),
        }
    }

    pub fn create_input(&self) -> InputNode<'a> {
        let id = self.ids.borrow_mut().get_id();

        let mut inputs = self.inputs.clone();
        inputs.insert(id, f64::default());

        InputNode {
            id,
            builder: GraphBuilder {
                root: NodeRef::InputNode(id),
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
            NodeRef::InputNode(id) => match self.inputs.get(&id) {
                None => panic!("Failed to fetch input node with id {:?}", id),
                Some(v) => *v,
            },
            NodeRef::OpNode(id) => match self.nodes.get(&id) {
                None => panic!("Failed to fetch operation node with id {:?}", id),
                Some(n) => self.get_value_for_node(n),
            },
        }
    }

    pub fn get_value(&self) -> f64 {
        self.get_node_value(self.root)
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

impl<'a> Add<&GraphBuilder<'a>> for &GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: &GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Add, self.clone(), rhs.clone())
    }
}

impl<'a> Add<f64> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self, rhs, false)
    }
}

impl<'a> Add<&InputNode<'a>> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: &InputNode<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Add, self, rhs.builder.clone())
    }
}

impl<'a> Add<f64> for InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self.builder, rhs, true)
    }
}

impl<'a> Add<f64> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self.builder.clone(), rhs, true)
    }
}

impl<'a> Add<GraphBuilder<'a>> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Add, self.builder.clone(), rhs.clone())
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
        let ids = &Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);
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

    #[test]
    fn test_graph_builder_multiple_inputs() {
        let ids = &mut IdGenerator::new();
        let ids = &Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);
        let input1 = &graph.create_input();
        let id1 = input1.id;
        let input2 = &graph.create_input();
        let id2 = input2.id;

        let g1 = input1 + 1.;
        let g2 = input2 + 2.;

        let mut g = g1 + g2 + input1;

        g.set_input(id1, 1.5);
        g.set_input(id2, 2.5);
        assert_eq!(g.get_value(), 8.5);
    }
}
