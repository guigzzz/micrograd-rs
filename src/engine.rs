use std::{
    cell::RefCell,
    collections::HashMap,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use num::traits::Pow;

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Mul,
    Add,
    Sub,
    Div,
    Pow,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

#[derive(Debug, Clone, Copy)]
pub enum NodeRef {
    OpNode(NodeId),
    InputNode(NodeId),
}

impl NodeRef {
    fn as_op_node_id(self) -> Option<NodeId> {
        match self {
            Self::OpNode(id) => Some(id),
            _ => None,
        }
    }
}

impl Into<NodeId> for NodeRef {
    fn into(self) -> NodeId {
        match self {
            Self::OpNode(id) => id,
            Self::InputNode(id) => id,
        }
    }
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

impl<'a> Into<&'a GraphBuilder<'a>> for &'a InputNode<'a> {
    fn into(self) -> &'a GraphBuilder<'a> {
        &self.builder
    }
}

impl<'a> Into<GraphBuilder<'a>> for InputNode<'a> {
    fn into(self) -> GraphBuilder<'a> {
        self.builder
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImmediateNode {
    id: NodeId,
    original_value: f64,
}

impl ImmediateNode {
    pub fn new(id: NodeId, original_value: f64) -> ImmediateNode {
        ImmediateNode { id, original_value }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Node {
    Operation(GraphBuilderNode),
    Immediate(ImmediateNode),
}

impl Into<Option<GraphBuilderNode>> for &Node {
    fn into(self) -> Option<GraphBuilderNode> {
        match self {
            Node::Operation(n) => Some(*n),
            _ => None,
        }
    }
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

    pub fn new() -> IdGenerator {
        IdGenerator { current_id: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct GraphBuilder<'a> {
    pub root: NodeRef,
    nodes: HashMap<NodeId, Node>,
    ids: Rc<RefCell<&'a mut IdGenerator>>,
    inputs: HashMap<NodeId, f64>,
}

#[derive(Debug)]
pub struct Data {
    pub value: f64,
    pub gradient: f64,
}

impl Data {
    pub fn new(v: f64) -> Data {
        Data {
            value: v,
            gradient: 0.,
        }
    }
}

#[derive(Debug)]
pub struct RunnableGraph {
    root: NodeRef,
    nodes: HashMap<NodeId, Node>,
    inputs: HashMap<NodeId, f64>,
    data: HashMap<NodeId, Data>,
}

impl RunnableGraph {
    pub fn set_input(&mut self, inp: NodeId, val: f64) {
        let node = self.inputs.get_mut(&inp).unwrap();
        *node = val;
    }

    fn update_data_value(&mut self, id: NodeId, v: f64) {
        match self.data.get_mut(&id) {
            None => {
                let _ = self.data.insert(id, Data::new(v));
            }
            Some(d) => (*d).value = v,
        }
    }

    fn get_node_value(&mut self, id: NodeRef) -> f64 {
        match id {
            NodeRef::InputNode(id) => match self.inputs.get(&id) {
                None => panic!("Failed to fetch input node with id {:?}", id),
                Some(v) => *v,
            },
            NodeRef::OpNode(id) => match self.nodes.get(&id) {
                None => panic!("Failed to fetch operation node with id {:?}", id),
                Some(n) => {
                    let value = self.get_value_for_node(&n.clone());
                    self.update_data_value(id, value);
                    value
                }
            },
        }
    }

    pub fn forward(&mut self) -> f64 {
        self.get_node_value(self.root)
    }

    fn get_value_for_node(&mut self, node: &Node) -> f64 {
        match node {
            Node::Operation(n) => {
                let left_val = self.get_node_value(n.left_id);
                let right_val = self.get_node_value(n.right_id);
                match n.operation {
                    Operation::Mul => left_val * right_val,
                    Operation::Add => left_val + right_val,
                    Operation::Sub => left_val - right_val,
                    Operation::Div => left_val / right_val,
                    Operation::Pow => left_val.pow(right_val),
                    Operation::Relu => {
                        if left_val < 0. {
                            0.
                        } else {
                            left_val
                        }
                    }
                }
            }
            Node::Immediate(v) => {
                let value = self
                    .data
                    .get(&v.id)
                    .map(|d| d.value)
                    .unwrap_or(v.original_value);

                self.update_data_value(v.id, value);

                value
            }
        }
    }

    fn data_for_id_mut(&mut self, id: NodeId) -> &mut Data {
        self.data.get_mut(&id).unwrap()
    }

    fn data_for_id(&self, id: NodeId) -> &Data {
        self.data.get(&id).unwrap()
    }

    fn grad_for_id(&self, id: NodeId) -> f64 {
        self.data_for_id(id).gradient
    }

    fn value_for_id(&self, id: NodeRef) -> f64 {
        match id {
            NodeRef::InputNode(n) => {
                let v = self.inputs.get(&n).unwrap();
                *v
            }
            NodeRef::OpNode(n) => self.data_for_id(n).value,
        }
    }

    fn update(
        &mut self,
        id: NodeRef,
        operation: Operation,
        root_value: f64,
        root_grad: f64,
        other_value: f64,
    ) {
        {
            if id.as_op_node_id().is_none() {
                return;
            }

            let data = self.data_for_id_mut(id.into());
            match operation {
                Operation::Add => {
                    data.gradient += root_grad;
                }
                Operation::Mul => {
                    data.gradient += other_value * root_grad;
                }
                Operation::Relu => {
                    data.gradient += if root_value > 0. { 1.0 } else { 0.0 } * root_grad
                }
                v => todo!("{:?}", v),
            }
        }
    }

    fn _backwards(&mut self, id: NodeRef) {
        let root_value = self.value_for_id(id);

        let id = match id {
            NodeRef::OpNode(id) => id,
            _ => return,
        };

        let root_grad = self.grad_for_id(id);

        let node = self.nodes.get(&id).unwrap();
        let node: Option<GraphBuilderNode> = node.into();

        let _ = node.map(|node| {
            let right_value = self.value_for_id(node.right_id.into());
            self.update(
                node.left_id,
                node.operation,
                root_value,
                root_grad,
                right_value,
            );
            self._backwards(node.left_id.into());

            let left_value = self.value_for_id(node.left_id.into());
            self.update(
                node.right_id,
                node.operation,
                root_value,
                root_grad,
                left_value,
            );
            self._backwards(node.right_id.into());
        });
    }

    pub fn zero_grads(&mut self) {
        self.data.values_mut().for_each(|v| {
            v.gradient = 0.;
        })
    }

    pub fn backwards(&mut self, out_grad: f64) {
        let root_value = self.value_for_id(self.root);

        let operation = {
            let node: Option<GraphBuilderNode> = self.nodes.get(&self.root.into()).unwrap().into();
            node.unwrap().operation
        };

        self.update(self.root, operation, root_value, out_grad, 0.);

        self._backwards(self.root.into())
    }

    pub fn update_weights(&mut self, learning_rate: f64) {
        self.data.values_mut().for_each(|v| {
            v.value -= learning_rate * v.gradient;
        })
    }
}

impl<'a> GraphBuilder<'a> {
    fn combine(op: Operation, left: GraphBuilder<'a>, right: GraphBuilder) -> GraphBuilder<'a> {
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

    fn with_immediate(op: Operation, left: GraphBuilder<'a>, right_val: f64) -> GraphBuilder<'a> {
        let mut nodes = left.nodes.clone();

        let mut ids = left.ids.borrow_mut();

        let id = ids.get_id();
        nodes.insert(id, Node::Immediate(ImmediateNode::new(id, right_val)));

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
            ids: left.ids.clone(),
            inputs: left.inputs.clone(),
        }
    }

    fn with_immediate2(op: Operation, left: f64, right: GraphBuilder<'a>) -> GraphBuilder<'a> {
        let mut nodes = right.nodes.clone();

        let mut ids = right.ids.borrow_mut();

        let id = ids.get_id();
        nodes.insert(id, Node::Immediate(ImmediateNode::new(id, left)));

        let new_root = GraphBuilderNode {
            operation: op,
            left_id: NodeRef::OpNode(id),
            right_id: right.root,
        };

        let root_id = ids.get_id();
        nodes.insert(root_id, Node::Operation(new_root));

        GraphBuilder {
            root: NodeRef::OpNode(root_id),
            nodes,
            ids: right.ids.clone(),
            inputs: right.inputs.clone(),
        }
    }

    pub fn new(ids: Rc<RefCell<&'a mut IdGenerator>>) -> GraphBuilder<'a> {
        GraphBuilder {
            root: NodeRef::OpNode(NodeId(0)),
            nodes: HashMap::new(),
            ids,
            inputs: HashMap::new(),
        }
    }

    pub fn new_of_immediate(ids: Rc<RefCell<&'a mut IdGenerator>>, val: f64) -> GraphBuilder<'a> {
        let id = ids.borrow_mut().get_id();
        GraphBuilder {
            root: NodeRef::OpNode(id),
            nodes: HashMap::from([(id, Node::Immediate(ImmediateNode::new(id, val)))]),
            ids,
            inputs: HashMap::new(),
        }
    }

    pub fn create_immediate(&self, val: f64) -> GraphBuilder<'a> {
        let id = self.ids.borrow_mut().get_id();
        GraphBuilder {
            root: NodeRef::OpNode(id),
            nodes: HashMap::from([(id, Node::Immediate(ImmediateNode::new(id, val)))]),
            ids: self.ids.clone(),
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
                ids: self.ids.clone(),
                inputs,
            },
        }
    }

    pub fn make(&self) -> RunnableGraph {
        RunnableGraph {
            root: self.root,
            nodes: self.nodes.clone(),
            inputs: self.inputs.clone(),
            data: HashMap::new(),
        }
    }

    pub fn relu(self) -> GraphBuilder<'a> {
        GraphBuilder::with_immediate(Operation::Relu, self.clone(), 0.)
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
        GraphBuilder::with_immediate(Operation::Add, self, rhs)
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
        GraphBuilder::with_immediate(Operation::Add, self.builder, rhs)
    }
}

impl<'a> Add<f64> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, self.builder.clone(), rhs)
    }
}

impl<'a> Add<GraphBuilder<'a>> for f64 {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Add, rhs, self)
    }
}

impl<'a> Add<&InputNode<'a>> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn add(self, rhs: &InputNode<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Add, self.builder.clone(), rhs.builder.clone())
    }
}

impl<'a> Sub<&InputNode<'a>> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn sub(self, rhs: &InputNode<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Sub, self.builder.clone(), rhs.builder.clone())
    }
}

impl<'a> Sub<GraphBuilder<'a>> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn sub(self, rhs: GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Sub, self.clone(), rhs.clone())
    }
}

impl<'a> Neg for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn neg(self) -> Self::Output {
        GraphBuilder::with_immediate2(Operation::Sub, 0., self.builder.clone())
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

impl<'a> Mul<&GraphBuilder<'a>> for &GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: &GraphBuilder) -> Self::Output {
        GraphBuilder::combine(Operation::Mul, self.clone(), rhs.clone())
    }
}

impl<'a> Mul<GraphBuilder<'a>> for f64 {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Mul, rhs.clone(), self)
    }
}

impl<'a> Mul<&InputNode<'a>> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: &InputNode<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Mul, self.clone(), rhs.builder.clone())
    }
}

impl<'a> Mul<&InputNode<'a>> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn mul(self, rhs: &InputNode<'a>) -> Self::Output {
        GraphBuilder::combine(Operation::Mul, self.builder.clone(), rhs.builder.clone())
    }
}

impl<'a> Pow<f64> for &InputNode<'a> {
    type Output = GraphBuilder<'a>;

    fn pow(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Pow, self.builder.clone(), rhs)
    }
}

impl<'a> Pow<f64> for GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn pow(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Pow, self.clone(), rhs)
    }
}

impl<'a> Div<f64> for &GraphBuilder<'a> {
    type Output = GraphBuilder<'a>;

    fn div(self, rhs: f64) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Div, self.clone(), rhs)
    }
}

impl<'a> Div<&GraphBuilder<'a>> for f64 {
    type Output = GraphBuilder<'a>;

    fn div(self, rhs: &GraphBuilder<'a>) -> Self::Output {
        GraphBuilder::with_immediate(Operation::Div, rhs.clone(), self)
    }
}

impl<'a> Mul<f64> for GraphBuilder<'a> {
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
        let ids = Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);
        let input = &graph.create_input();

        let builder = input + 1.;

        let mut g = builder.make();

        assert_eq!(g.forward(), 1.);

        g.set_input(input.id, 1.);
        assert_eq!(g.forward(), 2.);

        let builder = builder * 4.;
        let mut g = builder.make();

        g.set_input(input.id, 2.);
        assert_eq!(g.forward(), 12.);
    }

    #[test]
    fn test_graph_builder_multiple_inputs() {
        let ids = &mut IdGenerator::new();
        let ids = Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);
        let input1 = &graph.create_input();
        let input2 = &graph.create_input();

        let g1 = input1 + 1.;
        let g2 = input2 + 2.;

        let g3 = &g1 + &g2 + input1;

        let g4 = &g1 * &g2 * input2;

        let g = g3 + g4;

        let mut g = g.make();

        g.set_input(input1.id, 1.5);
        g.set_input(input2.id, 2.5);
        assert_eq!(g.forward(), 36.625);
    }

    #[test]
    fn test_complex() {
        let ids = &mut IdGenerator::new();
        let ids = Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);
        let a = &graph.create_input();
        let b = &graph.create_input();

        let c = a + b;

        let d = a * b + b.pow(3.);

        let c = c + 1.;
        let c = 1. + c + -a;
        let d = d * 2. + (b + a).relu();

        let d = 3. * d + (b - a).relu();
        let e = c - d;
        let f = e.pow(2.);
        let g = &f / 2.0 + 10. / &f;

        let mut g = g.make();

        g.set_input(a.id, -4.);
        g.set_input(b.id, 2.);

        assert_eq!(g.forward(), 2.4);
    }

    #[test]
    fn test_back() {
        let ids = &mut IdGenerator::new();
        let ids = Rc::new(RefCell::new(ids));

        let graph = GraphBuilder::new(ids);
        let a = &graph.create_input();
        let b = &graph.create_input();

        let c = (a + b) * 2.;

        let c = c.relu();

        let g = &mut c.make();

        g.set_input(a.id, 1.);
        g.set_input(b.id, 2.);

        let v = g.forward();

        assert_eq!(v, 6.);

        g.backwards(2.);
    }
}
