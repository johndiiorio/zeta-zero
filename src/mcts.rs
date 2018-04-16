use petgraph::{Graph, Direction};
use petgraph::graph::NodeIndex;
use petgraph::Directed;
use std::ops::Index;
use std::f64;

pub struct Node {
    num_visited: i32,
    value: i32,
    fen: String,
}

pub fn create_mcts_graph(fen: String) -> (Graph<Node, u32, Directed>, NodeIndex) {
    let mut g = Graph::<Node, u32, Directed>::new();
    let index = add_new_node(&mut g, None, fen);
    (g, index)
}

pub fn add_new_node(g: &mut Graph<Node, u32, Directed>, parent: Option<NodeIndex>, fen: String) -> NodeIndex {
    let index = g.add_node(Node {num_visited: 1, value: 0, fen });
    match parent {
        None => {},
        Some(e) => {
            g.add_edge(e, index, 0);
        }
    }
    index
}

// TODO use value from neural net in equation
pub fn find_node_maximizing_bound(g: &Graph<Node, u32, Directed>, node_index: NodeIndex) -> NodeIndex {
    let children_indexes: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();
    if children_indexes.len() == 0 {
        return node_index
    }

    let mut max_value = 0 as f64;
    let mut best_node_index = children_indexes[0];

    let mut total_children_visits = 0;
    for _ in &children_indexes {
        total_children_visits += g.index(node_index).num_visited;
    }
    for child_index in children_indexes {
        let node = g.index(node_index);
        let value = (node.value / node.num_visited) as f64 +
            (2 as f64).sqrt() * (total_children_visits as f64).sqrt() / (1 + g.index(child_index).num_visited) as f64;
        if value > max_value {
            max_value = value;
            best_node_index = child_index;
        }
    }
    best_node_index
}