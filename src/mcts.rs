use petgraph::{Graph, Direction};
use petgraph::graph::NodeIndex;
use petgraph::Directed;
use std::ops::Index;
use std::f64;
use chess_utils::{State, get_legal_states, is_terminal};

struct Node {
    num_visited: i32,
    value: i32,
    state: State,
}

struct BestNodeIndex {
    node_index: NodeIndex,
    in_tree: bool,
    terminal: bool
}

pub fn run_mcts(state: State) {
    let (g, root_index) = create_mcts_graph(state);
    let best_node_index = find_node_maximizing_bound(&g, root_index);
}

fn create_mcts_graph(state: State) -> (Graph<Node, u32, Directed>, NodeIndex) {
    let mut g = Graph::<Node, u32, Directed>::new();
    let index = add_new_node(&mut g, None, state);
    (g, index)
}

fn add_new_node(g: &mut Graph<Node, u32, Directed>, parent: Option<NodeIndex>, state: State) -> NodeIndex {
    let index = g.add_node(Node {num_visited: 1, value: 0, state });
    match parent {
        None => {},
        Some(e) => {
            g.add_edge(e, index, 0);
        }
    }
    index
}

// TODO use value from neural net in equation
fn find_node_maximizing_bound(g: &Graph<Node, u32, Directed>, node_index: NodeIndex) -> BestNodeIndex {
    let children_indexes: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();
    let legal_states = get_legal_states(g.index(node_index).state);

    if children_indexes.len() == 0 && legal_states.len() == 0 { // Terminal node
        return BestNodeIndex {
            node_index,
            in_tree: false,
            terminal: true
        }
    }
    if children_indexes.len() == 0 { // Choose first node from legal_states
        return BestNodeIndex {
            node_index: legal_states[0],
            in_tree: false,
            terminal: false
        }
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
