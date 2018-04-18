use petgraph::{Graph, Direction};
use petgraph::graph::NodeIndex;
use petgraph::Directed;
use std::ops::Index;
use std::ops::IndexMut;
use std::f64;
use chess_utils::{NodeState, State};

struct Node {
    num_visited: i32,
    value: i32,
    state: NodeState,
}

struct MCTSData {
    value: i32,
    terminal: bool
}

pub fn run_mcts(state: NodeState) {
    let (g, root_index) = create_mcts_graph(state);
    let best_node_index = recurse_mcts(g, root_index);
}

fn create_mcts_graph(state: NodeState) -> (Graph<Node, u32, Directed>, NodeIndex) {
    let mut g = Graph::<Node, u32, Directed>::new();
    let index = add_new_node(&mut g, None, state);
    (g, index)
}

fn add_new_node(g: &mut Graph<Node, u32, Directed>, parent: Option<NodeIndex>, state: NodeState) -> NodeIndex {
    let index = g.add_node(Node {num_visited: 0, value: 0, state });
    match parent {
        None => {},
        Some(e) => {
            g.add_edge(e, index, 0);
        }
    }
    index
}

// TODO use value from neural net in equation and backpropagation
fn recurse_mcts(mut g: Graph<Node, u32, Directed>, node_index: NodeIndex) -> MCTSData {
    // Nodes in tree before additions
    let children_before_addition: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();
    let mut added_nodes = false;
    {
        let current_node = g.index_mut(node_index);
        if current_node.state.is_terminal() {
            return MCTSData {
                value: current_node.value,
                terminal: true
            }
        }


        // Increment the number of times that the current node was visited
        current_node.num_visited += 1;
    }

    let legal_states;
    let do_add =
    {
        let current_node = g.index(node_index);
        // All possible states of the current node
        legal_states = current_node.state.get_legal_states();
        current_node.num_visited == 1
    };

    // If just visited now, add all possible states
    // There should be no children of the current node
    if do_add {
        added_nodes = true;
        for legal_state in legal_states {
            add_new_node(&mut g, Some(node_index), legal_state);
        }
    }

    // All children indexes
    let children_indexes: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();

    // Calculate best node to visit
    let mut max_value = -1 as f64;
    let mut best_node_index = children_indexes[0];
    let mut total_children_visits = 0;
    for i in children_indexes.clone() {
        total_children_visits += g.index(i).num_visited;
    }

    {
        let current_node = g.index(node_index);
        for child_index in children_indexes {
            let value = (current_node.value / current_node.num_visited) as f64 +
                (2 as f64).sqrt() * (total_children_visits as f64).sqrt() / (1 + g.index(child_index).num_visited) as f64;
            if value > max_value {
                max_value = value;
                best_node_index = child_index;
            }
        }
    }
    let mcts_data: MCTSData;

    // Check if best node was just added
    if added_nodes {
        if children_before_addition.contains(&best_node_index) {
            mcts_data = recurse_mcts(g, best_node_index);
        } else {
            // don't recurse
        }
    } else {
        mcts_data = recurse_mcts(g, best_node_index);
    }
    mcts_data
}
