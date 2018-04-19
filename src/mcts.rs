use petgraph::{Graph, Direction};
use petgraph::graph::NodeIndex;
use petgraph::Directed;
use std::ops::Index;
use std::ops::IndexMut;
use traits::{State};

struct Node<T> {
    num_visited: i32,
    value: i32,
    state: T,
    policies: Vec<u32>
}

struct NeuralNetworkData {
    value: i32,
    policies: Vec<u32>
}

struct MCTSData {
    value: i32,
    policy: u32
}

pub fn run_mcts<T: State>(state: T) {
    let (g, root_index) = create_mcts_graph(state);
    let mcts_data = recurse_mcts(g, root_index);
}

fn create_mcts_graph<T: State>(state: T) -> (Graph<Node<T>, u32, Directed>, NodeIndex) {
    let mut g = Graph::<Node<T>, u32, Directed>::new();
    let index = add_new_node(&mut g, None, state);
    (g, index)
}

fn add_new_node<T: State>(g: &mut Graph<Node<T>, u32, Directed>, parent: Option<NodeIndex>, state: T) -> NodeIndex {
    let index = g.add_node(Node {state, num_visited: 0, value: 0, policies: Vec::new() });
    match parent {
        None => {},
        Some(e) => {
            g.add_edge(e, index, 0);
        }
    }
    index
}

fn recurse_mcts<T: State>(mut g: Graph<Node<T>, u32, Directed>, node_index: NodeIndex) -> MCTSData {
    // Nodes in tree before additions
    let children_before_addition: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();
    let mut should_add_nodes = false;
    {
        // Return data if current node is terminal
        let current_node = g.index_mut(node_index);
        if current_node.num_visited == 1 {
            should_add_nodes = true;
        }
        let terminal_data = current_node.state.is_terminal();
        let terminal_value = terminal_data.value.unwrap();
        if terminal_data.is_terminal {
            return MCTSData {
                value: terminal_value,
                policy: normalize(terminal_value, -1, 1)
            }
        }

        // Always increment the number of times that the current node was visited
        current_node.num_visited += 1;
    }

    // If just visited now, add all possible states (should be no children of the current node)
    let legal_states;
    if should_add_nodes {
        {
            let current_node = g.index(node_index);
            legal_states = current_node.state.get_legal_states();
        }

        for legal_state in legal_states {
            add_new_node(&mut g, Some(node_index), legal_state);
        }
    }

    // All children indexes
    let children_indexes: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();

    // Calculate best node to visit
    let mut max_value = -1 as f32;
    let mut best_node_index = children_indexes[0];
    let mut total_children_visits = 0;
    for i in children_indexes.clone() {
        total_children_visits += g.index(i).num_visited;
    }
    {
        let current_node = g.index(node_index);
        let nn_data = predict(&current_node.state);
        for (i, child_index) in children_indexes.iter().enumerate() {
            let selection_node_value = calculate_selection_node_value(
                current_node,
                g.index(*child_index),
                total_children_visits,
                nn_data.policies[i]
            );
            if selection_node_value > max_value {
                max_value = selection_node_value;
                best_node_index = *child_index;
            }
        }
    }

    // Check if best node was just added
    if should_add_nodes {
        if children_before_addition.contains(&best_node_index) {
            return recurse_mcts(g, best_node_index);
        } else {
            // Leaf node, don't recurse, backpropagate values
//            let current_node = g.index(node_index);
            return recurse_mcts(g, best_node_index);

        }
    } else {
        return recurse_mcts(g, best_node_index);
    }
}

fn calculate_selection_node_value<T: State>(current_node: &Node<T>, child_node: &Node<T>, total_children_visits: i32, policy_value: u32) -> f32 {
    let exploitation = (current_node.value / current_node.num_visited) as f32;
    let c = (2 as f32).sqrt();
    let exploration = (total_children_visits as f32).sqrt() / (1 + child_node.num_visited) as f32;
    exploitation + c * (policy_value as f32) * exploration
}

// Normalize value in range [min, max] to [0, 1]
fn normalize(x: i32, min: i32, max: i32) -> u32 {
    ((x - min) / (max - min)) as u32
}

// TODO hook this up with neural net
fn predict<T: State>(state: &T) -> NeuralNetworkData {
    NeuralNetworkData {
        value: 0,
        policies: Vec::new()
    }
}
