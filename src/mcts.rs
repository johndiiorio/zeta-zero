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
    policy: Vec<u32>
}

struct NeuralNetworkData {
    value: i32,
    policy: Vec<u32>
}

#[derive(Clone)]
struct MCTSData {
    value: i32,
    policy: Vec<u32>,
}

pub fn run_mcts<T: State>(state: T, num_iterations: u32) {
    let (mut g, root_index) = create_mcts_graph(state);
    for i in 0..num_iterations {
        let mcts_data = recurse_mcts(&mut g, root_index);
        if i % 100 == 0 {
            println!("Value: {}, num_states: {}, policy: {:?}", mcts_data.value, g.node_count(), mcts_data.policy);
        }
    }
}

fn create_mcts_graph<T: State>(state: T) -> (Graph<Node<T>, u32, Directed>, NodeIndex) {
    let mut g = Graph::<Node<T>, u32, Directed>::new();
    let index = add_new_node(&mut g, None, state);
    (g, index)
}

fn add_new_node<T: State>(g: &mut Graph<Node<T>, u32, Directed>, parent: Option<NodeIndex>, state: T) -> NodeIndex {
    let index = g.add_node(Node {state, num_visited: 0, value: 0, policy: Vec::new() });
    match parent {
        None => {},
        Some(e) => {
            g.add_edge(e, index, 0);
        }
    }
    index
}

fn recurse_mcts<T: State>(mut g: &mut Graph<Node<T>, u32, Directed>, node_index: NodeIndex) -> MCTSData {
    // Nodes in tree before additions
    let children_before_addition: Vec<NodeIndex> = g.neighbors_directed(node_index, Direction::Outgoing).collect();
    let mut should_add_nodes = false;
    {
        // Return data if current node is terminal
        let current_node = g.index_mut(node_index);
        if current_node.num_visited == 0 {
            should_add_nodes = true;
        }
        let terminal_data = current_node.state.is_terminal();
        if terminal_data.is_terminal {
            let terminal_value = terminal_data.value.unwrap();
            return MCTSData {
                value: terminal_value,
                policy: vec![normalize(terminal_value, -1, 1)]
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

    // Neural network prediction
    let nn_data: NeuralNetworkData;

    // Calculate best node to visit
    let mut max_value = -1 as f32;
    let mut best_node_index = children_indexes[0];
    let mut total_children_visits = 0;
    for i in children_indexes.clone() {
        total_children_visits += g.index(i).num_visited;
    }
    {
        let current_node = g.index(node_index);
        nn_data = predict(&current_node.state, children_indexes.len());
        for (i, child_index) in children_indexes.iter().enumerate() {
            let selection_node_value = calculate_selection_node_value(
                current_node,
                g.index(*child_index),
                total_children_visits,
                nn_data.policy[i]
            );
            if selection_node_value > max_value {
                max_value = selection_node_value;
                best_node_index = *child_index;
            }
        }
    }

    // Data returned from children nodes in the tree
    let mcts_data;

    // Check if best node was just added
    if should_add_nodes {
        if children_before_addition.contains(&best_node_index) {
            mcts_data = recurse_mcts(&mut g, best_node_index);
        } else {
            // Leaf node, don't recurse, backpropagate values
            return MCTSData {
                value: nn_data.value,
                policy: nn_data.policy
            };
        }
    } else {
        mcts_data = recurse_mcts(&mut g, best_node_index);
    }


    // Modify the tree on the way back up the stack
    let cloned_mcts_data = mcts_data.clone();
    let current_node = g.index_mut(node_index);
    current_node.value += cloned_mcts_data.value;
    current_node.policy = cloned_mcts_data.policy;

    mcts_data
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
//fn predict<T: State>(state: &T) -> NeuralNetworkData {
//    NeuralNetworkData {
//        value: 0,
//        policy: Vec::new()
//    }
//}
// TODO remove dummy function for testing
fn predict<T: State>(_state: &T, num_elements: usize) -> NeuralNetworkData {
    let mut policy = Vec::new();
    for _ in 0..num_elements {
        policy.push(0);
    }
    NeuralNetworkData {
        value: 0,
        policy
    }
}

