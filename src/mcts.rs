use petgraph::{Direction, Directed};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use std::ops::Index;
use std::ops::IndexMut;
use traits::State;

pub struct MCTSData<T: State> {
    pub value: i32,
    pub policy: Vec<u32>,
    pub best_state: Option<BestState<T>>
}

pub struct BestState<T: State> {
    pub state: T,
    pub node_index: NodeIndex
}

pub struct Node<T: State + Clone> {
    state: Box<T>,
    num_visited: i32,
    value: i32,
    policy: Vec<u32>
}

struct NeuralNetworkData {
    value: i32,
    policy: Vec<u32>
}

#[derive(Clone)]
struct RecursiveMCTSData {
    value: i32,
    policy: Vec<u32>,
    terminal: bool
}

// Runs MCTS a given number of times from a root state
// Returns the best state from a position
pub fn run_mcts<T: State>(mut g: &mut StableGraph<Node<T>, u32, Directed>, root_index: NodeIndex, num_iterations: u32) -> MCTSData<T> {
    let mut option_recursive_data: Option<RecursiveMCTSData> = None;
    for _ in 0..num_iterations {
        let recursive_mcts_data = recurse_mcts(&mut g, root_index);
        let reached_terminal_position = recursive_mcts_data.terminal;
        option_recursive_data = Some(recursive_mcts_data);
        if reached_terminal_position {
            break
        }
    }
    let unwrapped_mcts_data = option_recursive_data.unwrap();
    let most_visited = most_visited_state(g, root_index);
    MCTSData {
        value: unwrapped_mcts_data.value,
        policy: unwrapped_mcts_data.policy,
        best_state: most_visited
    }
}

pub fn create_mcts_graph<T: State>(state: T) -> (StableGraph<Node<T>, u32, Directed>, NodeIndex) {
    let mut g = StableGraph::<Node<T>, u32, Directed>::new();
    let index = add_new_node(&mut g, None, state);
    (g, index)
}

fn add_new_node<T: State>(g: &mut StableGraph<Node<T>, u32, Directed>, parent: Option<NodeIndex>, state: T) -> NodeIndex {
    let index = g.add_node(Node {
        state: Box::new(state),
        num_visited: 0,
        value: 0,
        policy: Vec::new()
    });
    match parent {
        None => {},
        Some(e) => {
            g.add_edge(e, index, 0);
        }
    }
    index
}

fn recurse_mcts<T: State>(mut g: &mut StableGraph<Node<T>, u32, Directed>, node_index: NodeIndex) -> RecursiveMCTSData {
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
            return RecursiveMCTSData {
                value: terminal_value,
                policy: vec![normalize(terminal_value, -1, 1)],
                terminal: true
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
        nn_data = predict(&*current_node.state, children_indexes.len());
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
            return RecursiveMCTSData {
                value: nn_data.value,
                policy: nn_data.policy,
                terminal: false
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

fn most_visited_state<T: State>(g: &StableGraph<Node<T>, u32, Directed>, node_index: NodeIndex) -> Option<BestState<T>> {
    let mut most_visited_count = -1;
    let mut most_visited_state = None;
    let mut most_visited_node_index = None;
    for curr_node_index in g.neighbors_directed(node_index, Direction::Outgoing) {
        let curr_node = g.index(curr_node_index);
        if curr_node.num_visited > most_visited_count {
            most_visited_count = curr_node.num_visited;
            most_visited_state = Some(curr_node.state.clone());
            most_visited_node_index = Some(curr_node_index);
        }
    }
    if most_visited_state.is_none() || most_visited_node_index.is_none() {
        return None
    }
    Some(BestState {
        state: *most_visited_state.unwrap(),
        node_index: most_visited_node_index.unwrap()
    })
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

pub fn remove_subtree_keep_index<T: State>(g: &mut StableGraph<Node<T>, u32, Directed>, root_index: NodeIndex, keep_index: NodeIndex) {
    let children_indexes: Vec<NodeIndex> = g.neighbors_directed(root_index, Direction::Outgoing).collect();
    for child_index in children_indexes {
        if child_index != keep_index {
            remove_subtree(g, child_index);
        }
    }
}

pub fn remove_subtree<T: State>(g: &mut StableGraph<Node<T>, u32, Directed>, root_index: NodeIndex) {
    let children_indexes: Vec<NodeIndex> = g.neighbors_directed(root_index, Direction::Outgoing).collect();
    if children_indexes.len() == 0 {
        g.remove_node(root_index);
    } else {
        for child_index in children_indexes {
            remove_subtree(g, child_index);
        }
    }
}

