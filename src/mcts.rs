use petgraph::Graph;
use petgraph::graph::NodeIndex;

pub struct Node {
    pub num_visited: u32,
    pub value: i32,
    pub fen: String,
}

pub fn create_mcts_graph(fen: String) -> Graph<Node, u32> {
    let mut g = Graph::<Node, u32>::new();
    add_node(&mut g, None, fen);
    g
}

pub fn add_node(g: &mut Graph<Node, u32>, parent: Option<NodeIndex>, fen: String) -> NodeIndex {
    let index = g.add_node(Node {num_visited: 0, value: 0, fen });
    match parent {
        None => (),
        Some(e) => {
            g.add_edge(e, index, 0);
            ()
        }
    }
    index
}
