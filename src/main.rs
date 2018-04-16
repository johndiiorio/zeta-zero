extern crate time;
extern crate chess;
extern crate rand;
extern crate petgraph;

mod chess_utils;
mod mcts;

use time::PreciseTime;

fn main() {
    let start = PreciseTime::now();
    let high = 10;
    let mut sum = 0;
    for _ in 0..high {
        sum += chess_utils::num_moves_per_random_game();
    }
    let end = PreciseTime::now();
    println!("After {} iterations, average ply per game is {}", high, sum / high);
    println!("Total seconds: {}", start.to(end));

    let (g, root_index) = mcts::create_mcts_graph("hello".to_string());
    println!("{:?}", mcts::find_node_maximizing_bound(&g, root_index));
}
