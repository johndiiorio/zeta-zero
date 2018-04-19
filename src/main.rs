extern crate time;
extern crate chess;
extern crate petgraph;

mod chess_utils;
mod mcts;
mod traits;

use time::PreciseTime;

fn main() {
    let start = PreciseTime::now();
    mcts::run_mcts(chess_utils::get_root_state());
    let end = PreciseTime::now();
    println!("Total seconds: {}", start.to(end));
}
