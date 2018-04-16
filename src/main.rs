extern crate time;
extern crate chess;
extern crate petgraph;

mod chess_utils;
mod mcts;

use time::PreciseTime;

fn main() {
    let start = PreciseTime::now();
    let end = PreciseTime::now();
    println!("Total seconds: {}", start.to(end));
}
