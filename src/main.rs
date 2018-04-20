extern crate chess;
extern crate petgraph;
extern crate time;
extern crate yaml_rust;

mod chess_utils;
mod mcts;
mod self_play;
mod traits;

use std::fs::File;
use std::io::Read;
use time::PreciseTime;
use yaml_rust::{Yaml, YamlLoader};

fn parse_config(config_path: &str) -> Yaml {
    let mut f = File::open(config_path).expect("Config file not found");
    let mut contents = String::new();
    f.read_to_string(&mut contents).expect("Failed to read config file");
    YamlLoader::load_from_str(&contents).unwrap()[0].clone()
}

fn main() {
    let start = PreciseTime::now();
    let config = parse_config("config.yml");

    self_play::create_games(
        config["play"]["games_per_iteration"].as_i64().unwrap() as u32,
        config["play"]["mcts_per_move"].as_i64().unwrap() as u32,
    );

    let end = PreciseTime::now();
    println!("Total seconds: {}", start.to(end));
}
