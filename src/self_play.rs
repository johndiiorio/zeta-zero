use chess_utils;
use mcts;

pub fn create_games(num_games: u32, mcts_per_move: u32) {
    for _ in 0..num_games {
        create_game(mcts_per_move)
    }
}

fn create_game(mcts_per_move: u32) {
    let _mcts_data = mcts::run_mcts(
        chess_utils::get_root_state(),
        mcts_per_move
    );
}