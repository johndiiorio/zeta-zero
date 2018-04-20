use traits::State;
use mcts;

pub fn create_games<T: State>(num_games: u32, mcts_per_move: u32) {
    for _ in 0..num_games {
        create_game::<T>(mcts_per_move)
    }
}

fn create_game<T: State>(mcts_per_move: u32) {
    let _mcts_data = mcts::run_mcts(
        T::get_root_state(),
        mcts_per_move
    );
}