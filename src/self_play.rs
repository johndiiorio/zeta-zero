use traits::State;
use mcts;

pub fn create_games<T: State>(num_games: u32, mcts_per_move: u32) {
    for _ in 0..num_games {
        create_game::<T>(mcts_per_move)
    }
}

fn create_game<T: State>(mcts_per_move: u32) {
    let mut state = T::get_root_state();
    let mut count = 0;
    loop {
        let mcts_data = mcts::run_mcts(
            state,
            mcts_per_move
        );
        if mcts_data.best_state.is_none() {
            break;
        }
        println!("Moved at count: {}", count);
        count += 1;
        state = mcts_data.best_state.unwrap();
    }
    println!("Finished one game with count: {}", count);
}