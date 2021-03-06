use super::traits::State;
use super::mcts;

pub fn create_games<T: State>(num_games: u32, mcts_per_move: u32) {
    for _ in 0..num_games {
        create_game::<T>(mcts_per_move)
    }
}

fn create_game<T: State>(mcts_per_move: u32) {
    let root_state = T::get_root_state();
    // let legal_states = root_state.get_legal_states();
    let mut game_states = vec![root_state.clone()];
    let (mut g, mut root_index) = mcts::create_mcts_graph(root_state);

    let mut count = 0;
    loop {
        let mcts_data = mcts::run_mcts(&mut g, root_index, mcts_per_move);
        if mcts_data.best_state.is_none() {
            break;
        }
        println!("Moved at count: {} with nodes in graph: {}", count, g.node_count());
        let best_state = mcts_data.best_state.unwrap();
		println!("{}", best_state.state);

        // let mut found = false;
        // for legal_state in &legal_states {
        //     if legal_state == best_state.state {
        //         found = true;
        //     }
        // }
        mcts::remove_subtree_keep_index(&mut g, root_index, best_state.node_index);

        // Update loop values
        game_states.push(best_state.state);
        root_index = best_state.node_index;
        count += 1;
    }
    println!("Finished one game with count: {}", count);
}