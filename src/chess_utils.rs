use rand::{thread_rng, Rng};
use chess::{MoveGen, Board, BoardStatus, Piece};

pub struct State {
    state: Board
}

pub fn get_legal_actions(state: State) -> Vec<State> {
    return MoveGen::new(state.state, true).collect();
}

pub fn is_terminal(state: State) -> bool {
    state.state.status() != BoardStatus::Ongoing || game_drawn(state.state)
}

fn game_drawn(board: Board) -> bool {
    let pawn_moves = board.pieces(Piece::Pawn).popcnt();
    let knight_moves = board.pieces(Piece::Knight).popcnt();
    let bishop_moves = board.pieces(Piece::Bishop).popcnt();
    let rook_moves = board.pieces(Piece::Rook).popcnt();
    let queen_moves = board.pieces(Piece::Queen).popcnt();
    pawn_moves + knight_moves + bishop_moves + rook_moves + queen_moves == 0
}
