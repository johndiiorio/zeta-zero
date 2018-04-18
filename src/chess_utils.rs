use chess::{MoveGen, Board, BoardStatus, Piece};

pub struct NodeState {
    state: Board
}

impl State for NodeState {
    fn get_legal_states(&self) -> Vec<Self> {
        let mut states: Vec<NodeState> = Vec::new();
        let move_gen_iter: MoveGen = MoveGen::new(self.state, true);
        for chess_move in move_gen_iter {
            let board = self.state.clone();
            states.push(NodeState {
                state: board.make_move(chess_move)
            });
        }
        states
    }

    fn is_terminal(&self) -> bool {
        self.state.status() != BoardStatus::Ongoing || game_drawn(self.state)
    }
}

pub trait State: Sized {
    fn get_legal_states(&self) -> Vec<Self>;
    fn is_terminal(&self) -> bool;
}

fn game_drawn(board: Board) -> bool {
    let pawn_moves = board.pieces(Piece::Pawn).popcnt();
    let knight_moves = board.pieces(Piece::Knight).popcnt();
    let bishop_moves = board.pieces(Piece::Bishop).popcnt();
    let rook_moves = board.pieces(Piece::Rook).popcnt();
    let queen_moves = board.pieces(Piece::Queen).popcnt();
    pawn_moves + knight_moves + bishop_moves + rook_moves + queen_moves == 0
}
