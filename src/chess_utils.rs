use chess::{MoveGen, Board, BoardStatus, Piece, Color};
use traits::{State, Terminal};

impl State for Board {
    fn get_legal_states(&self) -> Vec<Self> {
        let mut states: Vec<Board> = Vec::new();
        let move_gen_iter: MoveGen = MoveGen::new(*self, true);
        for chess_move in move_gen_iter {
            let board = self.clone();
            states.push(board.make_move(chess_move));
        }
        states
    }

    fn get_root_state() -> Board {
//        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()).unwrap()
        Board::from_fen("7k/5K2/8/6R1/8/8/8/8 w - - 0 1".to_string()).unwrap()
    }

    fn is_terminal(&self) -> Terminal {
        let status = self.status();
        let is_drawn = status == BoardStatus::Stalemate || game_drawn(*self);
        if status == BoardStatus::Ongoing && !is_drawn {
            return Terminal {
                is_terminal: false,
                value: None
            }
        }

        // TODO check that this isn't Color::White
        let white_to_move = self.side_to_move() == Color::Black;
        let value;
        if is_drawn {
            value = 0;
        } else if white_to_move {
            value = 1;
        } else {
            value = -1;
        }

        Terminal {
            is_terminal: true,
            value: Some(value)
        }
    }
}


fn game_drawn(board: Board) -> bool {
    let pawn_moves = board.pieces(Piece::Pawn).popcnt();
    let knight_moves = board.pieces(Piece::Knight).popcnt();
    let bishop_moves = board.pieces(Piece::Bishop).popcnt();
    let rook_moves = board.pieces(Piece::Rook).popcnt();
    let queen_moves = board.pieces(Piece::Queen).popcnt();
    pawn_moves + knight_moves + bishop_moves + rook_moves + queen_moves == 0
}
