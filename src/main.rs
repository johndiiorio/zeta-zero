extern crate rand;
extern crate chess;
extern crate time;

use rand::Rng;
use time::PreciseTime;
use chess::MoveGen;
use chess::Board;
use chess::BoardStatus;
use chess::Piece;

fn main() {
    let start = PreciseTime::now();
    let high = 1000;
    let mut sum = 0;
    for _ in 0..high {
        sum += num_moves_per_random_game();
    }
    let end = PreciseTime::now();
    println!("After {} iterations, average ply per game is {}", high, sum / high);
    println!("Total seconds: {}", start.to(end));
}

fn num_moves_per_random_game() -> u32 {
    let mut board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()).unwrap();
    let mut count = 0;

    while !is_game_over(board) {
        count += 1;
        let mut moves_iter = MoveGen::new(board, true);
        let random_move_location = rand::thread_rng().gen_range(0, moves_iter.len());
        let random_move = moves_iter.nth(random_move_location).unwrap();
        
        board = board.make_move(random_move);
    }
    count
}

fn is_game_over(board: Board) -> bool {
    board.status() != BoardStatus::Ongoing || game_drawn(board)
} 

fn game_drawn(board: Board) -> bool {
    let pawn_moves = board.pieces(Piece::Pawn).popcnt();
    let knight_moves = board.pieces(Piece::Knight).popcnt();
    let bishop_moves = board.pieces(Piece::Bishop).popcnt();
    let rook_moves = board.pieces(Piece::Rook).popcnt();
    let queen_moves = board.pieces(Piece::Queen).popcnt();
    pawn_moves + knight_moves + bishop_moves + rook_moves + queen_moves == 0
}
