use std::fmt::Display;

pub trait State: Sized + Clone + Display {
    fn get_legal_states(&self) -> Vec<Self>;
    fn get_root_state() -> Self;
    fn is_terminal(&self) -> Terminal;
}

pub struct Terminal {
    pub is_terminal: bool,
    pub value: Option<i32>
}
