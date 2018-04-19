pub trait State: Sized {
    fn get_legal_states(&self) -> Vec<Self>;
    fn is_terminal(&self) -> Terminal;
}

pub struct Terminal {
    pub is_terminal: bool,
    pub value: Option<i32>
}