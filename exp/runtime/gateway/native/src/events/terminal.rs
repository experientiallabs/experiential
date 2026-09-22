//! Terminal classifications and lossless incomplete-cause disclosure.

use super::Event;

impl Event {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Completed
                | Event::Incomplete
                | Event::IncompleteToolArguments
                | Event::StoppedAtSequence(_)
                | Event::PausedTurn
                | Event::Failed(_)
        )
    }
}
