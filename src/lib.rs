#![feature(assert_matches)]

mod inner;

use core::fmt;
use std::str::FromStr;

use crate::inner::grammar::{ExecResult, Grammar, GrammarError};

#[derive(Debug, Clone)]
pub struct ParseDiceError {
    grammar_error: GrammarError,
}

impl From<GrammarError> for ParseDiceError {
    fn from(grammar_error: GrammarError) -> Self {
        Self { grammar_error }
    }
}

impl fmt::Display for ParseDiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.grammar_error.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct Dice {
    root: Grammar,
}

impl Dice {
    fn from_str(input: &str) -> Result<Self, ParseDiceError> {
        Ok(Self {
            root: Grammar::parse(input)?,
        })
    }

    pub fn roll(&self, rng: &mut impl rand::Rng) -> Result<ExecResult, String> {
        self.root.exec(rng)
    }
}

impl FromStr for Dice {
    type Err = ParseDiceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Dice::from_str(s)
    }
}

impl fmt::Display for Dice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.root)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_dice_from_str() {
        let x = "-d20+!3d4";
        let result = x.parse::<Dice>();
        match result {
            Ok(dice) => {
                println!("{}", dice);
                println!("{:?}", dice.roll(&mut thread_rng()));
            }
            Err(err) => println!("{}", err),
        }
    }
}
