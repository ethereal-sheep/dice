#![feature(assert_matches)]
#![feature(iter_intersperse)]

mod inner;

use core::fmt;
use std::str::FromStr;

use inner::grammar::{GrammarTestOptions, TestDetails};

use crate::inner::grammar::{ExecDetails, Grammar, GrammarError, GrammarExecOptions};

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

impl ParseDiceError {
    pub fn formatted_error_string(&self) -> String {
        self.grammar_error.formatted_error_string()
    }
}

#[derive(Debug, Clone)]
pub struct RollOptions {
    pub is_debug: bool,
}

impl From<RollOptions> for GrammarExecOptions {
    fn from(options: RollOptions) -> Self {
        Self {
            is_debug: options.is_debug,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestOptions {
    pub is_debug: bool,
    pub test_size: usize,
}

impl From<TestOptions> for GrammarTestOptions {
    fn from(options: TestOptions) -> Self {
        Self {
            is_debug: options.is_debug,
            test_size: options.test_size,
        }
    }
}

#[derive(Clone)]
pub struct Dice {
    grammar: Grammar,
}

impl Dice {
    fn from_str(input: &str) -> Result<Self, ParseDiceError> {
        Ok(Self {
            grammar: Grammar::parse(input)?,
        })
    }

    pub fn min(&self) -> i64 {
        self.grammar.min()
    }

    pub fn max(&self) -> i64 {
        self.grammar.max()
    }

    pub fn roll(
        &self,
        rng: &mut impl rand::Rng,
        options: RollOptions,
    ) -> Result<ExecDetails, String> {
        self.grammar.exec(rng, options.into())
    }

    pub fn test(
        &self,
        rng: &mut (impl rand::Rng + Clone),
        options: TestOptions,
    ) -> Result<TestDetails, String> {
        self.grammar.test(rng, options.into())
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
        write!(f, "{}", self.grammar)
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_dice_from_str() {
        let x = "-d20+z3d4";
        let result = x.parse::<Dice>();
        match result {
            Ok(dice) => {
                println!("{}", dice);
                println!(
                    "{:?}",
                    dice.roll(&mut thread_rng(), RollOptions { is_debug: true })
                );
            }
            Err(err) => println!("{}", err),
        }
    }
}
