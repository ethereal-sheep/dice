#![feature(assert_matches)]
#![feature(iter_intersperse)]

mod inner;

use core::fmt;
use std::{
    rc::Rc,
    str::FromStr,
    time::{Duration, Instant},
};

use crate::inner::grammar::{Grammar, GrammarError};

#[derive(Debug, Clone)]
pub enum ExecOutput {
    Value(i64),
    Array(Vec<i64>),
}

type ExecResult = Result<ExecOutput, String>;

#[derive(Debug, Clone)]
pub struct ExecLineDetail {
    pub name: &'static str,
    pub operation: String,
    pub output: String,
}

#[derive(Debug, Clone)]
pub struct ExecDetails {
    pub output: ExecOutput,
    pub details: Vec<ExecLineDetail>,
}

type ExecResultWithDetails = Result<ExecDetails, String>;

#[derive(Debug, Clone)]
pub struct TestLineDetail {
    pub name: &'static str,
    pub operation: String,
    pub time_taken: Duration,
}

#[derive(Debug, Clone)]
pub struct TestDetails {
    pub output: Vec<i64>,
    pub time_taken: Duration,
    pub details: Vec<TestLineDetail>,
}

type TestResultWithDetails = Result<TestDetails, String>;

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

#[derive(Clone)]
pub struct RollOptions {
    pub include_line_details: bool,
}

pub struct OperationTestInfo<'a> {
    code: &'a str,
    name: &'a str,
    operation_index: usize,
    iteration_index: usize,
    test_size: usize,
    start_time: Instant,
}

impl<'a> OperationTestInfo<'a> {
    pub fn operation_code(&self) -> &'a str {
        self.code
    }
    pub fn operation_name(&self) -> &'a str {
        self.name
    }
    pub fn operation_index(&self) -> usize {
        self.operation_index
    }
    pub fn iteration_index(&self) -> usize {
        self.iteration_index
    }
    pub fn start_time(&self) -> Instant {
        self.start_time
    }
    pub fn is_first(&self) -> bool {
        self.iteration_index() == 0
    }
    pub fn is_last(&self) -> bool {
        self.iteration_index() == self.test_size - 1
    }
}

pub struct OverallTestInfo<'a> {
    operation_test_info: OperationTestInfo<'a>,
    operations_count: usize,
    start_time: Instant,
}

impl<'a> OverallTestInfo<'a> {
    pub fn current_test_info(&self) -> &'a OperationTestInfo {
        &self.operation_test_info
    }
    pub fn operations_count(&self) -> usize {
        self.operations_count
    }
    pub fn test_size(&self) -> usize {
        self.operation_test_info.test_size
    }
    pub fn start_time(&self) -> Instant {
        self.start_time
    }
    pub fn total_test_index(&self) -> usize {
        self.current_test_info().operation_index() * self.test_size()
            + self.current_test_info().iteration_index()
    }
    pub fn total_test_count(&self) -> usize {
        self.operations_count() * self.test_size()
    }
    pub fn is_first(&self) -> bool {
        self.total_test_index() == 0
    }
    pub fn is_last(&self) -> bool {
        self.total_test_index() == self.total_test_count() - 1
    }
}

pub type TestIntervalCallback = dyn Fn(&OverallTestInfo);

#[derive(Clone)]
pub struct TestOptions {
    pub is_debug: bool,
    pub test_size: usize,
    pub interval_callback: Option<Rc<TestIntervalCallback>>,
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

    pub fn roll(&self, rng: &mut impl rand::Rng, options: RollOptions) -> ExecResultWithDetails {
        self.grammar.exec(rng, options)
    }

    pub fn test(
        &self,
        rng: &mut (impl rand::Rng + Clone),
        options: TestOptions,
    ) -> TestResultWithDetails {
        self.grammar.test(rng, options)
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
                let result = dice.roll(
                    &mut thread_rng(),
                    RollOptions {
                        include_line_details: true,
                    },
                );
                match result {
                    Ok(details) => assert!(details.details.len() > 0),
                    Err(_) => assert!(false),
                }

                let result = dice.test(
                    &mut thread_rng(),
                    TestOptions {
                        is_debug: true,
                        test_size: 100,
                        interval_callback: None,
                    },
                );
                match result {
                    Ok(details) => assert!(details.details.len() > 0),
                    Err(_) => assert!(false),
                }
            }
            Err(_) => assert!(false),
        }
    }
}
