use crate::inner::token::{Token, Tokenizer};
use core::fmt;
use owo_colors::OwoColorize;
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
pub(crate) enum Grammar {
    Num(u64),                        // num*       -> num
    Sum(Box<Grammar>, Box<Grammar>), // num , num  -> num
    Sub(Box<Grammar>, Box<Grammar>), // num , num  -> num
    Mul(Box<Grammar>, Box<Grammar>), // num , num  -> num
    Neg(Box<Grammar>),               // num        -> num
    Adv(u64, Box<Grammar>),          // num*, arr  -> arr
    Dis(u64, Box<Grammar>),          // num*, arr  -> arr
    Cho(u64, Box<Grammar>),          // num*, arr  -> arr
    Pic(u64, Box<Grammar>),          // num*, arr  -> arr
    Die(u64, u64),                   // num*, num* -> arr
    Arr(Vec<Grammar>),               // arr        -> arr
}

#[derive(Debug)]
pub enum ExecResult {
    Value(i64),
    Array(Vec<i64>),
}

impl ExecResult {
    pub fn into_value(self) -> i64 {
        match self {
            ExecResult::Value(n) => n,
            ExecResult::Array(vec) => vec.into_iter().fold(0, |a, b| a + b),
        }
    }
    pub fn into_array(self) -> Result<Vec<i64>, String> {
        match self {
            ExecResult::Value(n) => Err(format!("Expected array, but got Value({}) instead", n)),
            ExecResult::Array(vec) => Ok(vec),
        }
    }
}

impl Grammar {
    pub(crate) fn parse(input: &str) -> Result<Grammar, GrammarError> {
        let mut tokenizer = Tokenizer::new(input);
        let result = expression(&mut tokenizer)?;

        if tokenizer.expended_count() < input.len() {
            return Err(GrammarError {
                error_type: GrammarErrorType::RedundantTokensAfterExpression,
                error_index: tokenizer.expended_count(),
                error_length: input.len() - tokenizer.expended_count(),
                input_string: input.into(),
            });
        }
        Ok(result)
    }

    pub(crate) fn exec(&self, rng: &mut impl rand::Rng) -> Result<ExecResult, String> {
        match &self {
            Grammar::Num(n) => Ok(ExecResult::Value(*n as i64)),
            Grammar::Sum(lhs, rhs) => Ok(ExecResult::Value(
                lhs.exec(rng)?.into_value() + rhs.exec(rng)?.into_value(),
            )),
            Grammar::Sub(lhs, rhs) => Ok(ExecResult::Value(
                lhs.exec(rng)?.into_value() - rhs.exec(rng)?.into_value(),
            )),
            Grammar::Mul(lhs, rhs) => Ok(ExecResult::Value(
                lhs.exec(rng)?.into_value() * rhs.exec(rng)?.into_value(),
            )),
            Grammar::Neg(lhs) => Ok(ExecResult::Value(-lhs.exec(rng)?.into_value())),
            Grammar::Adv(lhs, rhs) => {
                let mut v = rhs.exec(rng)?.into_array()?;
                let new_size = *lhs as usize;
                if v.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                v.sort_by(|a, b| b.cmp(a));
                while v.len() != new_size {
                    v.pop();
                }
                Ok(ExecResult::Array(v))
            }
            Grammar::Dis(lhs, rhs) => {
                let mut v = rhs.exec(rng)?.into_array()?;
                let new_size = *lhs as usize;
                if v.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                v.sort_by(|a, b| a.cmp(b));
                while v.len() != new_size {
                    v.pop();
                }
                Ok(ExecResult::Array(v))
            }
            Grammar::Pic(lhs, rhs) => {
                let mut v = rhs.exec(rng)?.into_array()?;
                let new_size = *lhs as usize;
                if v.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                v.shuffle(rng);
                while v.len() != new_size {
                    v.pop();
                }
                Ok(ExecResult::Array(v))
            }
            Grammar::Cho(lhs, rhs) => {
                let v = rhs.exec(rng)?.into_array()?;
                let mut r: Vec<i64> = vec![];
                let new_size = *lhs as usize;
                while r.len() != new_size {
                    let i = rng.gen_range(0..v.len());
                    r.push(v[i]);
                }
                Ok(ExecResult::Array(r))
            }
            Grammar::Die(lhs, rhs) => {
                let new_size = *lhs as usize;
                let mut r: Vec<i64> = vec![];
                while r.len() != new_size {
                    let x = rng.gen_range(1..*rhs);
                    r.push(x as i64);
                }
                Ok(ExecResult::Array(r))
            }
            Grammar::Arr(vec) => Ok(ExecResult::Array(
                vec.iter()
                    .map(|grammar| grammar.exec(rng).map(|res| res.into_value()))
                    .collect::<Result<Vec<i64>, String>>()?,
            )),
        }
    }

    fn array_length(&self) -> Option<usize> {
        match &self {
            Grammar::Adv(len, _) => Some(*len as usize),
            Grammar::Dis(len, _) => Some(*len as usize),
            Grammar::Cho(len, _) => Some(*len as usize),
            Grammar::Pic(len, _) => Some(*len as usize),
            Grammar::Die(len, _) => Some(*len as usize),
            Grammar::Arr(vec) => Some(vec.len()),
            _ => None,
        }
    }

    fn num(num: u64) -> Self {
        Self::Num(num)
    }

    fn sum(lhs: Self, rhs: Self) -> Self {
        Self::Sum(Box::new(lhs), Box::new(rhs))
    }

    fn sub(lhs: Self, rhs: Self) -> Self {
        Self::Sub(Box::new(lhs), Box::new(rhs))
    }

    fn mul(lhs: Self, rhs: Self) -> Self {
        Self::Mul(Box::new(lhs), Box::new(rhs))
    }

    fn adv(lhs: u64, rhs: Self) -> Self {
        Self::Adv(lhs, Box::new(rhs))
    }

    fn dis(lhs: u64, rhs: Self) -> Self {
        Self::Dis(lhs, Box::new(rhs))
    }

    fn cho(lhs: u64, rhs: Self) -> Self {
        Self::Cho(lhs, Box::new(rhs))
    }

    fn pic(lhs: u64, rhs: Self) -> Self {
        Self::Pic(lhs, Box::new(rhs))
    }

    fn die(lhs: u64, rhs: u64) -> Self {
        Self::Die(lhs, rhs)
    }

    fn neg(child: Self) -> Self {
        Self::Neg(Box::new(child))
    }

    fn arr(children: Vec<Self>) -> Self {
        Self::Arr(children)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum GrammarErrorType {
    UnknownToken(String),
    UnexpectedEnd {
        expected: &'static str,
    },
    UnexpectedToken {
        token: Token,
        expected: &'static str,
    },
    IncompatibleArrayLength {
        token: Token,
        length: usize,
        expected: usize,
    },
    RedundantTokensAfterExpression,
}

#[derive(Debug, Clone)]
pub(crate) struct GrammarError {
    error_index: usize,
    error_length: usize,
    input_string: String,
    error_type: GrammarErrorType,
}

impl fmt::Display for GrammarErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            GrammarErrorType::UnknownToken(chars) => write!(f, "Unknown token \"{}\"", chars),
            GrammarErrorType::UnexpectedEnd { expected } => {
                write!(f, "Unexpected end of expression; expected {}", expected)
            }
            GrammarErrorType::UnexpectedToken { token, expected } => {
                write!(f, "Unexpected token {:?}; expected {}", token, expected)
            }
            GrammarErrorType::IncompatibleArrayLength {
                token,
                length,
                expected,
            } => {
                write!(
                    f,
                    "Incompatible Array length; {:?} expects length of {} but received {}",
                    token, expected, length
                )
            }
            GrammarErrorType::RedundantTokensAfterExpression => {
                write!(
                    f,
                    "Redundant tokens after expression; perhaps missing a token?"
                )
            }
        }
    }
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n{}{} {}",
            self.input_string,
            (0..self.error_index).map(|_| ' ').collect::<String>(),
            (0..self.error_length)
                .map(|_| '^')
                .collect::<String>()
                .red()
                .bold(),
            self.error_type.to_string().red().bold(),
        )
    }
}

/**
 * Expression
 *    = Term (("+" / "-") Expression)+
 */
fn expression(tokenizer: &mut Tokenizer) -> Result<Grammar, GrammarError> {
    let left = term(tokenizer)?;
    while let Some(Token::Plus | Token::Minus) = tokenizer.peek() {
        if let Some(Token::Plus) = tokenizer.peek() {
            tokenizer.next();
            let right = expression(tokenizer)?;
            return Ok(Grammar::sum(left, right));
        } else {
            tokenizer.next();
            let right = expression(tokenizer)?;
            return Ok(Grammar::sub(left, right));
        }
    }
    Ok(left)
}

/**
 * Term
 *    = Negative ("*" Term)+
 */
fn term(tokenizer: &mut Tokenizer) -> Result<Grammar, GrammarError> {
    let left = negative(tokenizer)?;
    if let Some(Token::Multiply) = tokenizer.peek() {
        tokenizer.next();
        let right = negative(tokenizer)?;
        return Ok(Grammar::mul(left, right));
    }
    Ok(left)
}

/**
 * Negative
 *    = ("-")+ Any
 */
fn negative(tokenizer: &mut Tokenizer) -> Result<Grammar, GrammarError> {
    if let Some(Token::Minus) = tokenizer.peek() {
        tokenizer.next();
        let right = any(tokenizer)?;
        return Ok(Grammar::neg(right));
    }
    any(tokenizer)
}

/**
 * Any
 *    = Number / Primary
 */
fn any(tokenizer: &mut Tokenizer) -> Result<Grammar, GrammarError> {
    if let Some(Token::Number(n)) = tokenizer.peek() {
        tokenizer.next();
        return Ok(Grammar::num(n));
    }
    primary(tokenizer, "Expression")
}

/**
 * Primary
 *    = Dice / Modifier / Array
 */
fn primary(tokenizer: &mut Tokenizer, expected: &'static str) -> Result<Grammar, GrammarError> {
    match tokenizer.peek() {
        Some(Token::LeftParenthesis) => {
            tokenizer.next();
            let mut arr: Vec<Grammar> = Vec::new();
            loop {
                arr.push(expression(tokenizer)?);
                match tokenizer.peek() {
                    Some(Token::RightParenthesis) => {
                        tokenizer.next();
                        break;
                    }
                    Some(Token::Comma) => tokenizer.next(),
                    Some(token) => {
                        return Err(GrammarError {
                            error_type: GrammarErrorType::UnexpectedToken {
                                token,
                                expected: "',' or ')'",
                            },
                            error_index: tokenizer.expended_count(),
                            error_length: tokenizer.peek_token_count(),
                            input_string: tokenizer.input_str().into(),
                        });
                    }
                    None => {
                        return Err(GrammarError {
                            error_type: GrammarErrorType::UnexpectedEnd {
                                expected: "',' or ')'",
                            },
                            error_index: tokenizer.expended_count(),
                            error_length: 1,
                            input_string: tokenizer.input_str().into(),
                        });
                    }
                };
            }
            Ok(Grammar::arr(arr))
        }
        Some(Token::Dice(lhs, rhs)) => {
            tokenizer.next();
            Ok(Grammar::die(lhs, rhs))
        }
        Some(Token::Advantage(lhs)) => {
            tokenizer.next();
            let expended_count_before = tokenizer.expended_count();
            let rhs = primary(tokenizer, "Array")?;
            let len = rhs.array_length().unwrap();
            if len < lhs as usize {
                return Err(GrammarError {
                    error_type: GrammarErrorType::IncompatibleArrayLength {
                        token: Token::Advantage(lhs),
                        length: len,
                        expected: lhs as usize,
                    },
                    error_index: expended_count_before,
                    error_length: tokenizer.expended_count() - expended_count_before,
                    input_string: tokenizer.input_str().into(),
                });
            }
            Ok(Grammar::adv(lhs, rhs))
        }
        Some(Token::Disadvantage(lhs)) => {
            tokenizer.next();
            let expended_count_before = tokenizer.expended_count();
            let rhs = primary(tokenizer, "Array")?;
            let len = rhs.array_length().unwrap();
            if len < lhs as usize {
                return Err(GrammarError {
                    error_type: GrammarErrorType::IncompatibleArrayLength {
                        token: Token::Disadvantage(lhs),
                        length: len,
                        expected: lhs as usize,
                    },
                    error_index: expended_count_before,
                    error_length: tokenizer.expended_count() - expended_count_before,
                    input_string: tokenizer.input_str().into(),
                });
            }
            Ok(Grammar::dis(lhs, rhs))
        }
        Some(Token::Pick(lhs)) => {
            tokenizer.next();
            let expended_count_before = tokenizer.expended_count();
            let rhs = primary(tokenizer, "Array")?;
            let len = rhs.array_length().unwrap();
            if len < lhs as usize {
                return Err(GrammarError {
                    error_type: GrammarErrorType::IncompatibleArrayLength {
                        token: Token::Pick(lhs),
                        length: len,
                        expected: lhs as usize,
                    },
                    error_index: expended_count_before,
                    error_length: tokenizer.expended_count() - expended_count_before,
                    input_string: tokenizer.input_str().into(),
                });
            }
            Ok(Grammar::pic(lhs, rhs))
        }
        Some(Token::Choose(lhs)) => {
            tokenizer.next();
            let rhs = primary(tokenizer, "Array")?;
            Ok(Grammar::cho(lhs, rhs))
        }
        Some(Token::Unknown(c)) => Err(GrammarError {
            error_type: GrammarErrorType::UnknownToken(c),
            error_index: tokenizer.expended_count(),
            error_length: tokenizer.peek_token_count(),
            input_string: tokenizer.input_str().into(),
        }),
        Some(token) => Err(GrammarError {
            error_type: GrammarErrorType::UnexpectedToken { token, expected },
            error_index: tokenizer.expended_count(),
            error_length: tokenizer.peek_token_count(),
            input_string: tokenizer.input_str().into(),
        }),
        None => Err(GrammarError {
            error_type: GrammarErrorType::UnexpectedEnd { expected },
            error_index: tokenizer.expended_count(),
            error_length: 1,
            input_string: tokenizer.input_str().into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    // use rand::thread_rng;
    use std::assert_matches::assert_matches;

    use super::*;

    #[test]
    fn test_parse_empty() {
        let x = "";
        let result = Grammar::parse(x);
        assert_matches!(
            result,
            Err(GrammarError {
                error_type: GrammarErrorType::UnexpectedEnd {
                    expected: "Expression"
                },
                error_index: 0,
                error_length: 1,
                input_string: _,
            })
        );
    }

    #[test]
    fn test_parse_unknown_token() {
        let x = "abc";
        let result = Grammar::parse(x);
        assert_matches!(
            result,
            Err(GrammarError {
                error_type: GrammarErrorType::UnknownToken(_),
                error_index: 0,
                error_length: 2,
                input_string: _,
            })
        );
        if let Err(GrammarError {
            error_type: GrammarErrorType::UnknownToken(s),
            error_index: _,
            error_length: _,
            input_string: _,
        }) = result
        {
            assert_eq!(s.as_str(), "ab");
        }
    }

    #[test]
    fn test_parse_unexpected_token() {
        let x = "3!4";
        let result = Grammar::parse(x);
        assert_matches!(
            result,
            Err(GrammarError {
                error_type: GrammarErrorType::UnexpectedToken {
                    token: Token::Number(4),
                    expected: "Array"
                },
                error_index: 2,
                error_length: 1,
                input_string: _,
            })
        );
    }

    #[test]
    fn test_parse_mismatched_array_length() {
        let x = "3$(1,2)";
        let result = Grammar::parse(x);
        assert_matches!(
            result,
            Err(GrammarError {
                error_type: GrammarErrorType::IncompatibleArrayLength {
                    token: Token::Advantage(3),
                    length: 2,
                    expected: 3,
                },
                error_index: 2,
                error_length: 5,
                input_string: _,
            })
        );
    }
}
