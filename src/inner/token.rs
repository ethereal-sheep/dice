use crate::inner::iter::Countable;
use std::str::Chars;

#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    Number(u64),
    Plus,
    Minus,
    Multiply,
    Advantage(u64),
    Disadvantage(u64),
    Dice(u64, u64),
    Range(i64, i64),
    Pick(u64),
    Choose(u64),
    LeftParenthesis,
    RightParenthesis,
    Comma,
    Unknown(String),
}

type MatchTokenFn = fn(&mut Countable<Chars>) -> Option<Token>;

fn consume_unsigned(chars: &mut Countable<Chars>) -> Option<u64> {
    if let Some('1'..='9') = chars.peek() {
        let mut value = 0u64;
        while let Some('0'..='9') = chars.peek() {
            let x = chars.next().and_then(|c| c.to_digit(10)).unwrap() as u64;
            value = value * 10 + x;
        }
        Some(value)
    } else {
        None
    }
}

fn consume_signed(chars: &mut Countable<Chars>) -> Option<i64> {
    match chars.peek() {
        Some('-') => {
            chars.next();
            consume_unsigned(chars).map(|u| -(u as i64))
        }
        Some('1'..='9') => consume_unsigned(chars).map(|u| u as i64),
        Some('0') => {
            chars.next();
            Some(0)
        }
        _ => None,
    }
}

static TOKENIZERS: &[MatchTokenFn] = &[
    |chars| {
        // number
        let lhs = consume_signed(chars)?;

        let rhs = chars
            .next()
            .and_then(|c| if c == '.' { chars.next() } else { None })
            .and_then(|c| if c == '.' { chars.peek() } else { None })
            .and_then(|_| consume_signed(chars))?;
        Some(Token::Range(lhs, rhs))
    },
    |chars| {
        // number
        if let Some('1'..='9') = chars.peek() {
            let value = consume_unsigned(chars)?;
            match chars.peek() {
                Some('d' | 'D') => {
                    chars.next();
                    let rhs = consume_unsigned(chars)?;
                    return Some(Token::Dice(value, rhs));
                }
                Some('a' | 'A') => {
                    chars.next();
                    return Some(Token::Advantage(value));
                }
                Some('z' | 'Z') => {
                    chars.next();
                    return Some(Token::Disadvantage(value));
                }
                Some('p' | 'P') => {
                    chars.next();
                    return Some(Token::Pick(value));
                }
                Some('c' | 'C') => {
                    chars.next();
                    return Some(Token::Choose(value));
                }
                _ => {}
            }
            return Some(Token::Number(value));
        } else if let Some('0') = chars.next() {
            return Some(Token::Number(0));
        }
        None
    },
    |chars| {
        // plus
        if let Some('+') = chars.next() {
            return Some(Token::Plus);
        }
        None
    },
    |chars| {
        // minus
        if let Some('-') = chars.next() {
            return Some(Token::Minus);
        }
        None
    },
    |chars| {
        // mulitplu
        if let Some('*') = chars.next() {
            return Some(Token::Multiply);
        }
        None
    },
    |chars| {
        // advantage
        if let Some('a' | 'A') = chars.next() {
            return Some(Token::Advantage(1));
        }
        None
    },
    |chars| {
        // disadvantage
        if let Some('z' | 'Z') = chars.next() {
            return Some(Token::Disadvantage(1));
        }
        None
    },
    |chars| {
        // pick
        if let Some('p' | 'P') = chars.next() {
            return Some(Token::Pick(1));
        }
        None
    },
    |chars| {
        // choose
        if let Some('c' | 'C') = chars.next() {
            return Some(Token::Choose(1));
        }
        None
    },
    |chars| {
        // dice
        if let Some('d' | 'D') = chars.next() {
            if let Some('1'..='9') = chars.peek() {
                let mut rhs = 0u64;
                while let Some('0'..='9') = chars.peek() {
                    let x = chars.next().and_then(|c| c.to_digit(10)).unwrap() as u64;
                    rhs = rhs * 10 + x;
                }
                return Some(Token::Dice(1, rhs));
            }
        }
        None
    },
    |chars| {
        // left parenthesis
        if let Some('(') = chars.next() {
            return Some(Token::LeftParenthesis);
        }
        None
    },
    |chars| {
        // right parenthesis
        if let Some(')') = chars.next() {
            return Some(Token::RightParenthesis);
        }
        None
    },
    |chars| {
        // comma
        if let Some(',') = chars.next() {
            return Some(Token::Comma);
        }
        None
    },
    |chars| {
        // unknown
        if let Some(c) = chars.next() {
            for tokenizer in TOKENIZERS {
                let mut temp_it = chars.clone();
                match tokenizer(&mut temp_it) {
                    Some(Token::Unknown(rhs)) => {
                        *chars = temp_it;
                        return Some(Token::Unknown(format!("{}{}", c, rhs)));
                    }
                    Some(_) => break,
                    None => continue,
                }
            }
            return Some(Token::Unknown(format!("{}", c)));
        }
        None
    },
];

#[derive(Debug)]
pub(crate) struct Tokenizer<'a> {
    input: &'a str,
    chars: Countable<Chars<'a>>,
    lookahead: Option<Token>,
    lookahead_char_count: usize,
}

impl<'a, 'b> Tokenizer<'a> {
    pub fn new(input: &'b str) -> Self
    where
        'b: 'a,
    {
        let mut new = Self {
            input,
            chars: Countable::new(input.chars()),
            lookahead: Some(Token::Number(0)), // will be dropped on first next
            lookahead_char_count: 0,
        };
        new.next();
        new
    }

    pub fn peek(&self) -> Option<Token> {
        self.lookahead.clone()
    }

    pub fn expended_count(&self) -> usize {
        self.chars.curr_index() - self.lookahead_char_count
    }

    pub fn peek_token_count(&self) -> usize {
        self.lookahead_char_count
    }

    pub fn input_str(&self) -> &str {
        self.input
    }

    fn remove_white_space(&mut self) {
        while let Some(' ') = self.chars.peek() {
            self.chars.next();
        }
    }
}

impl Iterator for Tokenizer<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.lookahead.clone();
        if next.is_some() {
            self.remove_white_space();
            self.lookahead = (|| {
                for tokenizer in TOKENIZERS {
                    let mut temp_it = self.chars.clone();
                    if let Some(token) = tokenizer(&mut temp_it) {
                        self.lookahead_char_count = temp_it.diff(&self.chars);
                        self.chars.advance_by(self.lookahead_char_count);
                        return Some(token);
                    }
                }
                self.lookahead_char_count = 0;
                None
            })();
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static BASE_DICE_STRING: &str = "123 + 2p(123,z4d2, d8, 4d4d4)1..=2";

    #[test]
    fn test_tokenizer_next() {
        let mut tokenizer = Tokenizer::new(BASE_DICE_STRING);

        assert_eq!(tokenizer.next(), Some(Token::Number(123)));
        assert_eq!(tokenizer.next(), Some(Token::Plus));
        assert_eq!(tokenizer.next(), Some(Token::Pick(2)));
        assert_eq!(tokenizer.next(), Some(Token::LeftParenthesis));
        assert_eq!(tokenizer.next(), Some(Token::Number(123)));
        assert_eq!(tokenizer.next(), Some(Token::Comma));
        assert_eq!(tokenizer.next(), Some(Token::Disadvantage(1)));
        assert_eq!(tokenizer.next(), Some(Token::Dice(4, 2)));
        assert_eq!(tokenizer.next(), Some(Token::Comma));
        assert_eq!(tokenizer.next(), Some(Token::Dice(1, 8)));
        assert_eq!(tokenizer.next(), Some(Token::Comma));
        assert_eq!(tokenizer.next(), Some(Token::Dice(4, 4)));
        assert_eq!(tokenizer.next(), Some(Token::Dice(1, 4)));
        assert_eq!(tokenizer.next(), Some(Token::RightParenthesis));
        assert_eq!(tokenizer.next(), Some(Token::Range(1, 2)));
        assert_eq!(tokenizer.next(), None);
    }

    #[test]
    fn test_tokenizer_pos_range() {
        let mut tokenizer = Tokenizer::new("0..=2");
        assert_eq!(tokenizer.next(), Some(Token::Range(0, 2)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("0..2");
        assert_eq!(tokenizer.next(), Some(Token::Range(0, 1)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("-2..2");
        assert_eq!(tokenizer.next(), Some(Token::Range(-2, 1)));
        assert_eq!(tokenizer.next(), None);
    }

    #[test]
    fn test_tokenizer_neg_range() {
        let mut tokenizer = Tokenizer::new("0..=-2");
        assert_eq!(tokenizer.next(), Some(Token::Range(0, -2)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("0..-2");
        assert_eq!(tokenizer.next(), Some(Token::Range(0, -1)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("2..-2");
        assert_eq!(tokenizer.next(), Some(Token::Range(2, -1)));
        assert_eq!(tokenizer.next(), None);
    }

    #[test]
    fn test_tokenizer_zero_range() {
        let mut tokenizer = Tokenizer::new("0..=0");
        assert_eq!(tokenizer.next(), Some(Token::Range(0, 0)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("0..0");
        assert_eq!(tokenizer.next(), Some(Token::Range(0, 0)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("1..=1");
        assert_eq!(tokenizer.next(), Some(Token::Range(1, 1)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("1..1");
        assert_eq!(tokenizer.next(), Some(Token::Range(1, 1)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("-1..=-1");
        assert_eq!(tokenizer.next(), Some(Token::Range(-1, -1)));
        assert_eq!(tokenizer.next(), None);
        let mut tokenizer = Tokenizer::new("-1..-1");
        assert_eq!(tokenizer.next(), Some(Token::Range(-1, -1)));
        assert_eq!(tokenizer.next(), None);
    }

    #[test]
    fn test_tokenizer_unknown() {
        let mut tokenizer = Tokenizer::new("d6 d4 d2 dddd4");

        assert_eq!(tokenizer.next(), Some(Token::Dice(1, 6)));
        assert_eq!(tokenizer.next(), Some(Token::Dice(1, 4)));
        assert_eq!(tokenizer.next(), Some(Token::Dice(1, 2)));
        assert_eq!(tokenizer.next(), Some(Token::Unknown("ddd".into())));
        assert_eq!(tokenizer.next(), Some(Token::Dice(1, 4)));
        assert_eq!(tokenizer.next(), None);
    }

    #[test]
    fn test_tokenizer_peek() {
        let mut tokenizer = Tokenizer::new(BASE_DICE_STRING);

        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), tokenizer.next());
        assert_eq!(tokenizer.peek(), None);
    }
}
