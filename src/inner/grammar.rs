use crate::inner::token::{Token, Tokenizer};
use crate::{ExecDetails, ExecLineDetail, ExecOutput, ExecResult, ExecResultWithDetails};
use core::fmt;
use owo_colors::OwoColorize;
use rand::{seq::SliceRandom, Rng, RngCore};
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

type ExecResultWithLineDetail = Result<(ExecOutput, Option<ExecLineDetail>), String>;

fn vi_to_string(v: &[i64]) -> String {
    const MAX_CHARS: usize = 30;
    let mut string = format!(
        "({})",
        v.iter()
            .map(i64::to_string)
            .intersperse(", ".into())
            .collect::<String>()
    );

    for i in (1..=3).rev() {
        if string.len() > MAX_CHARS {
            string = format!(
                "({}, ({} more..))",
                v.iter()
                    .take(i)
                    .map(i64::to_string)
                    .intersperse(", ".into())
                    .collect::<String>(),
                v.len() - i,
            );
        }
    }

    if string.len() > MAX_CHARS {
        string = format!("({} values..)", v.len(),);
    }

    string
}

impl ExecOutput {
    pub fn value(&self) -> i64 {
        match self {
            ExecOutput::Value(n) => *n,
            ExecOutput::Array(vec) => vec.iter().sum::<i64>(),
        }
    }

    pub fn array(&self) -> Result<Vec<i64>, String> {
        match self {
            ExecOutput::Value(n) => Err(format!("Expected array, but got Value({}) instead", n)),
            ExecOutput::Array(vec) => Ok(vec.clone()),
        }
    }

    fn with_line_detail(
        self,
        name: &'static str,
        operation: Option<String>,
    ) -> (ExecOutput, Option<ExecLineDetail>) {
        let line_detail = operation.map(|operation| ExecLineDetail {
            name,
            operation,
            output: self.to_string(),
        });
        (self, line_detail)
    }

    fn with_details(self, details: Vec<ExecLineDetail>) -> ExecDetails {
        ExecDetails {
            output: self,
            details,
        }
    }
}

impl fmt::Display for ExecOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecOutput::Value(n) => write!(f, "{}", n),
            ExecOutput::Array(vec) => write!(f, "{}", vi_to_string(vec)),
        }
    }
}

impl ExecDetails {
    pub fn value(&self) -> i64 {
        self.output.value()
    }

    pub fn result_string(&self) -> String {
        match &self.output {
            ExecOutput::Value(i) => i.to_string(),
            ExecOutput::Array(v) => vi_to_string(v),
        }
    }

    pub fn raw_string(&self) -> String {
        match &self.output {
            ExecOutput::Value(i) => i.to_string(),
            ExecOutput::Array(v) => v
                .iter()
                .map(i64::to_string)
                .intersperse(",".into())
                .collect::<String>(),
        }
    }
}

#[derive(Debug, Clone)]
struct TestLineDetail {
    name: &'static str,
    operation: String,
    time_taken: Duration,
}

#[derive(Debug, Clone)]
pub struct TestDetails {
    pub output: Vec<i64>,
    time_taken: Duration,
    details: Vec<TestLineDetail>,
}

impl fmt::Display for TestLineDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let front_width = f.width().unwrap_or(8);
        writeln!(
            f,
            "{:>front_width$} {:8} => {:>4.2}s",
            self.name.bold().bright_yellow(),
            self.operation.bright_magenta(),
            self.time_taken.as_secs_f32(),
        )
    }
}

impl fmt::Display for TestDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let front_width = f.width().unwrap_or(8);
        if let Some(middle_width) = self
            .details
            .iter()
            .map(|line| line.operation.len())
            .max()
            .map(|w| w.max(15))
        {
            for line in self.details.iter() {
                writeln!(
                    f,
                    "{:>front_width$} {:middle_width$} => {:>4.2}s",
                    line.name.bold().bright_yellow(),
                    line.operation.bright_magenta(),
                    line.time_taken.as_millis(),
                )?;
            }
            write!(
                f,
                "{:>front_width$} {:.<middle_width$} => {:>4.2}s",
                "Finished".bold().bright_cyan(),
                "",
                self.time_taken.as_secs_f32(),
            )
        } else {
            write!(
                f,
                "{:>front_width$} => {:>4.2}s",
                "Result".bold().bright_cyan(),
                self.time_taken.as_secs_f32(),
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinMax {
    min: i64,
    max: i64,
}

pub enum MinMaxOutput {
    Value(MinMax),
    Array(Vec<MinMax>),
}

type MinMaxResult = Result<MinMaxOutput, String>;

impl MinMaxOutput {
    pub fn value(&self) -> MinMax {
        match self {
            MinMaxOutput::Value(m) => m.clone(),
            MinMaxOutput::Array(vec) => {
                vec.iter().fold(MinMax { min: 0, max: 0 }, |acc, x| MinMax {
                    min: acc.min + x.min,
                    max: acc.max + x.max,
                })
            }
        }
    }

    pub fn array(&self) -> Result<Vec<MinMax>, String> {
        match self {
            MinMaxOutput::Value(m) => {
                Err(format!("Expected array, but got Value({:?}) instead", m))
            }
            MinMaxOutput::Array(vec) => Ok(vec.clone()),
        }
    }
}

macro_rules! exec_fn {
    () => {
        Box<dyn Fn(&mut dyn RngCore) -> ExecResult>
    };
    ($($T:ty), *) => {
        Box<dyn Fn($($T), *, &mut dyn RngCore) -> ExecResult>
    };
}

macro_rules! min_max_fn {
    () => {
        Box<dyn Fn() -> MinMaxResult>
    };
    ($($T:ty), *) => {
        Box<dyn Fn($($T), *) -> MinMaxResult>
    };
}

macro_rules! to_string_fn {
    () => {
        Box<dyn Fn() -> String>
    };
    ($($T:ty), *) => {
        Box<dyn Fn($($T), *) -> String>
    };
}

pub(crate) enum GrammarRule {
    Aggregate {
        name: &'static str,
        children: Vec<GrammarRule>,
    },
    Number {
        name: &'static str,
        number: i64,
    },
    Generator {
        name: &'static str,
        size: usize,
        exec: exec_fn!(),
        min_max: min_max_fn!(),
        to_string: to_string_fn!(),
    },
    UnaryNumber {
        name: &'static str,
        operation: String,
        size: usize,
        child: Box<GrammarRule>,
        exec: exec_fn!(i64),
        min_max: min_max_fn!(MinMax),
        to_string: to_string_fn!(i64),
    },
    UnaryArray {
        name: &'static str,
        operation: String,
        size: usize,
        child: Box<GrammarRule>,
        exec: exec_fn!(&Vec<i64>),
        min_max: min_max_fn!(&Vec<MinMax>),
        to_string: to_string_fn!(&Vec<i64>),
    },
    Binary {
        name: &'static str,
        operation: String,
        size: usize,
        lhs: Box<GrammarRule>,
        rhs: Box<GrammarRule>,
        exec: exec_fn!(i64, i64),
        min_max: min_max_fn!(MinMax, MinMax),
        to_string: to_string_fn!(i64, i64),
    },
}

impl fmt::Display for GrammarRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarRule::Aggregate { children, .. } => {
                write!(f, "(")?;
                for (i, child) in children.iter().enumerate() {
                    if i != 0usize {
                        write!(f, ", ")?;
                    }
                    child.fmt(f)?;
                }
                write!(f, ")")
            }
            GrammarRule::Number { number, .. } => write!(f, "{}", number),
            GrammarRule::Generator { to_string, .. } => write!(f, "{}", to_string()),
            GrammarRule::UnaryNumber { name, child, .. } => {
                write!(f, "{}(", name)?;
                child.fmt(f)?;
                write!(f, ")")
            }
            GrammarRule::UnaryArray { name, child, .. } => {
                write!(f, "{}(", name)?;
                child.fmt(f)?;
                write!(f, ")")
            }
            GrammarRule::Binary { name, lhs, rhs, .. } => {
                write!(f, "{}(", name)?;
                lhs.fmt(f)?;
                write!(f, ", ")?;
                rhs.fmt(f)?;
                write!(f, ")")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarExecOptions {
    pub is_debug: bool,
}

#[derive(Debug, Clone)]
pub struct GrammarTestOptions {
    pub is_debug: bool,
    pub test_size: usize,
}

type ExecStackFn = Rc<
    dyn Fn(
        &Vec<ExecOutput>,
        &mut dyn rand::RngCore,
        &GrammarExecOptions,
    ) -> ExecResultWithLineDetail,
>;

#[derive(Clone)]
struct StackFn {
    name: &'static str,
    operation: String,
    exec: ExecStackFn,
}

impl fmt::Display for StackFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name.fmt(f)
    }
}

impl fmt::Debug for StackFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StackFn({})", self.name)
    }
}

impl GrammarRule {
    fn num(num: u64) -> Self {
        Self::Number {
            name: "Num",
            number: num as i64,
        }
    }

    fn arr(children: Vec<Self>) -> Self {
        Self::Aggregate {
            name: "Agg",
            children,
        }
    }

    fn sum(lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            name: "Sum",
            operation: ".. + ..".into(),
            size: 1,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            exec: Box::new(move |lhs, rhs, _| Ok(ExecOutput::Value(lhs + rhs))),
            min_max: Box::new(move |lhs, rhs| {
                Ok(MinMaxOutput::Value(MinMax {
                    min: lhs.min + rhs.min,
                    max: lhs.max + rhs.max,
                }))
            }),
            to_string: Box::new(move |lhs, rhs| format!("{} + {}", lhs, rhs)),
        }
    }

    fn sub(lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            name: "Sub",
            operation: ".. - ..".into(),
            size: 1,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            exec: Box::new(move |lhs, rhs, _| Ok(ExecOutput::Value(lhs - rhs))),
            min_max: Box::new(move |lhs, rhs| {
                Ok(MinMaxOutput::Value(MinMax {
                    min: lhs.min - rhs.max,
                    max: lhs.max - rhs.min,
                }))
            }),
            to_string: Box::new(move |lhs, rhs| format!("{} - {}", lhs, rhs)),
        }
    }

    fn mul(lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            name: "Mul",
            operation: ".. x ..".into(),
            size: 1,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            exec: Box::new(move |lhs, rhs, _| Ok(ExecOutput::Value(lhs * rhs))),
            min_max: Box::new(move |lhs, rhs| {
                let possible_values = [
                    lhs.min * rhs.min,
                    lhs.min * rhs.max,
                    lhs.max * rhs.min,
                    lhs.max * lhs.max,
                ];

                Ok(MinMaxOutput::Value(MinMax {
                    min: *possible_values.iter().min().unwrap(),
                    max: *possible_values.iter().max().unwrap(),
                }))
            }),
            to_string: Box::new(move |lhs, rhs| format!("{} x {}", lhs, rhs)),
        }
    }

    fn adv(lhs: u64, rhs: Self) -> Self {
        Self::UnaryArray {
            name: "Adv",
            operation: format!("{}A(...)", lhs),
            size: lhs as usize,
            child: Box::new(rhs),
            exec: Box::new(move |rhs, _| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut new = rhs.clone();
                new.sort_by(|a, b| b.cmp(a));
                new.resize(new_size, 0);
                Ok(ExecOutput::Array(new))
            }),
            min_max: Box::new(move |rhs| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut min_values = rhs.iter().map(|m| m.min).collect::<Vec<_>>();
                let mut max_values = rhs.iter().map(|m| m.max).collect::<Vec<_>>();

                min_values.sort_by(|a, b| b.cmp(a));
                max_values.sort_by(|a, b| b.cmp(a));
                min_values.resize(new_size, 0);
                max_values.resize(new_size, 0);

                Ok(MinMaxOutput::Array(
                    min_values
                        .iter()
                        .zip(max_values.iter())
                        .map(|m| MinMax {
                            min: *m.0,
                            max: *m.1,
                        })
                        .collect::<_>(),
                ))
            }),
            to_string: Box::new(move |rhs| format!("{}A{}", lhs, vi_to_string(rhs))),
        }
    }

    fn dis(lhs: u64, rhs: Self) -> Self {
        Self::UnaryArray {
            name: "Dis",
            operation: format!("{}Z(...)", lhs),
            size: lhs as usize,
            child: Box::new(rhs),
            exec: Box::new(move |rhs, _| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut new = rhs.clone();
                new.sort();
                while new.len() != new_size {
                    new.pop();
                }
                Ok(ExecOutput::Array(new))
            }),
            min_max: Box::new(move |rhs| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut min_values = rhs.iter().map(|m| m.min).collect::<Vec<_>>();
                let mut max_values = rhs.iter().map(|m| m.max).collect::<Vec<_>>();

                min_values.sort();
                max_values.sort();
                min_values.resize(new_size, 0);
                max_values.resize(new_size, 0);

                Ok(MinMaxOutput::Array(
                    min_values
                        .iter()
                        .zip(max_values.iter())
                        .map(|m| MinMax {
                            min: *m.0,
                            max: *m.1,
                        })
                        .collect::<_>(),
                ))
            }),
            to_string: Box::new(move |rhs| format!("{}Z{}", lhs, vi_to_string(rhs))),
        }
    }

    fn cho(lhs: u64, rhs: Self) -> Self {
        Self::UnaryArray {
            name: "Cho",
            size: lhs as usize,
            operation: format!("{}C(...)", lhs),
            child: Box::new(rhs),
            exec: Box::new(move |rhs, rng| {
                let new_size = lhs as usize;
                let mut r: Vec<i64> = vec![];
                while r.len() != new_size {
                    let i = rng.gen_range(0..rhs.len());
                    r.push(rhs[i]);
                }
                Ok(ExecOutput::Array(r))
            }),
            min_max: Box::new(move |rhs| {
                let min = rhs.iter().map(|o| o.min).min().unwrap();
                let max = rhs.iter().map(|o| o.max).max().unwrap();
                let new_size = lhs as usize;
                Ok(MinMaxOutput::Array(vec![MinMax { min, max }; new_size]))
            }),
            to_string: Box::new(move |rhs| format!("{}C{}", lhs, vi_to_string(rhs))),
        }
    }

    fn pic(lhs: u64, rhs: Self) -> Self {
        Self::UnaryArray {
            name: "Pic",
            operation: format!("{}P(...)", lhs),
            size: lhs as usize,
            child: Box::new(rhs),
            exec: Box::new(move |rhs, rng| {
                let k = lhs as usize;
                let n = rhs.len();
                if rhs.len() < k {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                const THRESHOLD: f64 = 0.7;
                let v = if (k as f64) < (THRESHOLD * n as f64) {
                    let mut seen = vec![false; rhs.len()];
                    (0..k)
                        .map(|_| loop {
                            let i = rng.gen_range(0..rhs.len());
                            if seen[i] {
                                continue;
                            }
                            seen[i] = true;
                            return rhs[i];
                        })
                        .collect()
                } else {
                    let mut v = rhs.clone();
                    v.shuffle(rng);
                    v.resize(k, 0);
                    v
                };
                Ok(ExecOutput::Array(v))
            }),
            min_max: Box::new(move |rhs| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut min_values = rhs.iter().map(|m| m.min).collect::<Vec<_>>();
                let mut max_values = rhs.iter().map(|m| m.max).collect::<Vec<_>>();

                min_values.sort();
                max_values.sort_by(|a: &i64, b| b.cmp(a));
                min_values.resize(new_size, 0);
                max_values.resize(new_size, 0);

                Ok(MinMaxOutput::Array(
                    min_values
                        .iter()
                        .zip(max_values.iter())
                        .map(|m| MinMax {
                            min: *m.0,
                            max: *m.1,
                        })
                        .collect::<_>(),
                ))
            }),
            to_string: Box::new(move |rhs| format!("{}P{}", lhs, vi_to_string(rhs))),
        }
    }

    fn die(lhs: u64, rhs: u64) -> Self {
        Self::Generator {
            name: "Die",
            size: lhs as usize,
            exec: Box::new(move |rng| {
                let new_size = lhs as usize;
                let mut r: Vec<i64> = vec![];
                while r.len() != new_size {
                    let x = rng.gen_range(1..=rhs);
                    r.push(x as i64);
                }
                Ok(ExecOutput::Array(r))
            }),
            min_max: Box::new(move || {
                let min = 1;
                let max = rhs as i64;
                let new_size = lhs as usize;
                Ok(MinMaxOutput::Array(vec![MinMax { min, max }; new_size]))
            }),
            to_string: Box::new(move || {
                format!(
                    "{}D{}",
                    if lhs == 1 { "".into() } else { lhs.to_string() },
                    rhs
                )
            }),
        }
    }

    fn neg(rhs: Self) -> Self {
        Self::UnaryNumber {
            name: "Neg",
            operation: "-(..)".into(),
            size: 1,
            child: Box::new(rhs),
            exec: Box::new(move |rhs, _| Ok(ExecOutput::Value(-rhs))),
            min_max: Box::new(move |rhs| {
                let min = -rhs.max;
                let max = -rhs.min;
                Ok(MinMaxOutput::Value(MinMax { min, max }))
            }),
            to_string: Box::new(move |rhs| format!("-({})", rhs)),
        }
    }

    fn rge(lhs: i64, rhs: i64) -> Self {
        let output = if lhs < rhs {
            (lhs..=rhs).collect::<Vec<_>>()
        } else {
            (rhs..=lhs).rev().collect()
        };

        let min_max_arr = output
            .iter()
            .cloned()
            .map(|i| MinMax { min: i, max: i })
            .collect::<Vec<_>>();

        Self::Generator {
            name: "Rge",
            size: output.len(),
            exec: Box::new(move |_| Ok(ExecOutput::Array(output.clone()))),
            min_max: Box::new(move || Ok(MinMaxOutput::Array(min_max_arr.clone()))),
            to_string: Box::new(move || format!("({lhs}..{rhs})")),
        }
    }

    fn len(&self) -> usize {
        match &self {
            GrammarRule::Aggregate { children, .. } => children.len(),
            GrammarRule::Number { .. } => 1,
            GrammarRule::Generator { size, .. } => *size,
            GrammarRule::UnaryNumber { size, .. } => *size,
            GrammarRule::UnaryArray { size, .. } => *size,
            GrammarRule::Binary { size, .. } => *size,
        }
    }

    fn min_max(&self) -> MinMaxResult {
        match &self {
            GrammarRule::Aggregate { children, .. } => Ok(MinMaxOutput::Array(
                children
                    .iter()
                    .map(|c| c.min_max().map(|o| o.value()))
                    .collect::<Result<Vec<MinMax>, String>>()?,
            )),
            GrammarRule::Number { number, .. } => Ok(MinMaxOutput::Value(MinMax {
                min: *number,
                max: *number,
            })),
            GrammarRule::Generator { min_max, .. } => min_max(),
            GrammarRule::UnaryNumber { child, min_max, .. } => {
                min_max(child.min_max().map(|o| o.value())?)
            }
            GrammarRule::UnaryArray { child, min_max, .. } => {
                min_max(&child.min_max().and_then(|o| o.array())?)
            }
            GrammarRule::Binary {
                lhs, rhs, min_max, ..
            } => min_max(
                lhs.min_max().map(|o| o.value())?,
                rhs.min_max().map(|o| o.value())?,
            ),
        }
    }

    fn dfs(self, callstack: &mut Vec<StackFn>) -> usize {
        let f: StackFn = match self {
            GrammarRule::Aggregate { name, children } => {
                let indices: Vec<usize> = children
                    .into_iter()
                    .map(|child| child.dfs(callstack))
                    .collect();
                StackFn {
                    name,
                    operation: "Aggregate".into(),
                    exec: Rc::new(move |stack: &Vec<ExecOutput>, _, _| {
                        Ok(ExecOutput::Array(
                            indices
                                .iter()
                                .map(|i| stack.get(*i).unwrap().value())
                                .collect(),
                        )
                        .with_line_detail(name, None))
                    }),
                }
            }
            GrammarRule::Number { name, number } => StackFn {
                name,
                operation: format!("{}", number),
                exec: Rc::new(move |_: &Vec<ExecOutput>, _, _| {
                    Ok(ExecOutput::Value(number).with_line_detail(name, None))
                }),
            },
            GrammarRule::Generator {
                name,
                exec,
                to_string,
                ..
            } => StackFn {
                name,
                operation: (to_string)(),
                exec: Rc::new(move |_: &Vec<ExecOutput>, rng, options| {
                    Ok(exec(rng)?.with_line_detail(name, options.is_debug.then(&to_string)))
                }),
            },
            GrammarRule::UnaryNumber {
                name,
                operation,
                child,
                exec,
                to_string,
                ..
            } => {
                let index = child.dfs(callstack);
                StackFn {
                    name,
                    operation,
                    exec: Rc::new(move |stack: &Vec<ExecOutput>, rng, options| {
                        let rhs = stack.get(index).unwrap().value();
                        Ok(exec(rhs, rng)?
                            .with_line_detail(name, options.is_debug.then(|| to_string(rhs))))
                    }),
                }
            }
            GrammarRule::UnaryArray {
                name,
                operation,
                child,
                exec,
                to_string,
                ..
            } => {
                let index = child.dfs(callstack);
                StackFn {
                    name,
                    operation,
                    exec: Rc::new(move |stack: &Vec<ExecOutput>, rng, options| {
                        let v = &stack.get(index).unwrap().array()?;
                        Ok(exec(v, rng)?
                            .with_line_detail(name, options.is_debug.then(|| to_string(v))))
                    }),
                }
            }
            GrammarRule::Binary {
                name,
                operation,
                lhs,
                rhs,
                exec,
                to_string,
                ..
            } => {
                let lhs_index = lhs.dfs(callstack);
                let rhs_index = rhs.dfs(callstack);
                StackFn {
                    name,
                    operation,
                    exec: Rc::new(move |stack: &Vec<ExecOutput>, rng, options| {
                        let lhs = stack.get(lhs_index).unwrap().value();
                        let rhs = stack.get(rhs_index).unwrap().value();
                        Ok(exec(lhs, rhs, rng)?
                            .with_line_detail(name, options.is_debug.then(|| to_string(lhs, rhs))))
                    }),
                }
            }
        };
        let n = callstack.len();
        callstack.push(f);
        n
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

#[derive(Debug, Clone)]
pub(crate) struct GrammarError {
    error_index: usize,
    error_length: usize,
    input_string: String,
    error_type: GrammarErrorType,
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

impl GrammarError {
    pub(crate) fn formatted_error_string(&self) -> String {
        let index = self.error_index;
        format!(
            "{:index$}{} {}",
            "",
            (0..self.error_length)
                .map(|_| '^')
                .collect::<String>()
                .red()
                .bold(),
            self.error_type.to_string().red().bold()
        )
    }
}

/**
 * Expression
 *    = Term (("+" / "-") Expression)+
 */
fn expression(tokenizer: &mut Tokenizer) -> Result<GrammarRule, GrammarError> {
    let left = term(tokenizer)?;
    if let Some(Token::Plus) = tokenizer.peek() {
        tokenizer.next();
        let right = expression(tokenizer)?;
        return Ok(GrammarRule::sum(left, right));
    } else if let Some(Token::Minus) = tokenizer.peek() {
        tokenizer.next();
        let right = expression(tokenizer)?;
        return Ok(GrammarRule::sub(left, right));
    }
    Ok(left)
}

/**
 * Term
 *    = Negative ("*" Term)+
 */
fn term(tokenizer: &mut Tokenizer) -> Result<GrammarRule, GrammarError> {
    let left = negative(tokenizer)?;
    if let Some(Token::Multiply) = tokenizer.peek() {
        tokenizer.next();
        let right = term(tokenizer)?;
        return Ok(GrammarRule::mul(left, right));
    }
    Ok(left)
}

/**
 * Negative
 *    = ("-")+ Any
 */
fn negative(tokenizer: &mut Tokenizer) -> Result<GrammarRule, GrammarError> {
    if let Some(Token::Minus) = tokenizer.peek() {
        tokenizer.next();
        let right = any(tokenizer)?;
        return Ok(GrammarRule::neg(right));
    }
    any(tokenizer)
}

/**
 * Any
 *    = Number / Primary
 */
fn any(tokenizer: &mut Tokenizer) -> Result<GrammarRule, GrammarError> {
    if let Some(Token::Number(n)) = tokenizer.peek() {
        tokenizer.next();
        return Ok(GrammarRule::num(n));
    }
    primary(tokenizer, "Expression")
}

/**
 * Primary
 *    = Dice / Modifier / Array
 */
fn primary(tokenizer: &mut Tokenizer, expected: &'static str) -> Result<GrammarRule, GrammarError> {
    match tokenizer.peek() {
        Some(Token::LeftParenthesis) => {
            tokenizer.next();
            let mut arr: Vec<GrammarRule> = Vec::new();
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
            Ok(GrammarRule::arr(arr))
        }
        Some(Token::Dice(lhs, rhs)) => {
            tokenizer.next();
            Ok(GrammarRule::die(lhs, rhs))
        }
        Some(Token::Advantage(lhs)) => {
            tokenizer.next();
            let expended_count_before = tokenizer.expended_count();
            let rhs = primary(tokenizer, "Array")?;
            let len = rhs.len();
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
            Ok(GrammarRule::adv(lhs, rhs))
        }
        Some(Token::Disadvantage(lhs)) => {
            tokenizer.next();
            let expended_count_before = tokenizer.expended_count();
            let rhs = primary(tokenizer, "Array")?;
            let len = rhs.len();
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
            Ok(GrammarRule::dis(lhs, rhs))
        }
        Some(Token::Pick(lhs)) => {
            tokenizer.next();
            let expended_count_before = tokenizer.expended_count();
            let rhs = primary(tokenizer, "Array")?;
            let len = rhs.len();
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
            Ok(GrammarRule::pic(lhs, rhs))
        }
        Some(Token::Choose(lhs)) => {
            tokenizer.next();
            let rhs = primary(tokenizer, "Array")?;
            Ok(GrammarRule::cho(lhs, rhs))
        }
        Some(Token::Range(lhs, rhs)) => {
            tokenizer.next();
            Ok(GrammarRule::rge(lhs, rhs))
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

const HORIZONTAL_BAR_SECTIONS: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

fn choose_char_for_progress_at_interval(
    interval_start: f64,
    interval_size: f64,
    progress_value: f64,
) -> char {
    let normalized_position = ((progress_value - interval_start) / interval_size).clamp(0.0, 1.0);
    let index = (normalized_position * 8.0).floor() as usize;
    HORIZONTAL_BAR_SECTIONS[index]
}

fn progress_string(progress_size: f64, interval_count: usize, progress_value: f64) -> String {
    let normalized_interval_count = interval_count.max(1);
    let interval_size = progress_size / normalized_interval_count as f64;
    let which_interval =
        ((progress_value / interval_size).floor() as usize).clamp(0, interval_count);
    let bottom_interval_count = which_interval;
    let top_interval_count = interval_count - which_interval - 1;
    let interval_start = interval_size * bottom_interval_count as f64;
    let character =
        choose_char_for_progress_at_interval(interval_start, interval_size, progress_value);

    format!(
        "{:█<bottom_interval_count$}{}{:<top_interval_count$}",
        "", character, ""
    )
}

#[derive(Debug, Clone)]
pub(crate) struct Grammar {
    // root: GrammarRule,
    compiled_string: String,
    callstack: Vec<StackFn>,
    min_max: MinMax,
}

impl Grammar {
    pub(crate) fn parse(input: &str) -> Result<Self, GrammarError> {
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
        let mut callstack: Vec<StackFn> = vec![];
        let compiled_string = result.to_string();
        let min_max = result.min_max().unwrap().value();
        let _ = result.dfs(&mut callstack);

        Ok(Self {
            compiled_string,
            callstack,
            min_max,
        })
    }

    pub(crate) fn min(&self) -> i64 {
        self.min_max.min
    }

    pub(crate) fn max(&self) -> i64 {
        self.min_max.max
    }

    pub(crate) fn exec(
        &self,
        rng: &mut impl rand::Rng,
        options: GrammarExecOptions,
    ) -> ExecResultWithDetails {
        let mut stack: Vec<ExecOutput> = vec![];
        let mut details: Vec<ExecLineDetail> = vec![];
        for stack_fn in &self.callstack {
            let (output, line_detail) = (stack_fn.exec)(&stack, rng, &options)?;
            if let Some(detail) = line_detail {
                details.push(detail);
            }
            stack.push(output);
        }
        Ok(stack.pop().unwrap().with_details(details))
    }

    pub(crate) fn test(
        &self,
        rng: &mut impl rand::Rng,
        options: GrammarTestOptions,
    ) -> Result<TestDetails, String> {
        let n = options.test_size;
        let is_debug = options.is_debug;
        let start_time = Instant::now();
        let mut stacks: Vec<Vec<ExecOutput>> = vec![vec![]; n];
        let mut details: Vec<TestLineDetail> = vec![];
        let exec_options = GrammarExecOptions { is_debug: false };

        let front_width = 12;
        let middle_width = 15;

        if options.is_debug {
            eprintln!(
                "{:>front_width$} {:<middle_width$} => {}",
                "Running".bold().bright_cyan(),
                "test of size",
                options.test_size.bold().bright_yellow()
            );
        }

        for (j, stack_fn) in self.callstack.iter().enumerate() {
            let debug_message = if is_debug {
                let message = format!(
                    "{:>front_width$} {:<middle_width$} =>",
                    stack_fn.name.bright_yellow().bold(),
                    stack_fn.operation.bright_magenta()
                );
                eprint!("{message}");
                Some(message)
            } else {
                None
            };

            let stack_fn_start = Instant::now();
            let interval = if n < 100 { 1 } else { n / 100 };
            for (i, stack) in stacks.iter_mut().enumerate() {
                if i % interval == 0 {
                    if let Some(message) = &debug_message {
                        let percent = (i / interval) + 1;
                        eprint!(
                            "\x1b[2K\r{message} {}▏{percent}%",
                            progress_string(n as f64, 20, i as f64),
                        );
                    } else {
                        let interval = if (self.callstack.len() * n) < 100 {
                            1
                        } else {
                            (self.callstack.len() * n) / 100
                        };
                        let percent = ((j * n + i) / interval) + 1;
                        eprint!(
                            "\x1b[2K\r{:>front_width$} {:<middle_width$} => {}▏{percent:3}% {:<6.2}s",
                            "Testing".bold().bright_cyan(),
                            stack_fn.operation.bright_magenta(),
                            progress_string((self.callstack.len() * n) as f64, 20, (j * n + i) as f64),
                            start_time.elapsed().as_secs_f32().bold()
                        );
                    }
                }
                let (output, _) = (stack_fn.exec)(stack, rng, &exec_options)?;
                stack.push(output);
            }
            if let Some(message) = &debug_message {
                eprintln!(
                    "\x1b[2K\r{message} {:<5}ms",
                    stack_fn_start.elapsed().as_millis()
                );
                details.push(TestLineDetail {
                    name: stack_fn.name,
                    operation: stack_fn.operation.clone(),
                    time_taken: stack_fn_start.elapsed(),
                });
            }
        }

        if !is_debug {
            eprint!("\x1b[2K\r");
        }

        eprintln!(
            "{:>front_width$} {:.<middle_width$} => {:<6.2}s",
            "Finished".bold().bright_green(),
            "",
            start_time.elapsed().as_secs_f32().bold()
        );
        Ok(TestDetails {
            output: stacks.into_iter().map(|s| s[s.len() - 1].value()).collect(),
            time_taken: start_time.elapsed(),
            details,
        })
    }
}

impl fmt::Display for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.compiled_string)
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
        let x = "bbc";
        let result = Grammar::parse(x);
        assert_matches!(
            result,
            Err(GrammarError {
                error_type: GrammarErrorType::UnknownToken(_),
                error_index: 0,
                error_length: 2,
                ..
            })
        );
        if let Err(GrammarError {
            error_type: GrammarErrorType::UnknownToken(s),
            ..
        }) = result
        {
            assert_eq!(s.as_str(), "bb");
        }
    }

    #[test]
    fn test_parse_unexpected_token() {
        let x = "3z4";
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
        let x = "3a(1,2)";
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

        let x = "3a0..2";
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
                error_length: 4,
                input_string: _,
            })
        );
    }

    #[test]
    fn test_min_max() {
        let x = "3d20";
        let result = Grammar::parse(x);
        assert!(result.is_ok());
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 3);
        assert_eq!(grammar.max(), 60);

        let x = "30d200";
        let result = Grammar::parse(x);
        assert!(result.is_ok());
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 30);
        assert_eq!(grammar.max(), 6000);

        let x = "a3d20";
        let result = Grammar::parse(x);
        assert!(result.is_ok());
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 20);

        let x = "z3d20";
        let result = Grammar::parse(x);
        assert!(result.is_ok());
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 20);

        let x = "1..=20";
        let result = Grammar::parse(x);
        assert!(result.is_ok());
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 20);
    }
}
