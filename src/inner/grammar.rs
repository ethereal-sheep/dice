use crate::inner::token::{Token, Tokenizer};
use crate::{
    CompiledConstant, ExecDetails, ExecLineDetail, ExecOutput, ExecResult, ExecResultWithDetails,
    FlattenedFunction, OperationTestInfo, OverallTestInfo, RollOptions, TestDetails,
    TestLineDetail, TestOptions, TestResultWithDetails,
};
use core::fmt;
use either::Either;
use itertools::{kmerge, Itertools};
use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use owo_colors::OwoColorize;
use rand::{seq::SliceRandom, Rng, RngCore};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use std::{mem, thread};
use voracious_radix_sort::RadixSort;

type ExecResultWithLineDetail = Result<(ExecOutput, Option<ExecLineDetail>), String>;

fn vi_to_string(v: &[i64]) -> String {
    const MAX_CHARS: usize = 30;
    let mut string = format!(
        "({})",
        Itertools::intersperse(v.iter().map(i64::to_string), ", ".into()).collect::<String>()
    );

    for i in (1..=3).rev() {
        if string.len() > MAX_CHARS {
            string = format!(
                "({}, ({} more..))",
                Itertools::intersperse(v.iter().take(i).map(i64::to_string), ", ".into())
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

fn vgr_to_string(v: &[GrammarRule]) -> String {
    const MAX_CHARS: usize = 30;
    let mut string = format!(
        "({})",
        Itertools::intersperse(v.iter().map(GrammarRule::to_string), ", ".into())
            .collect::<String>()
    );

    for i in (1..=3).rev() {
        if string.len() > MAX_CHARS {
            string = format!(
                "({}, ({} more..))",
                Itertools::intersperse(v.iter().take(i).map(GrammarRule::to_string), ", ".into())
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

fn factorial(n: u64) -> BigUint {
    if n == 0 || n == 1 {
        BigUint::one()
    } else {
        n * factorial(n - 1)
    }
}

fn range_to_vi(lhs: i64, rhs: i64) -> Vec<i64> {
    if lhs < rhs {
        (lhs..=rhs).collect()
    } else {
        (rhs..=lhs).rev().collect()
    }
}

fn select_indices<T: Clone>(lhs: &[T], rhs: &[i64]) -> Vec<T> {
    rhs.iter()
        .map(|i| lhs[i.rem_euclid(lhs.len() as i64) as usize].clone())
        .collect()
}

impl ExecOutput {
    pub fn value(&self) -> i64 {
        match self {
            ExecOutput::Value(n) => *n,
            ExecOutput::Array(vec) => vec.iter().sum::<i64>(),
        }
    }

    pub fn expect_array(&self) -> Result<Vec<i64>, String> {
        match self {
            ExecOutput::Value(n) => Err(format!("Expected array, but got Value({}) instead", n)),
            ExecOutput::Array(vec) => Ok(vec.clone()),
        }
    }

    pub fn array(&self) -> Vec<i64> {
        match self {
            ExecOutput::Value(n) => vec![*n],
            ExecOutput::Array(vec) => vec.clone(),
        }
    }

    pub fn result_string(&self) -> String {
        match &self {
            ExecOutput::Value(i) => i.to_string(),
            ExecOutput::Array(v) => vi_to_string(v),
        }
    }

    pub fn raw_string(&self) -> String {
        match &self {
            ExecOutput::Value(i) => i.to_string(),
            ExecOutput::Array(v) => {
                Itertools::intersperse(v.iter().map(i64::to_string), ",".into()).collect::<String>()
            }
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
pub(crate) struct MinMax {
    min: i64,
    max: i64,
}

pub(crate) enum MinMaxOutput {
    Value(MinMax),
    Array(Vec<MinMax>),
    Permutation(Vec<MinMax>, Vec<i64>),
}

type MinMaxResult = Result<MinMaxOutput, String>;

impl MinMaxOutput {
    pub(crate) fn value(&self) -> MinMax {
        self.array()
            .into_iter()
            .fold(MinMax { min: 0, max: 0 }, |acc, x| MinMax {
                min: acc.min + x.min,
                max: acc.max + x.max,
            })
    }

    pub(crate) fn expect_array(&self) -> Result<Vec<MinMax>, String> {
        match self {
            MinMaxOutput::Value(m) => {
                Err(format!("Expected array, but got Value({:?}) instead", m))
            }
            any => Ok(any.array()),
        }
    }

    pub(crate) fn array(&self) -> Vec<MinMax> {
        match self {
            MinMaxOutput::Value(m) => vec![m.clone()],
            MinMaxOutput::Array(vec) => vec.clone(),
            MinMaxOutput::Permutation(vec, suffix) => {
                let modded_suffix = suffix
                    .iter()
                    .map(|i| i.rem_euclid(vec.len() as i64) as usize)
                    .collect::<Vec<_>>();

                let mut indices_map = modded_suffix
                    .iter()
                    .fold(vec![0; vec.len()], |mut index_counts, i| {
                        index_counts[*i] += 1;
                        index_counts
                    })
                    .into_iter()
                    .enumerate()
                    .collect::<Vec<_>>();

                indices_map.sort_by(|a, b| match b.1.cmp(&a.1) {
                    std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                    any => any,
                });

                let indices_map = indices_map.into_iter().map(|(i, _)| i).enumerate().fold(
                    vec![0; vec.len()],
                    |mut indices_map: Vec<usize>, (i, u)| {
                        indices_map[u] = i;
                        indices_map
                    },
                );

                modded_suffix
                    .into_iter()
                    .map(|i| vec[indices_map[i]].clone())
                    .collect()
            }
        }
    }

    pub(crate) fn select(&self, rhs: &[i64]) -> Self {
        match self {
            MinMaxOutput::Value(n) => MinMaxOutput::Array(vec![n.clone(); rhs.len()]),
            MinMaxOutput::Array(vec) => MinMaxOutput::Array(select_indices(vec, rhs)),
            MinMaxOutput::Permutation(vec, suffix) => {
                MinMaxOutput::Permutation(vec.clone(), select_indices(suffix, rhs))
            }
        }
    }
}

pub(crate) enum ExecFn<R, C>
where
    R: Send + Sync,
    C: Send + Sync,
{
    Random(usize, R),
    Constant(C),
}

impl<R, C> ExecFn<R, C>
where
    R: Send + Sync,
    C: Send + Sync,
{
    fn variable_count(&self) -> usize {
        match self {
            ExecFn::Random(variable_count, _) => *variable_count,
            ExecFn::Constant(_) => 0,
        }
    }
}

macro_rules! exec_fn {
    () => {
        ExecFn<Box<dyn Fn(&mut dyn RngCore) -> ExecResult + Send + Sync>, Box<dyn Fn() -> ExecResult + Send + Sync>>
    };
    ($($T:ty), *) => {
        ExecFn<Box<dyn Fn($($T), *, &mut dyn RngCore) -> ExecResult + Send + Sync>, Box<dyn Fn($($T), *) -> ExecResult + Send + Sync>>
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
        Box<dyn Fn() -> String + Send + Sync>
    };
    ($($T:ty), *) => {
        Box<dyn Fn($($T), *) -> String + Send + Sync>
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
    Select {
        name: &'static str,
        lhs: Box<GrammarRule>,
        rhs: Vec<i64>,
    },
    Generator {
        name: &'static str,
        size: usize,
        search_space_size: BigUint,
        step_count: usize,
        exec: exec_fn!(),
        to_string: to_string_fn!(),
        min_max: min_max_fn!(),
    },
    UnaryNumber {
        name: &'static str,
        operation: String,
        size: usize,
        child: Box<GrammarRule>,
        exec: exec_fn!(i64),
        to_string: to_string_fn!(i64),
        min_max: min_max_fn!(MinMax),
    },
    UnaryArray {
        name: &'static str,
        operation: String,
        size: usize,
        search_space_size: BigUint,
        step_count: usize,
        child: Box<GrammarRule>,
        exec: exec_fn!(&Vec<i64>),
        to_string: to_string_fn!(&Vec<i64>),
        min_max: min_max_fn!(&Vec<MinMax>),
    },
    Binary {
        name: &'static str,
        operation: String,
        size: usize,
        lhs: Box<GrammarRule>,
        rhs: Box<GrammarRule>,
        exec: exec_fn!(i64, i64),
        to_string: to_string_fn!(i64, i64),
        min_max: min_max_fn!(MinMax, MinMax),
    },
}

impl fmt::Display for GrammarRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            if let Some(output) = self.consteval() {
                return output.fmt(f);
            }
        }
        match self {
            GrammarRule::Aggregate { children, .. } => vgr_to_string(children).fmt(f),
            GrammarRule::Number { number, .. } => write!(f, "{}", number),
            GrammarRule::Generator { to_string, .. } => write!(f, "{}", to_string()),
            GrammarRule::UnaryNumber { name, child, .. } => {
                write!(f, "{}(", name)?;
                child.fmt(f)?;
                write!(f, ")")
            }
            GrammarRule::UnaryArray {
                name, child, size, ..
            } => {
                write!(f, "{}{}(", name, size)?;
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
            GrammarRule::Select { name, lhs, rhs, .. } => {
                write!(f, "{}(", name)?;
                lhs.fmt(f)?;
                write!(f, ", ")?;
                vi_to_string(rhs).fmt(f)?;
                write!(f, ")")
            }
        }
    }
}

type ExecStackFn = Arc<
    dyn Fn(&Vec<ExecOutput>, &mut dyn rand::RngCore, &RollOptions) -> ExecResultWithLineDetail
        + Send
        + Sync,
>;

#[derive(Clone)]
struct StackFn {
    name: &'static str,
    operation: String,
    step_count: usize,
    output_size: usize,
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

    fn sel(lhs: Self, rhs: Vec<i64>) -> Self {
        Self::Select {
            name: "Sel",
            lhs: Box::new(lhs),
            rhs,
        }
    }

    fn sum(lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            name: "Sum",
            operation: ".. + ..".into(),
            size: 1,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            exec: ExecFn::Constant(Box::new(move |lhs, rhs| Ok(ExecOutput::Value(lhs + rhs)))),
            to_string: Box::new(move |lhs, rhs| format!("{} + {}", lhs, rhs)),
            min_max: Box::new(move |lhs, rhs| {
                Ok(MinMaxOutput::Value(MinMax {
                    min: lhs.min + rhs.min,
                    max: lhs.max + rhs.max,
                }))
            }),
        }
    }

    fn sub(lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            name: "Sub",
            operation: ".. - ..".into(),
            size: 1,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            exec: ExecFn::Constant(Box::new(move |lhs, rhs| Ok(ExecOutput::Value(lhs - rhs)))),
            to_string: Box::new(move |lhs, rhs| format!("{} - {}", lhs, rhs)),
            min_max: Box::new(move |lhs, rhs| {
                Ok(MinMaxOutput::Value(MinMax {
                    min: lhs.min - rhs.max,
                    max: lhs.max - rhs.min,
                }))
            }),
        }
    }

    fn mul(lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            name: "Mul",
            operation: ".. x ..".into(),
            size: 1,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            exec: ExecFn::Constant(Box::new(move |lhs, rhs| Ok(ExecOutput::Value(lhs * rhs)))),
            to_string: Box::new(move |lhs, rhs| format!("{} x {}", lhs, rhs)),
            min_max: Box::new(move |lhs, rhs| {
                let possible_values = [
                    lhs.min * rhs.min,
                    lhs.min * rhs.max,
                    lhs.max * rhs.min,
                    lhs.max * rhs.max,
                ];

                Ok(MinMaxOutput::Value(MinMax {
                    min: *possible_values.iter().min().unwrap(),
                    max: *possible_values.iter().max().unwrap(),
                }))
            }),
        }
    }

    fn adv(lhs: u64, rhs: Self) -> Self {
        Self::UnaryArray {
            name: "Adv",
            operation: format!("{}A(...)", lhs),
            size: lhs as usize,
            search_space_size: BigUint::one(),
            step_count: rhs.len() * rhs.len().ilog2() as usize + lhs as usize,
            child: Box::new(rhs),
            exec: ExecFn::Constant(Box::new(move |rhs| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut new = rhs.clone();
                new.voracious_sort();
                new.reverse();
                new.resize(new_size, 0);
                Ok(ExecOutput::Array(new))
            })),
            to_string: Box::new(move |rhs| format!("{}A{}", lhs, vi_to_string(rhs))),
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
        }
    }

    fn dis(lhs: u64, rhs: Self) -> Self {
        Self::UnaryArray {
            name: "Dis",
            operation: format!("{}Z(...)", lhs),
            size: lhs as usize,
            search_space_size: BigUint::one(),
            step_count: rhs.len() * rhs.len().ilog2() as usize + lhs as usize,
            child: Box::new(rhs),
            exec: ExecFn::Constant(Box::new(move |rhs| {
                let new_size = lhs as usize;
                if rhs.len() < new_size {
                    return Err(format!(
                        "Unexpected array length on right hand side; expected length of {}",
                        lhs
                    ));
                }
                let mut new = rhs.clone();
                new.voracious_sort();
                while new.len() != new_size {
                    new.pop();
                }
                Ok(ExecOutput::Array(new))
            })),
            to_string: Box::new(move |rhs| format!("{}Z{}", lhs, vi_to_string(rhs))),
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
        }
    }

    fn cho(lhs: u64, rhs: Self) -> Self {
        let rhs_len = rhs.len();
        Self::UnaryArray {
            name: "Cho",
            size: lhs as usize,
            search_space_size: BigUint::from_usize(rhs.len()).unwrap().pow(lhs as u32),
            step_count: lhs as usize,
            operation: format!("{}C(...)", lhs),
            child: Box::new(rhs),
            exec: if rhs_len == 1 {
                ExecFn::Constant(Box::new(move |rhs| {
                    let new_size = lhs as usize;
                    Ok(ExecOutput::Array(vec![rhs[0]; new_size]))
                }))
            } else {
                ExecFn::Random(
                    lhs as usize,
                    Box::new(move |rhs, rng| {
                        let new_size = lhs as usize;
                        let mut r: Vec<i64> = vec![];
                        while r.len() != new_size {
                            let i = rng.gen_range(0..rhs.len());
                            r.push(rhs[i]);
                        }
                        Ok(ExecOutput::Array(r))
                    }),
                )
            },
            to_string: Box::new(move |rhs| format!("{}C{}", lhs, vi_to_string(rhs))),
            min_max: Box::new(move |rhs| {
                let min = rhs.iter().map(|o| o.min).min().unwrap();
                let max = rhs.iter().map(|o| o.max).max().unwrap();
                let new_size = lhs as usize;
                Ok(MinMaxOutput::Array(vec![MinMax { min, max }; new_size]))
            }),
        }
    }

    fn pic(lhs: u64, rhs: Self) -> Self {
        let k = lhs as usize;
        let n = rhs.len();
        const THRESHOLD: f64 = 0.75;
        let use_roll = (k as f64) < (THRESHOLD * n as f64);
        let step_count = if n == 1 {
            1
        } else if use_roll {
            n * n.ilog2() as usize
        } else {
            n + k
        };

        Self::UnaryArray {
            name: "Pic",
            operation: format!("{}P(...)", lhs),
            size: lhs as usize,
            search_space_size: factorial(n as u64) / factorial(n as u64 - lhs),
            step_count,
            child: Box::new(rhs),
            exec: ExecFn::Random(
                lhs as usize,
                Box::new(move |rhs, rng| {
                    if rhs.len() < k {
                        return Err(format!(
                            "Unexpected array length on right hand side; expected length of {}",
                            lhs
                        ));
                    }
                    let v = if use_roll {
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
            ),
            to_string: Box::new(move |rhs| format!("{}P{}", lhs, vi_to_string(rhs))),
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

                Ok(MinMaxOutput::Permutation(
                    min_values
                        .iter()
                        .zip(max_values.iter())
                        .take(new_size)
                        .map(|m| MinMax {
                            min: *m.0,
                            max: *m.1,
                        })
                        .collect::<_>(),
                    (0..new_size as i64).collect(),
                ))
            }),
        }
    }

    fn die(lhs: u64, rhs: u64) -> Self {
        Self::Generator {
            name: "Die",
            size: lhs as usize,
            search_space_size: BigUint::from_u64(rhs).unwrap().pow(lhs as u32),
            step_count: lhs as usize,
            exec: ExecFn::Random(
                lhs as usize,
                Box::new(move |rng| {
                    let new_size = lhs as usize;
                    let mut r: Vec<i64> = vec![];
                    while r.len() != new_size {
                        let x = rng.gen_range(1..=rhs);
                        r.push(x as i64);
                    }
                    Ok(ExecOutput::Array(r))
                }),
            ),
            to_string: Box::new(move || {
                format!(
                    "{}D{}",
                    if lhs == 1 { "".into() } else { lhs.to_string() },
                    rhs
                )
            }),
            min_max: Box::new(move || {
                let min = 1;
                let max = rhs as i64;
                let new_size = lhs as usize;
                Ok(MinMaxOutput::Array(vec![MinMax { min, max }; new_size]))
            }),
        }
    }

    fn neg(rhs: Self) -> Self {
        Self::UnaryNumber {
            name: "Neg",
            operation: "-(..)".into(),
            size: 1,
            child: Box::new(rhs),
            exec: ExecFn::Constant(Box::new(move |rhs| Ok(ExecOutput::Value(-rhs)))),
            to_string: Box::new(move |rhs| format!("-({})", rhs)),
            min_max: Box::new(move |rhs| {
                let min = -rhs.max;
                let max = -rhs.min;
                Ok(MinMaxOutput::Value(MinMax { min, max }))
            }),
        }
    }

    fn rge(lhs: i64, rhs: i64) -> Self {
        let output = range_to_vi(lhs, rhs);

        let min_max_arr = output
            .iter()
            .cloned()
            .map(|i| MinMax { min: i, max: i })
            .collect::<Vec<_>>();

        Self::Generator {
            name: "Rge",
            size: output.len(),
            search_space_size: BigUint::one(),
            step_count: output.len(),
            exec: ExecFn::Constant(Box::new(move || Ok(ExecOutput::Array(output.clone())))),
            to_string: Box::new(move || format!("({lhs}..{rhs})")),
            min_max: Box::new(move || Ok(MinMaxOutput::Array(min_max_arr.clone()))),
        }
    }

    fn len(&self) -> usize {
        match &self {
            GrammarRule::Aggregate { children, .. } => children.len(),
            GrammarRule::Number { .. } => 1,
            GrammarRule::Select { rhs, .. } => rhs.len(),
            GrammarRule::Generator { size, .. } => *size,
            GrammarRule::UnaryNumber { size, .. } => *size,
            GrammarRule::UnaryArray { size, .. } => *size,
            GrammarRule::Binary { size, .. } => *size,
        }
    }

    fn variable_count(&self) -> usize {
        match &self {
            GrammarRule::Aggregate { children, .. } => {
                children.iter().map(|g| g.variable_count()).sum()
            }
            GrammarRule::Number { .. } => 0,
            GrammarRule::Select { lhs, .. } => lhs.variable_count(),
            GrammarRule::Generator { exec, .. } => exec.variable_count(),
            GrammarRule::UnaryArray { child, exec, .. } => {
                child.variable_count() + exec.variable_count()
            }
            GrammarRule::Binary { rhs, lhs, exec, .. } => {
                lhs.variable_count() + rhs.variable_count() + exec.variable_count()
            }
            GrammarRule::UnaryNumber { child, exec, .. } => {
                child.variable_count() + exec.variable_count()
            }
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
            GrammarRule::Select { lhs, rhs, .. } => Ok(lhs.min_max()?.select(rhs)),
            GrammarRule::Generator { min_max, .. } => min_max(),
            GrammarRule::UnaryNumber { child, min_max, .. } => {
                min_max(child.min_max().map(|o| o.value())?)
            }
            GrammarRule::UnaryArray { child, min_max, .. } => {
                min_max(&child.min_max().and_then(|o| o.expect_array())?)
            }
            GrammarRule::Binary {
                lhs, rhs, min_max, ..
            } => min_max(
                lhs.min_max().map(|o| o.value())?,
                rhs.min_max().map(|o| o.value())?,
            ),
        }
    }

    fn consteval_indices(&self, indices: &[i64]) -> Option<Vec<i64>> {
        match self {
            GrammarRule::Aggregate { children, .. } => {
                let v = children
                    .iter()
                    .map(|c| c.consteval().map(|o| o.value()))
                    .collect::<Vec<_>>();
                let v: Vec<i64> = select_indices(&v, indices).into_iter().flatten().collect();
                (v.len() == indices.len()).then_some(v)
            }
            GrammarRule::Select { lhs, rhs, .. } => {
                lhs.consteval_indices(&select_indices(rhs, indices))
            }
            _ => Some(select_indices(&self.consteval()?.array(), indices)),
        }
    }

    fn flatten(&mut self, flattened_functions: &mut Vec<FlattenedFunction>) {
        match self {
            GrammarRule::Aggregate { children, .. } => children
                .iter_mut()
                .for_each(|c| c.flatten(flattened_functions)),
            GrammarRule::UnaryArray {
                child,
                size,
                operation,
                ..
            } => {
                child.flatten(flattened_functions);
                if child.len() == 1 && *size == 1 {
                    flattened_functions.push(FlattenedFunction {
                        operation: operation.to_string(),
                    });
                    *self = mem::take(child)
                }
            }
            GrammarRule::Select { lhs, .. } => lhs.flatten(flattened_functions),
            GrammarRule::UnaryNumber { child, .. } => child.flatten(flattened_functions),
            GrammarRule::Binary { lhs, rhs, .. } => {
                lhs.flatten(flattened_functions);
                rhs.flatten(flattened_functions);
            }
            _ => (),
        };
    }

    fn consteval(&self) -> Option<ExecOutput> {
        match self {
            GrammarRule::Aggregate { children, .. } => {
                let v = children
                    .iter()
                    .map_while(|c| c.consteval().map(|o| o.value()))
                    .collect::<Vec<_>>();
                (v.len() == children.len()).then_some(ExecOutput::Array(v))
            }
            GrammarRule::Number { number, .. } => Some(ExecOutput::Value(*number)),
            GrammarRule::Select { lhs, rhs, .. } => {
                Some(ExecOutput::Array(lhs.consteval_indices(rhs)?))
            }
            GrammarRule::Generator { exec, .. } => match exec {
                ExecFn::Constant(exec) => exec().ok(),
                _ => None,
            },
            GrammarRule::UnaryNumber { child, exec, .. } => match exec {
                ExecFn::Constant(exec) => exec(child.consteval()?.value()).ok(),
                _ => None,
            },
            GrammarRule::UnaryArray {
                child, exec, size, ..
            } => match exec {
                ExecFn::Constant(exec) => exec(&child.consteval()?.expect_array().ok()?).ok(),
                _ => {
                    let output = child.consteval()?.expect_array().ok()?;
                    Some(ExecOutput::Array(vec![
                        output
                            .windows(2)
                            .all(|w| w[0] == w[1])
                            .then(|| output[0])?;
                        *size
                    ]))
                }
            },
            GrammarRule::Binary { lhs, rhs, exec, .. } => match exec {
                ExecFn::Constant(exec) => {
                    exec(lhs.consteval()?.value(), rhs.consteval()?.value()).ok()
                }
                _ => None,
            },
        }
    }

    fn compile(
        self,
        callstack: &mut Vec<StackFn>,
        compiled_constants: &mut Vec<CompiledConstant>,
        search_space: &mut BigUint,
    ) -> Either<usize, ExecOutput> {
        if let Some(output) = self.consteval() {
            match self {
                GrammarRule::Number { .. } => (),
                GrammarRule::Generator { .. } => (),
                GrammarRule::Aggregate { .. } => (),
                _ => {
                    compiled_constants.push(CompiledConstant {
                        operation: self.to_string(),
                        output: output.to_string(),
                    });
                }
            }
            return Either::Right(output);
        };

        let f = match self {
            GrammarRule::Aggregate { name, children } => {
                let step_count = children.len();
                let indices = children
                    .into_iter()
                    .map(|child| child.compile(callstack, compiled_constants, search_space))
                    .collect::<Vec<_>>();
                StackFn {
                    name,
                    operation: "Aggregate".into(),
                    step_count,
                    output_size: 1,
                    exec: Arc::new(move |stack: &Vec<ExecOutput>, _, _| {
                        Ok(ExecOutput::Array(
                            indices
                                .iter()
                                .map(|e| {
                                    e.as_ref().right_or_else(|i| stack.get(*i).unwrap()).value()
                                })
                                .collect(),
                        )
                        .with_line_detail(name, None))
                    }),
                }
            }
            GrammarRule::Number { name, number } => StackFn {
                name,
                operation: format!("{}", number),
                step_count: 1,
                output_size: 1,
                exec: Arc::new(move |_: &Vec<ExecOutput>, _, _| {
                    Ok(ExecOutput::Value(number).with_line_detail(name, None))
                }),
            },
            GrammarRule::Generator {
                name,
                exec,
                to_string,
                search_space_size,
                step_count,
                size,
                ..
            } => {
                *search_space *= search_space_size;
                match exec {
                    ExecFn::Random(_, exec) => StackFn {
                        name,
                        operation: (to_string)(),
                        step_count,
                        output_size: size,
                        exec: Arc::new(move |_: &Vec<ExecOutput>, rng, options| {
                            Ok(exec(rng)?.with_line_detail(
                                name,
                                options.include_line_details.then(&to_string),
                            ))
                        }),
                    },
                    ExecFn::Constant(exec) => StackFn {
                        name,
                        operation: (to_string)(),
                        step_count,
                        output_size: size,
                        exec: Arc::new(move |_: &Vec<ExecOutput>, _, options| {
                            Ok(exec()?.with_line_detail(
                                name,
                                options.include_line_details.then(&to_string),
                            ))
                        }),
                    },
                }
            }
            GrammarRule::UnaryNumber {
                name,
                operation,
                child,
                exec,
                to_string,
                size,
                ..
            } => {
                let either = child.compile(callstack, compiled_constants, search_space);
                match exec {
                    ExecFn::Random(_, exec) => StackFn {
                        name,
                        operation,
                        step_count: 1,
                        output_size: size,
                        exec: Arc::new(move |stack: &Vec<ExecOutput>, rng, options| {
                            let rhs = either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .value();
                            Ok(exec(rhs, rng)?.with_line_detail(
                                name,
                                options.include_line_details.then(|| to_string(rhs)),
                            ))
                        }),
                    },
                    ExecFn::Constant(exec) => StackFn {
                        name,
                        operation,
                        step_count: 1,
                        output_size: size,
                        exec: Arc::new(move |stack: &Vec<ExecOutput>, _, options| {
                            let rhs = either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .value();
                            Ok(exec(rhs)?.with_line_detail(
                                name,
                                options.include_line_details.then(|| to_string(rhs)),
                            ))
                        }),
                    },
                }
            }
            GrammarRule::UnaryArray {
                name,
                operation,
                child,
                exec,
                to_string,
                search_space_size,
                step_count,
                size,
                ..
            } => {
                *search_space *= search_space_size;
                let either = child.compile(callstack, compiled_constants, search_space);
                match exec {
                    ExecFn::Random(_, exec) => StackFn {
                        name,
                        operation,
                        step_count,
                        output_size: size,
                        exec: Arc::new(move |stack: &Vec<ExecOutput>, rng, options| {
                            let v = &either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .expect_array()?;
                            Ok(exec(v, rng)?.with_line_detail(
                                name,
                                options.include_line_details.then(|| to_string(v)),
                            ))
                        }),
                    },
                    ExecFn::Constant(exec) => StackFn {
                        name,
                        operation,
                        step_count,
                        output_size: size,
                        exec: Arc::new(move |stack: &Vec<ExecOutput>, _, options| {
                            let v = &either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .expect_array()?;
                            Ok(exec(v)?.with_line_detail(
                                name,
                                options.include_line_details.then(|| to_string(v)),
                            ))
                        }),
                    },
                }
            }
            GrammarRule::Select { name, lhs, rhs, .. } => {
                let step_count = rhs.len();
                let either = lhs.compile(callstack, compiled_constants, search_space);
                StackFn {
                    name,
                    operation: "(...)|(...)".into(),
                    step_count,
                    output_size: step_count,
                    exec: Arc::new(move |stack: &Vec<ExecOutput>, _, options| {
                        let lhs = &either
                            .as_ref()
                            .right_or_else(|i| stack.get(*i).unwrap())
                            .expect_array()?;
                        let rhs = rhs.clone();
                        Ok(
                            ExecOutput::Array(select_indices(lhs, &rhs)).with_line_detail(
                                name,
                                options.include_line_details.then(|| {
                                    format!("{} | {}", vi_to_string(lhs), vi_to_string(&rhs))
                                }),
                            ),
                        )
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
                size,
                ..
            } => {
                let left_either = lhs.compile(callstack, compiled_constants, search_space);
                let right_either = rhs.compile(callstack, compiled_constants, search_space);
                match exec {
                    ExecFn::Random(_, exec) => StackFn {
                        name,
                        operation,
                        step_count: 1,
                        output_size: size,
                        exec: Arc::new(move |stack: &Vec<ExecOutput>, rng, options| {
                            let lhs = left_either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .value();
                            let rhs = right_either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .value();
                            Ok(exec(lhs, rhs, rng)?.with_line_detail(
                                name,
                                options.include_line_details.then(|| to_string(lhs, rhs)),
                            ))
                        }),
                    },
                    ExecFn::Constant(exec) => StackFn {
                        name,
                        operation,
                        step_count: 1,
                        output_size: size,
                        exec: Arc::new(move |stack: &Vec<ExecOutput>, _, options| {
                            let lhs = left_either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .value();
                            let rhs = right_either
                                .as_ref()
                                .right_or_else(|i| stack.get(*i).unwrap())
                                .value();
                            Ok(exec(lhs, rhs)?.with_line_detail(
                                name,
                                options.include_line_details.then(|| to_string(lhs, rhs)),
                            ))
                        }),
                    },
                }
            }
        };

        let n = callstack.len();
        callstack.push(f);
        Either::Left(n)
    }
}

impl Default for GrammarRule {
    fn default() -> Self {
        Self::num(0)
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
                    "Incompatible array length; {:?} expects length of {} but received {}",
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
 * Suffix
 *    = Integer / Range
 */
fn suffix(tokenizer: &mut Tokenizer) -> Result<Vec<i64>, GrammarError> {
    match tokenizer.peek() {
        Some(Token::Number(n)) => {
            tokenizer.next();
            Ok(vec![n as i64])
        }
        Some(Token::Minus) => {
            tokenizer.next();
            match tokenizer.peek() {
                Some(Token::Number(n)) => {
                    tokenizer.next();
                    Ok(vec![-(n as i64)])
                }
                Some(token) => Err(GrammarError {
                    error_type: GrammarErrorType::UnexpectedToken {
                        token,
                        expected: "Number",
                    },
                    error_index: tokenizer.expended_count(),
                    error_length: tokenizer.peek_token_count(),
                    input_string: tokenizer.input_str().into(),
                }),
                None => Err(GrammarError {
                    error_type: GrammarErrorType::UnexpectedEnd { expected: "Number" },
                    error_index: tokenizer.expended_count(),
                    error_length: 1,
                    input_string: tokenizer.input_str().into(),
                }),
            }
        }
        Some(Token::Range(lhs, rhs)) => {
            tokenizer.next();
            Ok(range_to_vi(lhs, rhs))
        }
        Some(token) => Err(GrammarError {
            error_type: GrammarErrorType::UnexpectedToken {
                token,
                expected: "Number or Range",
            },
            error_index: tokenizer.expended_count(),
            error_length: tokenizer.peek_token_count(),
            input_string: tokenizer.input_str().into(),
        }),
        None => Err(GrammarError {
            error_type: GrammarErrorType::UnexpectedEnd {
                expected: "Number or Range",
            },
            error_index: tokenizer.expended_count(),
            error_length: 1,
            input_string: tokenizer.input_str().into(),
        }),
    }
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
                if let Some(Token::Range(lhs, rhs)) = tokenizer.peek() {
                    tokenizer.next();
                    arr.append(
                        &mut range_to_vi(lhs, rhs)
                            .into_iter()
                            .map(|i| {
                                if i < 0 {
                                    GrammarRule::neg(GrammarRule::num(i.unsigned_abs()))
                                } else {
                                    GrammarRule::num(i as u64)
                                }
                            })
                            .collect(),
                    );
                } else {
                    arr.push(expression(tokenizer)?);
                }
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
        Some(Token::LeftSquareBracket) => {
            tokenizer.next();
            let lhs = primary(tokenizer, "Array")?;
            match tokenizer.peek() {
                Some(Token::Pipe) => {
                    tokenizer.next();
                    let mut arr: Vec<i64> = Vec::new();
                    loop {
                        arr.append(&mut suffix(tokenizer)?);
                        match tokenizer.peek() {
                            Some(Token::RightSquareBracket) => {
                                tokenizer.next();
                                break;
                            }
                            Some(Token::Comma) => tokenizer.next(),
                            Some(token) => {
                                return Err(GrammarError {
                                    error_type: GrammarErrorType::UnexpectedToken {
                                        token,
                                        expected: "',' or ']'",
                                    },
                                    error_index: tokenizer.expended_count(),
                                    error_length: tokenizer.peek_token_count(),
                                    input_string: tokenizer.input_str().into(),
                                });
                            }
                            None => {
                                return Err(GrammarError {
                                    error_type: GrammarErrorType::UnexpectedEnd {
                                        expected: "',' or ']'",
                                    },
                                    error_index: tokenizer.expended_count(),
                                    error_length: 1,
                                    input_string: tokenizer.input_str().into(),
                                });
                            }
                        };
                    }
                    Ok(GrammarRule::sel(lhs, arr))
                }
                Some(token) => Err(GrammarError {
                    error_type: GrammarErrorType::UnexpectedToken {
                        token,
                        expected: "'|'",
                    },
                    error_index: tokenizer.expended_count(),
                    error_length: tokenizer.peek_token_count(),
                    input_string: tokenizer.input_str().into(),
                }),
                None => Err(GrammarError {
                    error_type: GrammarErrorType::UnexpectedEnd { expected: "'|'" },
                    error_index: tokenizer.expended_count(),
                    error_length: 1,
                    input_string: tokenizer.input_str().into(),
                }),
            }
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

#[derive(Debug, Clone)]
pub(crate) struct Grammar {
    compiled_string: String,
    callstack: Vec<StackFn>,
    compiled_constants: Vec<CompiledConstant>,
    flattened_functions: Vec<FlattenedFunction>,
    search_space: BigUint,
    variable_count: usize,
    min_max: MinMax,
    runtime_value: Either<usize, ExecOutput>,
}

impl Grammar {
    pub(crate) fn parse(input: &str) -> Result<Self, GrammarError> {
        let mut tokenizer = Tokenizer::new(input);
        let mut result = expression(&mut tokenizer)?;

        if tokenizer.expended_count() < input.len() {
            return Err(GrammarError {
                error_type: GrammarErrorType::RedundantTokensAfterExpression,
                error_index: tokenizer.expended_count(),
                error_length: input.len() - tokenizer.expended_count(),
                input_string: input.into(),
            });
        }
        let mut callstack: Vec<StackFn> = vec![];
        let mut search_space = BigUint::one();
        let mut compiled_constants: Vec<CompiledConstant> = vec![];
        let mut flattened_functions: Vec<FlattenedFunction> = vec![];

        result.flatten(&mut flattened_functions);

        let compiled_string = format!("{:#}", result);
        let min_max = result.min_max().unwrap().value();
        let variable_count = result.variable_count();
        let runtime_value =
            result.compile(&mut callstack, &mut compiled_constants, &mut search_space);

        Ok(Self {
            compiled_string,
            compiled_constants,
            flattened_functions,
            callstack,
            search_space,
            variable_count,
            min_max,
            runtime_value,
        })
    }

    pub(crate) fn min(&self) -> i64 {
        self.min_max.min
    }

    pub(crate) fn max(&self) -> i64 {
        self.min_max.max
    }

    pub(crate) fn variable_count(&self) -> usize {
        self.variable_count
    }

    pub(crate) fn compiled_constants(&self) -> &[CompiledConstant] {
        &self.compiled_constants
    }

    pub(crate) fn flattened_functions(&self) -> &[FlattenedFunction] {
        &self.flattened_functions
    }

    pub(crate) fn consteval(&self) -> Option<ExecOutput> {
        self.runtime_value.as_ref().cloned().right()
    }

    pub(crate) fn exec(
        &self,
        rng: &mut impl rand::Rng,
        options: RollOptions,
    ) -> ExecResultWithDetails {
        let mut stack: Vec<ExecOutput> = vec![];
        let mut details: Vec<ExecLineDetail> = vec![];
        if let Some(output) = self.consteval() {
            return Ok(output.with_details(details));
        }
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
        options: TestOptions,
    ) -> TestResultWithDetails {
        let start_time = Instant::now();
        let mut details: Vec<TestLineDetail> = vec![];
        if let Some(output) = self.consteval() {
            return Ok(TestDetails {
                output: vec![output.value(); options.test_size],
                time_taken: start_time.elapsed(),
                details,
            });
        }
        let exec_options = RollOptions {
            include_line_details: false,
        };

        let mut info = OverallTestInfo {
            operation_test_info: OperationTestInfo {
                code: "",
                name: "",
                operation_index: 0,
                iteration_index: 0,
                step_count: 0,
                operation_output_size: 0,
                test_size: options.test_size,
                start_time,
            },
            operations_count: self.callstack.len(),
            step_count: self.callstack.iter().map(|f| f.step_count).sum(),
            iteration_index: 0,
            start_time,
        };

        let mut stacks: Vec<Vec<ExecOutput>> = vec![vec![]; options.test_size];
        for (j, stack_fn) in self.callstack.iter().enumerate() {
            info.operation_test_info = OperationTestInfo {
                code: stack_fn.name,
                name: &stack_fn.operation,
                operation_index: j,
                iteration_index: 0,
                step_count: stack_fn.step_count,
                operation_output_size: stack_fn.output_size,
                test_size: options.test_size,
                start_time: Instant::now(),
            };

            let stack_fn_start = Instant::now();
            for (i, stack) in stacks.iter_mut().enumerate() {
                info.operation_test_info.iteration_index = i;
                info.iteration_index += 1;
                if let Some(callback) = &options.interval_callback {
                    (callback)(&info);
                }
                let (output, _) = (stack_fn.exec)(stack, rng, &exec_options)?;
                stack.push(output);
            }
            if options.is_debug {
                details.push(TestLineDetail {
                    name: stack_fn.name,
                    operation: stack_fn.operation.clone(),
                    time_taken: stack_fn_start.elapsed(),
                });
            }
        }

        let mut output: Vec<i64> = stacks.into_iter().map(|s| s[s.len() - 1].value()).collect();
        output.voracious_sort();
        Ok(TestDetails {
            output,
            time_taken: start_time.elapsed(),
            details,
        })
    }

    pub(crate) fn test_mt(
        &self,
        rng: &mut (impl rand::Rng + Send + Sync + Clone + 'static),
        options: TestOptions,
    ) -> TestResultWithDetails {
        let start_time = Instant::now();
        let mut details: Vec<TestLineDetail> = vec![];
        if let Some(output) = self.consteval() {
            return Ok(TestDetails {
                output: vec![output.value(); options.test_size],
                time_taken: start_time.elapsed(),
                details,
            });
        }
        let exec_options = RollOptions {
            include_line_details: false,
        };

        let mut info = OverallTestInfo {
            operation_test_info: OperationTestInfo {
                code: "",
                name: "",
                operation_index: 0,
                iteration_index: 0,
                step_count: 0,
                operation_output_size: 0,
                test_size: options.test_size,
                start_time,
            },
            operations_count: self.callstack.len(),
            step_count: self.callstack.iter().map(|f| f.step_count).sum(),
            iteration_index: 0,
            start_time,
        };

        struct LineTaskCompleted {
            // thread_index: usize,
            operation_index: usize,
            // iteration_index: usize,
        }

        let (sx, rx) = mpsc::channel::<LineTaskCompleted>();

        const THREAD_COUNT: usize = 10;
        let threads = (0..THREAD_COUNT)
            .map(|_k| {
                let test_size = options.test_size / THREAD_COUNT;
                let sender = sx.clone();
                let callstack = self.callstack.clone();
                let exec_options = exec_options.clone();
                let mut rng = rng.clone();
                thread::spawn(move || -> Result<Vec<i64>, String> {
                    let mut stacks: Vec<Vec<ExecOutput>> = vec![vec![]; test_size];
                    for (j, stack_fn) in callstack.iter().enumerate() {
                        for stack in stacks.iter_mut() {
                            let (output, _) = (stack_fn.exec)(stack, &mut rng, &exec_options)?;
                            stack.push(output);
                            sender
                                .send(LineTaskCompleted {
                                    operation_index: j,
                                })
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    let mut output = stacks
                        .into_iter()
                        .map(|mut s| s.pop().unwrap().value())
                        .collect::<Vec<_>>();
                    output.voracious_sort();
                    Ok(output)
                })
            })
            .collect::<Vec<_>>();

        let mut function_completion = vec![0; self.callstack.len()];
        let mut function_start_time = vec![Instant::now(); self.callstack.len()];
        loop {
            let LineTaskCompleted {
                operation_index, ..
            } = rx.recv().map_err(|e| e.to_string())?;
            let is_start_operation = function_completion[operation_index] == 0;
            if is_start_operation {
                function_start_time[operation_index] = Instant::now();
            }

            if let Some(callback) = &options.interval_callback {
                info.operation_test_info = OperationTestInfo {
                    code: self.callstack[operation_index].name,
                    name: &self.callstack[operation_index].operation,
                    operation_index,
                    iteration_index: function_completion[operation_index],
                    step_count: self.callstack[operation_index].step_count,
                    operation_output_size: self.callstack[operation_index].output_size,
                    test_size: options.test_size,
                    start_time: function_start_time[operation_index],
                };
                info.iteration_index += 1;
                (callback)(&info);
            }

            function_completion[operation_index] += 1;
            let is_end_operation = function_completion[operation_index] == options.test_size;
            let is_complete = *function_completion.last().unwrap() == options.test_size;

            if is_end_operation && options.is_debug {
                details.push(TestLineDetail {
                    name: self.callstack[operation_index].name,
                    operation: self.callstack[operation_index].operation.clone(),
                    time_taken: function_start_time[operation_index].elapsed(),
                });
            }

            if is_complete {
                break;
            }
        }

        let astart_time = Instant::now();
        let output = kmerge(
            threads
                .into_iter()
                .map(|h| h.join().expect("Thread failed to join"))
                .process_results(|i| i.collect::<Vec<_>>())?,
        )
        .collect::<Vec<_>>();

        println!(
            "{:>12} Time taken: {:?}ms",
            "Notice".bold().bright_green(),
            astart_time.elapsed().as_millis()
        );

        Ok(TestDetails {
            output,
            time_taken: start_time.elapsed(),
            details,
        })
    }

    pub(crate) fn search_space(&self) -> &BigUint {
        &self.search_space
    }

    pub(crate) fn step_count(&self) -> usize {
        self.callstack.iter().map(|f| f.step_count).sum()
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

    fn parse(x: &str) -> GrammarRule {
        let mut tokenizer = Tokenizer::new(x);
        expression(&mut tokenizer).unwrap()
    }

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

        let x = "3a0..1";
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
    fn test_inline_range_in_array() {
        let x = "(0, 1..3, 4)";
        let rule = parse(x);
        assert_eq!(rule.min_max().unwrap().array().len(), 5);

        let x = "3p1..3";
        let rule = parse(x);
        assert_eq!(rule.min_max().unwrap().array().len(), 3);
    }

    #[test]
    fn test_inline_range_in_select_suffix() {
        let x = "[(1,2,3)|0, 1..3, -4]";
        let rule = parse(x);
        assert_eq!(rule.min_max().unwrap().array().len(), 5);
    }

    #[test]
    fn test_variable_count() {
        let x = "1 + 2";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.variable_count(), 0);

        let x = "1..10";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.variable_count(), 0);

        let x = "[(1,2,3)|0,2,4]";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.variable_count(), 0);

        let x = "d4";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.variable_count(), 1);

        let x = "10d4";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.variable_count(), 10);

        let x = "10p10d4";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.variable_count(), 20);
    }

    #[test]
    fn test_min_max() {
        let x = "3d20";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 3);
        assert_eq!(grammar.max(), 60);

        let x = "30d200";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 30);
        assert_eq!(grammar.max(), 6000);

        let x = "a3d20";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 20);

        let x = "z3d20";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 20);

        let x = "p1..20";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();
        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 20);

        let x = "[(1,2,3)|2,0,2]";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();

        assert_eq!(grammar.min(), 7);
        assert_eq!(grammar.max(), 7);

        let x = "[p(1,2,3)|0]";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();

        assert_eq!(grammar.min(), 1);
        assert_eq!(grammar.max(), 3);

        let x = "[[3p(1,2,3)|1,2,3]|0,0,0]";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();

        assert_eq!(grammar.min(), 3);
        assert_eq!(grammar.max(), 9);

        let x = "[[3a3p(1,2,3)|1,2,3]|0,0,0]";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();

        assert_eq!(grammar.min(), 6);
        assert_eq!(grammar.max(), 6);

        let x = "[[3c3p(1,2,3)|1,2,3]|0,0,0]";
        let result = Grammar::parse(x);
        let grammar = result.unwrap();

        assert_eq!(grammar.min(), 3);
        assert_eq!(grammar.max(), 9);
    }

    #[test]
    fn test_constant_fn() {
        let x = "3d20";
        let rule = parse(x);
        assert_eq!(rule.variable_count(), 3);
        assert!(rule.consteval().is_none());

        let x = "1 + 2";
        let rule = parse(x);
        assert_eq!(rule.variable_count(), 0);
        assert!(rule.consteval().is_some());
        assert_eq!(rule.consteval().unwrap().value(), 3);

        let x = "[[1..3|1,2,3]|0,0,0] + 1";
        let rule = parse(x);
        assert_eq!(rule.variable_count(), 0);
        assert!(rule.consteval().is_some());
        assert_eq!(rule.consteval().unwrap().value(), 7);

        let x = "3p[[1..3|1,2,3]|0,0,0]";
        let rule = parse(x);
        assert!(rule.consteval().is_some());
        assert_eq!(rule.consteval().unwrap().value(), 6);

        let x = "3a(1,2,3)";
        let rule = parse(x);
        assert!(rule.consteval().is_some());
        assert_eq!(rule.consteval().unwrap().value(), 6);
    }
}
