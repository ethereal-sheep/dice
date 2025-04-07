use crate::inner::token::{Token, Tokenizer};
use crate::{
    ExecDetails, ExecLineDetail, ExecOutput, ExecResult, ExecResultWithDetails, OperationTestInfo,
    OverallTestInfo, RollOptions, TestDetails, TestLineDetail, TestOptions, TestResultWithDetails,
};
use core::{fmt, num};
use owo_colors::OwoColorize;
use rand::{seq::SliceRandom, Rng, RngCore};
use std::collections::{HashMap, HashSet};
use std::{rc::Rc, time::Instant};

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
    Permutation(usize, Vec<MinMax>),
}

type MinMaxResult = Result<MinMaxOutput, String>;

impl MinMaxOutput {
    pub(crate) fn value(&self) -> MinMax {
        match self {
            MinMaxOutput::Value(m) => m.clone(),
            MinMaxOutput::Array(vec) => {
                vec.iter().fold(MinMax { min: 0, max: 0 }, |acc, x| MinMax {
                    min: acc.min + x.min,
                    max: acc.max + x.max,
                })
            }
            MinMaxOutput::Permutation(n, vec) => {
                vec.iter()
                    .take(*n)
                    .fold(MinMax { min: 0, max: 0 }, |acc, x| MinMax {
                        min: acc.min + x.min,
                        max: acc.max + x.max,
                    })
            }
        }
    }

    pub(crate) fn array(&self) -> Result<Vec<MinMax>, String> {
        match self {
            MinMaxOutput::Value(m) => {
                Err(format!("Expected array, but got Value({:?}) instead", m))
            }
            MinMaxOutput::Array(vec) => Ok(vec.clone()),
            MinMaxOutput::Permutation(n, vec) => Ok(vec.iter().cloned().take(*n).collect()),
        }
    }

    pub(crate) fn permutation(&self) -> Option<(usize, Vec<MinMax>)> {
        match self {
            MinMaxOutput::Value(_) => None,
            MinMaxOutput::Array(_) => None,
            MinMaxOutput::Permutation(n, vec) => Some((*n, vec.clone())),
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

macro_rules! possible_values_mod_fn {
    () => {
        Box<dyn Fn(&Self, usize) -> Vec<Vec<usize>>>
    };
}

macro_rules! possible_values_fn {
    () => {
        Box<dyn Fn(&Self) -> Vec<Vec<i64>>>
    };
}

macro_rules! select_min_max_fn {
    () => {
        Box<dyn Fn(&Self, &Vec<MinMax>, usize, bool) -> MinMaxResult>
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
        rhs: Box<GrammarRule>,
    },
    Generator {
        name: &'static str,
        size: usize,
        exec: exec_fn!(),
        min_max: min_max_fn!(),
        to_string: to_string_fn!(),
        possible_values: possible_values_fn!(),
        possible_values_mod: possible_values_mod_fn!(),
    },
    UnaryNumber {
        name: &'static str,
        operation: String,
        size: usize,
        child: Box<GrammarRule>,
        exec: exec_fn!(i64),
        min_max: min_max_fn!(MinMax),
        to_string: to_string_fn!(i64),
        possible_values: possible_values_fn!(),
        possible_values_mod: possible_values_mod_fn!(),
    },
    UnaryArray {
        name: &'static str,
        operation: String,
        size: usize,
        child: Box<GrammarRule>,
        exec: exec_fn!(&Vec<i64>),
        min_max: min_max_fn!(&Vec<MinMax>),
        to_string: to_string_fn!(&Vec<i64>),
        possible_values: possible_values_fn!(),
        possible_values_mod: possible_values_mod_fn!(),
        // select_min_max: select_min_max_fn!(),
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
        possible_values: possible_values_fn!(),
        possible_values_mod: possible_values_mod_fn!(),
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
            GrammarRule::Select { name, lhs, rhs, .. } => {
                write!(f, "{}(", name)?;
                lhs.fmt(f)?;
                write!(f, ", ")?;
                rhs.fmt(f)?;
                write!(f, ")")
            }
        }
    }
}

type ExecStackFn =
    Rc<dyn Fn(&Vec<ExecOutput>, &mut dyn rand::RngCore, &RollOptions) -> ExecResultWithLineDetail>;

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

fn sum_possible_values_mod(possible_values: &Vec<Vec<usize>>, modulo: usize) -> Vec<usize> {
    let mut set = HashSet::new();
    set.insert(0);
    for child in possible_values.iter() {
        let mut new_sums = HashSet::new();
        for value in child.iter().map(|i| i.rem_euclid(modulo)) {
            for s in set.iter() {
                new_sums.insert((s + value).rem_euclid(modulo));
            }
        }
        set = new_sums;
    }
    let mut v: Vec<usize> = set.iter().cloned().collect();
    v.sort();
    v
}

fn sum_possible_values(possible_values: &Vec<Vec<i64>>) -> Vec<i64> {
    let mut set = HashSet::new();
    set.insert(0);
    for child in possible_values.iter() {
        let mut new_sums = HashSet::new();
        for value in child.iter().cloned() {
            for s in set.iter() {
                new_sums.insert(s + value);
            }
        }
        set = new_sums;
    }
    let mut v: Vec<i64> = set.iter().cloned().collect();
    v.sort();
    v
}

fn mul_possible_values_mod(possible_values: &Vec<Vec<usize>>, modulo: usize) -> Vec<usize> {
    let mut set = HashSet::new();
    set.insert(1);
    for child in possible_values.iter() {
        let mut new_muls = HashSet::new();
        for value in child.iter().map(|i| i.rem_euclid(modulo)) {
            for s in set.iter() {
                new_muls.insert((s * value).rem_euclid(modulo));
            }
        }
        set = new_muls;
    }
    let mut v: Vec<usize> = set.iter().cloned().collect();
    v.sort();
    v
}

fn mul_possible_values(possible_values: &Vec<Vec<i64>>) -> Vec<i64> {
    let mut set = HashSet::new();
    set.insert(0);
    for child in possible_values.iter() {
        let mut new_sums = HashSet::new();
        for value in child.iter().cloned() {
            for s in set.iter() {
                new_sums.insert(s * value);
            }
        }
        set = new_sums;
    }
    let mut v: Vec<i64> = set.iter().cloned().collect();
    v.sort();
    v
}

fn merge_possible_values_mod(possible_values: &Vec<Vec<usize>>, modulo: usize) -> Vec<usize> {
    let mut set = HashSet::new();
    for child in possible_values.iter() {
        for value in child.iter().map(|i| i.rem_euclid(modulo)) {
            set.insert(value);
            if set.len() == modulo {
                break;
            }
        }
    }
    let mut v: Vec<usize> = set.iter().cloned().collect();
    v.sort();
    v
}

fn merge_possible_values(possible_values: &Vec<Vec<i64>>) -> Vec<i64> {
    let mut set = HashSet::new();
    for child in possible_values.iter() {
        for value in child.iter().cloned() {
            set.insert(value);
        }
    }
    let mut v: Vec<i64> = set.iter().cloned().collect();
    v.sort();
    v
}

fn possible_values_to_mod(possible_values: &Vec<i64>, modulo: usize) -> Vec<usize> {
    let mut set = HashSet::new();
    for value in possible_values
        .iter()
        .map(|i| i.rem_euclid(modulo as i64) as usize)
    {
        set.insert(value);
        if set.len() == modulo {
            break;
        }
    }
    let mut v: Vec<_> = set.iter().cloned().collect();
    v.sort();
    v
}

fn nested_possible_values_to_mod(
    possible_values: &Vec<Vec<i64>>,
    modulo: usize,
) -> Vec<Vec<usize>> {
    possible_values
        .iter()
        .map(|v| possible_values_to_mod(v, modulo))
        .collect()
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

    fn sel(lhs: Self, rhs: Self) -> Self {
        Self::Select {
            name: "Sel",
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
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
            possible_values: Box::new(move |something| {
                if let Self::Binary { lhs, rhs, .. } = something {
                    return vec![sum_possible_values(&vec![
                        lhs.possible_values(),
                        rhs.possible_values(),
                    ])];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::Binary { lhs, rhs, .. } = something {
                    return vec![sum_possible_values_mod(
                        &vec![
                            lhs.possible_values_mod(modulo),
                            rhs.possible_values_mod(modulo),
                        ],
                        modulo,
                    )];
                }
                vec![]
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
            exec: Box::new(move |lhs, rhs, _| Ok(ExecOutput::Value(lhs - rhs))),
            min_max: Box::new(move |lhs, rhs| {
                Ok(MinMaxOutput::Value(MinMax {
                    min: lhs.min - rhs.max,
                    max: lhs.max - rhs.min,
                }))
            }),
            to_string: Box::new(move |lhs, rhs| format!("{} - {}", lhs, rhs)),
            possible_values: Box::new(move |something| {
                if let Self::Binary { lhs, rhs, .. } = something {
                    return vec![sum_possible_values(&vec![
                        lhs.possible_values(),
                        rhs.possible_values()
                            .into_iter()
                            .map(|i| -i)
                            .rev()
                            .collect(),
                    ])];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::Binary { lhs, rhs, .. } = something {
                    return vec![sum_possible_values_mod(
                        &vec![
                            lhs.possible_values_mod(modulo),
                            rhs.possible_values_mod(modulo)
                                .into_iter()
                                .map(|i| -(i as i64).rem_euclid(modulo as i64) as usize)
                                .rev()
                                .collect(),
                        ],
                        modulo,
                    )];
                }
                vec![]
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
            exec: Box::new(move |lhs, rhs, _| Ok(ExecOutput::Value(lhs * rhs))),
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
            to_string: Box::new(move |lhs, rhs| format!("{} x {}", lhs, rhs)),
            possible_values: Box::new(move |something| {
                if let Self::Binary { lhs, rhs, .. } = something {
                    return vec![mul_possible_values(&vec![
                        lhs.possible_values(),
                        rhs.possible_values(),
                    ])];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::Binary { lhs, rhs, .. } = something {
                    return vec![mul_possible_values_mod(
                        &vec![
                            lhs.possible_values_mod(modulo),
                            rhs.possible_values_mod(modulo),
                        ],
                        modulo,
                    )];
                }
                vec![]
            }),
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
            possible_values: Box::new(move |something| {
                if let Self::UnaryArray { child, .. } = something {
                    let v = child.nested_possible_values();
                    // for each value, find how many values in other dice it has
                    let mut dice: Vec<HashMap<i64, (usize, usize)>> = vec![HashMap::new(); v.len()];
                    for i in 0..v.len() {
                        for lhs_value in v[i].iter().cloned() {
                            for j in (i + 1)..v.len() {
                                for rhs_value in v[j].iter().cloned() {
                                    if lhs_value == rhs_value {
                                        dice[i].entry(lhs_value).or_default().0 += 1;
                                        dice[j].entry(rhs_value).or_default().0 += 1;
                                        dice[i].entry(lhs_value).or_default().1 += 1;
                                        dice[j].entry(rhs_value).or_default().1 += 1;
                                    } else if lhs_value > rhs_value {
                                        dice[i].entry(lhs_value).or_default().1 += 1;
                                        dice[j].entry(rhs_value).or_default().0 += 1;
                                    } else {
                                        dice[i].entry(lhs_value).or_default().0 += 1;
                                        dice[j].entry(rhs_value).or_default().1 += 1;
                                    }
                                }
                            }
                        }
                    }

                    return (0..lhs as usize)
                        .map(|i| {
                            let mut set: HashSet<i64> = HashSet::new();
                            for map in dice.iter() {
                                for (value, (greater_eq, lesser_eq)) in map.iter() {
                                    let left_threshold = i;
                                    let right_threshold = v.len() - i - 1;
                                    if *greater_eq >= left_threshold
                                        && *lesser_eq >= right_threshold
                                    {
                                        set.insert(*value);
                                    }
                                }
                            }
                            let mut v: Vec<i64> = set.iter().cloned().collect();
                            v.sort();
                            v
                        })
                        .collect();
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                nested_possible_values_to_mod(&something.nested_possible_values(), modulo)
            }),
            // select_min_max: Box::new(move |something, lhs, modulo, maintain_order| {
            //     if let Self::UnaryArray { child, .. } = something {
            //         // return child.possible_values_mod(modulo);
            //         // child.
            //     }
            //     Ok(vec![])
            // }),
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
            possible_values: Box::new(move |something| {
                if let Self::UnaryArray { child, .. } = something {
                    let v = child.nested_possible_values();
                    // for each value, find how many values in other dice it has
                    let mut dice: Vec<HashMap<i64, (usize, usize)>> = vec![HashMap::new(); v.len()];
                    for i in 0..v.len() {
                        for lhs_value in v[i].iter().cloned() {
                            for j in (i + 1)..v.len() {
                                for rhs_value in v[j].iter().cloned() {
                                    if lhs_value == rhs_value {
                                        dice[i].entry(lhs_value).or_default().0 += 1;
                                        dice[j].entry(rhs_value).or_default().0 += 1;
                                        dice[i].entry(lhs_value).or_default().1 += 1;
                                        dice[j].entry(rhs_value).or_default().1 += 1;
                                    } else if lhs_value > rhs_value {
                                        dice[i].entry(lhs_value).or_default().0 += 1;
                                        dice[j].entry(rhs_value).or_default().1 += 1;
                                    } else {
                                        dice[i].entry(lhs_value).or_default().1 += 1;
                                        dice[j].entry(rhs_value).or_default().0 += 1;
                                    }
                                }
                            }
                        }
                    }

                    return (0..lhs as usize)
                        .map(|i| {
                            let mut set: HashSet<i64> = HashSet::new();
                            for map in dice.iter() {
                                for (value, (greater_eq, lesser_eq)) in map.iter() {
                                    let left_threshold = i;
                                    let right_threshold = v.len() - i - 1;
                                    if *lesser_eq >= left_threshold
                                        && *greater_eq >= right_threshold
                                    {
                                        set.insert(*value);
                                    }
                                }
                            }
                            let mut v: Vec<i64> = set.iter().cloned().collect();
                            v.sort();
                            v
                        })
                        .collect();
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::UnaryArray { child, .. } = something {
                    let mut v = child.nested_possible_values_mod(modulo);
                    v.sort_by_key(|v| *v.last().unwrap());
                    v.resize(lhs as usize, vec![]);
                    return v;
                }
                vec![]
            }),
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
            possible_values: Box::new(move |something| {
                if let Self::UnaryArray { child, .. } = something {
                    let v = merge_possible_values(&child.nested_possible_values());
                    return vec![v; lhs as usize];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::UnaryArray { child, .. } = something {
                    let v = merge_possible_values_mod(
                        &child.nested_possible_values_mod(modulo),
                        modulo,
                    );
                    return vec![v; lhs as usize];
                }
                vec![]
            }),
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

                Ok(MinMaxOutput::Permutation(
                    new_size,
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
            possible_values: Box::new(move |something| {
                if let Self::UnaryArray { child, .. } = something {
                    let v = merge_possible_values(&child.nested_possible_values());
                    return vec![v; lhs as usize];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::UnaryArray { child, .. } = something {
                    let v = merge_possible_values_mod(
                        &child.nested_possible_values_mod(modulo),
                        modulo,
                    );
                    return vec![v; lhs as usize];
                }
                vec![]
            }),
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
            possible_values: Box::new(move |something| {
                if let Self::Generator { .. } = something {
                    let rhs = rhs as i64;
                    return vec![(1..=rhs).collect(); lhs as usize];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::Generator { .. } = something {
                    let rhs = rhs as usize;
                    let range: Vec<_> = if rhs >= modulo {
                        (0..modulo).collect()
                    } else {
                        (1..=rhs).collect()
                    };
                    return vec![range; lhs as usize];
                }
                vec![]
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
            possible_values: Box::new(move |something| {
                if let Self::UnaryNumber { child, .. } = something {
                    return vec![child
                        .possible_values()
                        .into_iter()
                        .map(|i| -i)
                        .rev()
                        .collect()];
                }
                vec![]
            }),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::UnaryNumber { child, .. } = something {
                    return vec![child
                        .possible_values_mod(modulo)
                        .into_iter()
                        .map(|i| (modulo - i).rem_euclid(modulo))
                        .rev()
                        .collect()];
                }
                vec![]
            }),
        }
    }

    fn rge(lhs: i64, rhs: i64) -> Self {
        let output = if lhs < rhs {
            (lhs..=rhs).collect::<Vec<_>>()
        } else {
            (rhs..=lhs).rev().collect()
        };

        let possible_values = output.clone();
        let possible_values_mod = output.clone();

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
            possible_values: Box::new(move |_| vec![possible_values.clone()]),
            possible_values_mod: Box::new(move |something, modulo: usize| {
                if let Self::Generator { .. } = something {
                    return if possible_values_mod.len() >= modulo {
                        (0..modulo).collect()
                    } else {
                        let mut v: Vec<usize> = possible_values_mod
                            .iter()
                            .map(|i| i.rem_euclid(modulo as i64) as usize)
                            .collect();
                        v.sort();
                        v
                    }
                    .iter()
                    .map(|i| vec![*i])
                    .collect();
                }
                vec![]
            }),
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

    fn possible_values_mod(&self, modulo: usize) -> Vec<usize> {
        merge_possible_values_mod(&self.nested_possible_values_mod(modulo), modulo)
    }

    fn possible_values(&self) -> Vec<i64> {
        merge_possible_values(&self.nested_possible_values())
    }

    fn nested_possible_values_mod(&self, modulo: usize) -> Vec<Vec<usize>> {
        match &self {
            GrammarRule::Aggregate { children, .. } => children
                .iter()
                .map(|child| {
                    sum_possible_values_mod(&child.nested_possible_values_mod(modulo), modulo)
                })
                .collect(),
            GrammarRule::Number { number, .. } => {
                vec![vec![number.rem_euclid(modulo as i64) as usize]]
            }
            GrammarRule::Select { lhs, rhs, .. } => {
                let lhs_nested_possibilities = lhs.nested_possible_values_mod(modulo);
                rhs.nested_possible_values_mod(lhs.len())
                    .iter()
                    .map(|v| {
                        let mut set: HashSet<usize> = HashSet::new();
                        for i in v {
                            for possible in lhs_nested_possibilities[*i].iter() {
                                set.insert(*possible);
                                if set.len() == modulo {
                                    break;
                                }
                            }
                        }
                        let mut v: Vec<usize> = set.iter().cloned().collect();
                        v.sort();
                        v
                    })
                    .collect()
            }
            GrammarRule::Generator {
                possible_values_mod,
                ..
            } => possible_values_mod(self, modulo),

            GrammarRule::UnaryNumber {
                possible_values_mod,
                ..
            } => possible_values_mod(self, modulo),
            GrammarRule::UnaryArray {
                possible_values_mod,
                ..
            } => possible_values_mod(self, modulo),
            GrammarRule::Binary {
                possible_values_mod,
                ..
            } => possible_values_mod(self, modulo),
        }
    }

    fn nested_possible_values(&self) -> Vec<Vec<i64>> {
        match &self {
            GrammarRule::Aggregate { children, .. } => children
                .iter()
                .map(|child| sum_possible_values(&child.nested_possible_values()))
                .collect(),
            GrammarRule::Number { number, .. } => {
                vec![vec![*number]]
            }
            GrammarRule::Select { lhs, rhs, .. } => {
                let lhs_nested_possibilities = lhs.nested_possible_values();
                rhs.nested_possible_values_mod(lhs.len())
                    .iter()
                    .map(|v| {
                        let mut set: HashSet<i64> = HashSet::new();
                        for i in v {
                            for possible in lhs_nested_possibilities[*i].iter() {
                                set.insert(*possible);
                            }
                        }
                        let mut v: Vec<i64> = set.iter().cloned().collect();
                        v.sort();
                        v
                    })
                    .collect()
            }
            GrammarRule::Generator {
                possible_values, ..
            } => possible_values(self),

            GrammarRule::UnaryNumber {
                possible_values, ..
            } => possible_values(self),
            GrammarRule::UnaryArray {
                possible_values, ..
            } => possible_values(self),
            GrammarRule::Binary {
                possible_values, ..
            } => possible_values(self),
        }
    }

    // fn possible(&self, val: usize, modulo: usize) -> bool {
    //     match &self {
    //         GrammarRule::Aggregate { children, .. } => Ok(MinMaxOutput::Array(
    //             children
    //                 .iter()
    //                 .map(|c| c.min_max().map(|o| o.value()))
    //                 .collect::<Result<Vec<MinMax>, String>>()?,
    //         )),
    //         GrammarRule::Number { number, .. } => *number as usize == val,
    //         GrammarRule::Generator { min_max, .. } => min_max(),
    //         GrammarRule::UnaryNumber { child, min_max, .. } => {
    //             min_max(child.min_max().map(|o| o.value())?)
    //         }
    //         GrammarRule::UnaryArray { child, min_max, .. } => {
    //             min_max(&child.min_max().and_then(|o| o.array())?)
    //         }
    //         GrammarRule::ArrayArray {
    //             lhs, rhs, min_max, ..
    //         } => min_max(&lhs.min_max()?, &rhs.min_max()?),
    //         GrammarRule::Binary {
    //             lhs, rhs, min_max, ..
    //         } => min_max(
    //             lhs.min_max().map(|o| o.value())?,
    //             rhs.min_max().map(|o| o.value())?,
    //         ),
    //     }
    // }

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
            GrammarRule::Select { lhs, rhs, .. } => {
                let lhs = lhs.min_max()?.array()?;
                let rhs = rhs.min_max()?.array()?;

                Ok(MinMaxOutput::Array(
                    rhs.iter()
                        .map(|i| MinMax {
                            min: lhs[i.min.rem_euclid(lhs.len() as i64) as usize].min,
                            max: lhs[i.max.rem_euclid(lhs.len() as i64) as usize].max,
                        })
                        .collect(),
                ))
            }
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
                    Ok(exec(rng)?
                        .with_line_detail(name, options.include_line_details.then(&to_string)))
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
                        Ok(exec(rhs, rng)?.with_line_detail(
                            name,
                            options.include_line_details.then(|| to_string(rhs)),
                        ))
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
                        Ok(exec(v, rng)?.with_line_detail(
                            name,
                            options.include_line_details.then(|| to_string(v)),
                        ))
                    }),
                }
            }
            GrammarRule::Select { name, lhs, rhs, .. } => {
                let lhs_index = lhs.dfs(callstack);
                let rhs_index = rhs.dfs(callstack);
                StackFn {
                    name,
                    operation: "(...)|(...)".into(),
                    exec: Rc::new(move |stack: &Vec<ExecOutput>, _, options| {
                        let lhs = &stack.get(lhs_index).unwrap().array()?;
                        let rhs = &stack.get(rhs_index).unwrap().array()?;
                        Ok(ExecOutput::Array(
                            rhs.iter()
                                .map(|i| lhs[i.rem_euclid(lhs.len() as i64) as usize])
                                .collect(),
                        )
                        .with_line_detail(
                            name,
                            options
                                .include_line_details
                                .then(|| format!("{} | {}", vi_to_string(lhs), vi_to_string(rhs))),
                        ))
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
                        Ok(exec(lhs, rhs, rng)?.with_line_detail(
                            name,
                            options.include_line_details.then(|| to_string(lhs, rhs)),
                        ))
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
        Some(Token::LeftSquareBracket) => {
            tokenizer.next();
            let lhs = primary(tokenizer, "Array")?;
            match tokenizer.peek() {
                Some(Token::Pipe) => {
                    tokenizer.next();
                    let rhs = primary(tokenizer, "Array")?;
                    match tokenizer.peek() {
                        Some(Token::RightSquareBracket) => {
                            tokenizer.next();
                            return Ok(GrammarRule::sel(lhs, rhs));
                        }
                        Some(token) => {
                            return Err(GrammarError {
                                error_type: GrammarErrorType::UnexpectedToken {
                                    token,
                                    expected: "']'",
                                },
                                error_index: tokenizer.expended_count(),
                                error_length: tokenizer.peek_token_count(),
                                input_string: tokenizer.input_str().into(),
                            });
                        }
                        None => {
                            return Err(GrammarError {
                                error_type: GrammarErrorType::UnexpectedEnd { expected: "']'" },
                                error_index: tokenizer.expended_count(),
                                error_length: 1,
                                input_string: tokenizer.input_str().into(),
                            });
                        }
                    };
                }
                Some(token) => {
                    return Err(GrammarError {
                        error_type: GrammarErrorType::UnexpectedToken {
                            token,
                            expected: "'|'",
                        },
                        error_index: tokenizer.expended_count(),
                        error_length: tokenizer.peek_token_count(),
                        input_string: tokenizer.input_str().into(),
                    });
                }
                None => {
                    return Err(GrammarError {
                        error_type: GrammarErrorType::UnexpectedEnd { expected: "'|'" },
                        error_index: tokenizer.expended_count(),
                        error_length: 1,
                        input_string: tokenizer.input_str().into(),
                    });
                }
            };
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
        options: RollOptions,
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
        options: TestOptions,
    ) -> TestResultWithDetails {
        let start_time = Instant::now();
        let mut stacks: Vec<Vec<ExecOutput>> = vec![vec![]; options.test_size];
        let mut details: Vec<TestLineDetail> = vec![];
        let exec_options = RollOptions {
            include_line_details: false,
        };

        let mut info = OverallTestInfo {
            operation_test_info: OperationTestInfo {
                code: "",
                name: "",
                operation_index: 0,
                iteration_index: 0,
                test_size: options.test_size,
                start_time: Instant::now(),
            },
            operations_count: self.callstack.len(),
            start_time: Instant::now(),
        };

        for (j, stack_fn) in self.callstack.iter().enumerate() {
            info.operation_test_info = OperationTestInfo {
                code: stack_fn.name,
                name: &stack_fn.operation,
                operation_index: j,
                iteration_index: 0,
                test_size: options.test_size,
                start_time: Instant::now(),
            };

            let stack_fn_start = Instant::now();
            for (i, stack) in stacks.iter_mut().enumerate() {
                info.operation_test_info.iteration_index = i;
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

        // let x = "[(d3, d6, d20)|(d3, d6, d20)]";
        // // let result = Grammar::parse(x);
        // // let grammar = result.unwrap();

        // // assert_eq!(grammar.min(), 1);
        // // assert_eq!(grammar.max(), 20);

        // let mut tokenizer = Tokenizer::new(x);
        // let result = expression(&mut tokenizer).unwrap();
        // let p = result.nested_possible_values_mod(usize::MAX);
        // println!("{:?}", p);
    }

    #[test]
    fn test_nested_possible() {
        let x = "2a(p(-1,3,5), p(-1, 2,4,6), p(-1,3,5,8,7), p(10,1, 0))";
        // let x = "(3d6, 2d4)";
        let mut tokenizer = Tokenizer::new(x);
        let result = expression(&mut tokenizer).unwrap();
        let p = result.nested_possible_values_mod(10);
        println!("{:?}", p);
    }
}
