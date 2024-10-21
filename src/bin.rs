#![feature(assert_matches)]
use core::fmt;
use std::cmp::Ordering;

use clap::{arg, command, value_parser, Command};
use dice::{Dice, RollOptions, TestOptions};
use owo_colors::OwoColorize;
use rand::{rngs::StdRng, SeedableRng};

pub fn main() {
    let matches = command!()
        .subcommand(
            Command::new("roll")
                .about("Parses a dice script and rolls it")
                .arg(arg!(<SCRIPT> "Dice script to parse and roll").allow_hyphen_values(true))
                .arg(arg!(
                    -d --debug "Turn debugging information on"
                ))
                .arg(arg!(
                    --nologo "Suppresses logo in debug mode"
                ))
                .arg(arg!(
                    --raw "Returns the raw value if available"
                )),
        )
        .subcommand(
            Command::new("test")
                .about("Parses a dice script and tests it")
                .arg(arg!(<SCRIPT> "Dice script to parse and test").allow_hyphen_values(true))
                .arg(arg!(
                    -d --debug "Turn debugging information on"
                ))
                .arg(arg!(
                    --nologo "Suppresses logo in debug mode"
                ))
                .arg(
                    arg!(
                        -s --size <N> "Tests the script N number of times;\nmin. 1, max. 1,000,000, default 100,000"
                    )
                    .value_parser(value_parser!(u64).range(1..=1_000_000)),
                )
                .arg(
                    arg!(
                        --buckets <N> "Bucket the test output into N buckets;\nWill be overriden by output range if range < N;\nmin. 1, max. 200"
                    )
                    .value_parser(value_parser!(u64).range(1..=200)),
                )
                .arg(
                    arg!(
                        --height <N> "Sets the height of the output graph;\nmin. 1, max. 100, default 10"
                    )
                    .value_parser(value_parser!(u64).range(1..=200)),
                )
                .arg(
                    arg!(
                        --reference <REF> "Outputs test data in reference to REF"
                    )
                    .value_parser(value_parser!(i64)).allow_hyphen_values(true).allow_negative_numbers(true),
                ),
        )
        .subcommand(
            Command::new("seed")
                .about("Seeds the prevailing random state")
                .arg(arg!(<SEED> "Dice script to parse and roll"))
                .arg(arg!(
                    -d --debug "Turn debugging information on"
                )),
        )
        .get_matches();

    let start_width = 12;
    if let Some(matches) = matches.subcommand_matches("roll") {
        let is_debug = matches.get_flag("debug");
        if is_debug {
            if !matches.get_flag("nologo") {
                println!("{:>start_width$}", Logo {});
            } else {
                println!("{:>start_width$}", SimpleLogo {});
            }
        }

        if let Some(script) = matches.get_one::<String>("SCRIPT") {
            if is_debug {
                println!(
                    "{:>start_width$} {}",
                    "Compiling".bold().bright_cyan(),
                    script
                );
            }
            let result = script.parse::<Dice>();
            match result {
                Ok(dice) => {
                    if is_debug {
                        println!(
                            "{:>start_width$} {}",
                            "Compiled".bold().yellow(),
                            dice.bright_magenta()
                        );
                        println!(
                            "{:>start_width$} {} {}",
                            "Rolling".bold().bright_cyan(),
                            "with",
                            "Std RNG".bold().bright_cyan()
                        );
                    }
                    let result = dice.roll(&mut StdRng::from_entropy(), RollOptions { is_debug });
                    match result {
                        Ok(result) => {
                            if is_debug {
                                println!("{:start_width$}", result);
                            }
                            println!("{}", result.into_value())
                        }
                        Err(err) => println!("{:>start_width$} {}", "Error".red().bold(), err),
                    }
                }
                Err(err) => {
                    if is_debug {
                        println!(
                            "{:>start_width$} {}",
                            "Error".red().bold(),
                            err.formatted_error_string()
                        );
                    } else {
                        println!("{}", err);
                    }
                }
            }
        }
    } else if let Some(matches) = matches.subcommand_matches("test") {
        let is_debug = matches.get_flag("debug");
        if is_debug {
            if !matches.get_flag("nologo") {
                println!("{:>start_width$}", Logo {});
            } else {
                println!("{:>start_width$}", SimpleLogo {});
            }
        }

        if let Some(script) = matches.get_one::<String>("SCRIPT") {
            if is_debug {
                println!(
                    "{:>start_width$} {}",
                    "Compiling".bold().bright_cyan(),
                    script
                );
            }
            let result = script.parse::<Dice>();
            match result {
                Ok(dice) => {
                    let test_size: u64 = if let Some(n) = matches.get_one("size") {
                        *n
                    } else {
                        if is_debug {
                            println!(
                                "{:>start_width$} defaulting test size to {}",
                                "Notice".bold().bright_green(),
                                100000.bold().bright_yellow(),
                            );
                        }
                        100000
                    };

                    let options = TestOptions {
                        is_debug: matches.get_flag("debug"),
                        test_size: test_size as usize,
                    };
                    if is_debug {
                        println!(
                            "{:>start_width$} {}",
                            "Compiled".bold().yellow(),
                            dice.bright_magenta()
                        );
                        println!(
                            "{:>start_width$} with {}",
                            "Testing".bold().bright_cyan(),
                            "Std RNG".bold().bright_yellow()
                        );
                    }
                    let result = dice.test(&mut StdRng::from_entropy(), options);
                    match result {
                        Ok(result) => {
                            if is_debug {
                                eprintln!(
                                    "{:>12} {} data points",
                                    "Bucketing".bright_cyan().bold(),
                                    result.output.len().bright_yellow().bold()
                                );
                            }
                            let buckets = Buckets::from_data(&result.output);
                            buckets.print_graph(
                                start_width,
                                *matches.get_one::<u64>("height").unwrap_or(&10) as usize,
                                matches.get_one("reference").copied(),
                            );
                        }
                        Err(err) => println!("{:>start_width$} {}", "Error".red().bold(), err),
                    }
                }
                Err(err) => {
                    if is_debug {
                        println!(
                            "{:>start_width$} {}",
                            "Error".red().bold(),
                            err.formatted_error_string()
                        );
                    } else {
                        println!("{}", err);
                    }
                }
            }
        }
    }
}

struct Logo;

impl fmt::Display for Logo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(0);

        // let logo_a = r"     _ _      _____  _____  ";
        // let logo_b = r"  __| (_) ___/___ /\/\  . \ ";
        // let logo_c = r" / _` | |/ __/ _ \ /..\____\";
        // let logo_d = r"| (_| | | (_|  __/ \  /'  '/";
        // let logo_e = r" \__,_|_|\___\___|\/\/'__'/ ";

        let logo_a1 = r"     _ _      ".bright_magenta();
        let logo_a2 = r"_____  _____  ";

        let logo_b1 = r"  __| (_) ___".bright_magenta();
        let logo_b2 = r"/";
        let logo_b3 = r"___ ".bright_magenta();
        let logo_b4 = r"/\/\  . \ ";

        let logo_c1 = r" / _` | |/ __/ _ \".bright_magenta();
        let logo_c2 = r" /..\____\";

        let logo_d1 = r"| (_| | | (_|  __/".bright_magenta();
        let logo_d2 = r" \  /'  '/";

        let logo_e1 = r" \__,_|_|\___\___|".bright_magenta();
        let logo_e2 = r"\/\/'__'/ ";

        writeln!(f, "{:>width$}", "".bold().bright_blue())?;
        writeln!(f, "{:width$}{logo_a1}{logo_a2}", "")?;
        writeln!(f, "{:width$}{logo_b1}{logo_b2}{logo_b3}{logo_b4}", "")?;
        writeln!(f, "{:width$}{logo_c1}{logo_c2}", "")?;
        writeln!(f, "{:width$}{logo_d1}{logo_d2}", "")?;
        writeln!(
            f,
            "{:width$}{logo_e1}{logo_e2} v{}",
            "",
            env!("CARGO_PKG_VERSION").bold()
        )
    }
}
struct SimpleLogo;

impl fmt::Display for SimpleLogo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(0);
        write!(
            f,
            "{:>width$} v{}",
            "Dice".bright_blue().bold(),
            env!("CARGO_PKG_VERSION").bold()
        )
    }
}

const VERTICAL_BAR_SECTIONS: &[char] = &['-', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const HORIZONTAL_BAR_SECTIONS: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

struct Buckets {
    buckets: Vec<usize>,
    buckets_start: i64,
    bucket_size: usize,
    min_value: i64,
    max_value: i64,
}

fn get_max_bucket_count() -> usize {
    termsize::get().unwrap().cols as usize - 15
}

fn get_max_width(bucket_count: usize) -> usize {
    for i in (1..=get_max_bucket_count()).rev() {
        if bucket_count * i < get_max_bucket_count() {
            return i;
        }
    }
    1
}

fn choose_bucket_count_size_and_extra_range(value_range: usize) -> (usize, usize, usize) {
    let max_bucket_count = get_max_bucket_count();
    let min_bucket_size = value_range / max_bucket_count + 1;
    for bucket_size in min_bucket_size..=value_range {
        let bucket_count = (value_range - 1) / bucket_size + 1;
        let extra_range = (bucket_count - value_range % bucket_count) % bucket_count;
        if extra_range < bucket_size {
            return (bucket_count, bucket_size, extra_range);
        }
    }
    (1, value_range, 0)
}

fn choose_char_for_bucket_at_interval(
    interval_start: f64,
    interval_size: f64,
    bucket_value: f64,
) -> char {
    let normalized_position = ((bucket_value - interval_start) / interval_size).clamp(0.0, 1.0);
    let index = (normalized_position * 8.0).floor() as usize;
    VERTICAL_BAR_SECTIONS[index]
}

fn printed_width(number: i64) -> usize {
    if number < 0 {
        return printed_width(number.abs()) + 1;
    }
    if number == 0 {
        return 1;
    }
    let mut count = 0;
    let mut current = number;
    while current != 0 {
        count += 1;
        current /= 10;
    }
    count
}

struct BucketLine {
    chars: Vec<char>,
    bar_width: usize,
    reference_index: Option<usize>,
}

impl BucketLine {
    fn new(
        bar_width: usize,
        interval_start: f64,
        interval_size: f64,
        reference_index: Option<usize>,
        it: impl Iterator<Item = usize>,
    ) -> Self {
        Self {
            chars: it
                .map(|value| {
                    choose_char_for_bucket_at_interval(interval_start, interval_size, value as f64)
                })
                .collect(),
            bar_width,
            reference_index,
        }
    }
    fn base(bar_width: usize, buckets_size: usize, reference_index: Option<usize>) -> Self {
        Self {
            chars: vec!['▔'; buckets_size],
            bar_width,
            reference_index,
        }
    }
}

impl fmt::Display for BucketLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, c) in self.chars.iter().enumerate() {
            for _ in 0..self.bar_width {
                if *c == VERTICAL_BAR_SECTIONS[0] {
                    write!(f, "{}", c.bright_black())?;
                } else if self.reference_index == Some(i) {
                    write!(f, "{}", c.bright_magenta())?;
                } else {
                    write!(f, "{}", c)?;
                }
            }
        }
        write!(f, "")
    }
}

struct Bucket {
    start: i64,
    size: usize,
}

impl fmt::Display for Bucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.size == 1 {
            write!(f, "{}", self.start)
        } else if self.size == 2 {
            write!(f, "{}, {}", self.start, self.start + self.size as i64 - 1)
        } else {
            write!(f, "{}..={}", self.start, self.start + self.size as i64 - 1)
        }
    }
}

struct XAxis<'a> {
    bar_width: usize,
    buckets: &'a Buckets,
}

impl fmt::Display for XAxis<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bar_width = self.bar_width;
        if self.buckets.bucket_size == 1 && bar_width > printed_width(self.buckets.max_value) {
            for i in 0..self.buckets.buckets.len() {
                if i % 1 == 0 {
                    write!(f, "{:^bar_width$}", self.buckets.buckets_start + i as i64)?;
                } else {
                    write!(f, "{:^bar_width$}", "")?;
                }
            }
            return write!(f, "");
        }

        let graph_width = self.buckets.get_graph_width(bar_width);

        let front_length = printed_width(self.buckets.min_value);
        let back_length = printed_width(self.buckets.max_value);

        let mid_value =
            self.buckets.min_value + (self.buckets.max_value - self.buckets.min_value) / 2;
        let mid_length = printed_width(mid_value);
        let quartile_value =
            self.buckets.min_value + (self.buckets.max_value - self.buckets.min_value) / 4;
        let quartile_length = printed_width(quartile_value);
        let triquartile_value = mid_value + (mid_value - quartile_value);
        let triquartile_length = printed_width(triquartile_value);

        let mid_index = self.buckets.which(mid_value).unwrap() * bar_width + (bar_width - 1) / 2;
        let quartile_index =
            self.buckets.which(quartile_value).unwrap() * bar_width + (bar_width - 1) / 2;
        let triquatile_index =
            self.buckets.which(triquartile_value).unwrap() * bar_width + (bar_width - 1) / 2;

        let mid_start_position = mid_index - (mid_length - 1) / 2;
        let quartile_start_position = quartile_index - (quartile_length - 1) / 2;
        let triquartile_start_position = triquatile_index - (triquartile_length - 1) / 2;

        // check if overlap range or range too small
        // for 5 numbers
        if quartile_start_position > front_length
            && mid_start_position > quartile_start_position + quartile_length
            && triquartile_start_position > mid_start_position + mid_length
            && graph_width > back_length + triquartile_start_position + triquartile_length
        {
            let first_padding = quartile_start_position - front_length;
            let second_padding = mid_start_position - quartile_start_position - quartile_length;
            let third_padding = triquartile_start_position - mid_start_position - mid_length;
            let fourth_padding =
                graph_width - back_length - triquartile_start_position - triquartile_length;

            return write!(
                f,
                "{}{:first_padding$}{}{:second_padding$}{}{:third_padding$}{}{:fourth_padding$}{}",
                self.buckets.min_value,
                "",
                quartile_value,
                "",
                mid_value,
                "",
                triquartile_value,
                "",
                self.buckets.max_value
            );
        }

        // for 3 numbers
        if mid_start_position > front_length
            && graph_width > back_length + mid_start_position + mid_length
        {
            let first_padding = mid_start_position - front_length;
            let second_padding = graph_width - back_length - mid_start_position - mid_length;

            return write!(
                f,
                "{}{:first_padding$}{}{:second_padding$}{}",
                self.buckets.min_value, "", mid_value, "", self.buckets.max_value
            );
        }

        // for 2 numbers
        if graph_width > back_length + front_length {
            let padding = graph_width - back_length;

            return write!(
                f,
                "{}{:padding$}{}",
                self.buckets.min_value, "", self.buckets.max_value
            );
        }

        write!(f, "")
    }
}

impl Buckets {
    fn from_data(data: &Vec<i64>) -> Self {
        let min_value = *data.iter().min().unwrap_or(&0);
        let max_value = *data.iter().max().unwrap_or(&100);
        let value_range = (max_value - min_value + 1) as usize;
        let (bucket_count, bucket_size, _extra_range) =
            choose_bucket_count_size_and_extra_range(value_range);
        let bucket_min = min_value;

        let mut buckets: Vec<usize> = vec![0; bucket_count];
        let mut mode_index = 0;
        for num in data {
            let index = (*num - bucket_min) as usize / bucket_size;
            buckets[index] += 1;
            if buckets[index] > buckets[mode_index] {
                mode_index = index;
            }
        }

        Self {
            buckets,
            buckets_start: bucket_min,
            bucket_size,
            min_value,
            max_value,
        }
    }

    fn range(&self) -> usize {
        self.buckets.len() * self.bucket_size
    }

    fn which(&self, value: i64) -> Option<usize> {
        if value < self.min_value || value > self.max_value {
            return None;
        }
        let normalized_value = value - self.buckets_start;
        if normalized_value < 0 || normalized_value as usize >= self.range() {
            return None;
        }
        Some(normalized_value as usize / self.bucket_size)
    }

    fn get_bucket(&self, index: usize) -> Option<Bucket> {
        if index >= self.buckets.len() {
            return None;
        }
        Some(Bucket {
            start: self.buckets_start + (index * self.bucket_size) as i64,
            size: self.bucket_size,
        })
    }

    fn get_graph_width(&self, bar_width: usize) -> usize {
        self.buckets.len() * bar_width
    }

    fn get_x_axis(&self, bar_width: usize) -> XAxis {
        XAxis {
            buckets: self,
            bar_width,
        }
    }

    fn print_graph(&self, front_padding: usize, graph_height: usize, reference: Option<i64>) {
        let bar_width = get_max_width(self.buckets.len());
        let reference_index = reference.map(|i| self.which(i)).flatten();
        let interval_count = graph_height.clamp(1, 200);
        let graph_min = 0;
        let graph_max = {
            let max_bucket = *self.buckets.iter().max().unwrap_or(&100);
            max_bucket
        };
        let interval_size = (graph_max - graph_min) as f64 / interval_count as f64;

        let graph_width = self.get_graph_width(bar_width);
        let top_line = (interval_count as f64 + 0.5) * interval_size;
        eprintln!(
            "{:>front_padding$} histogram of {} buckets of size {}; occupying {} ({} + 4) rows and {} columns (bar width {})",
            "Drawing".bright_cyan().bold(),
            self.buckets.len().bright_yellow().bold(),
            self.bucket_size.bright_yellow().bold(),
            (graph_height + 4).bright_yellow().bold(),
            graph_height,
            graph_width.bright_yellow().bold(),
            bar_width.bright_yellow().bold(),
        );
        if let Some(reference_value) = reference {
            if let Some(index) = reference_index {
                let bucket = self.get_bucket(index).unwrap();
                let bucket_count = self.buckets[index];
                eprintln!(
                    "{:>front_padding$} value of {} belongs in bucket {} ({}) with size of {}",
                    "Reference".bright_magenta().bold(),
                    reference_value.bright_yellow().bold(),
                    index.bright_yellow().bold(),
                    bucket,
                    bucket_count.bright_yellow().bold()
                );
            } else {
                eprintln!(
                    "{:>front_padding$} value of {} is outside graph range",
                    "Reference".bright_magenta().bold(),
                    reference_value.bright_yellow().bold(),
                );
            }
        }
        eprintln!(
            "{:>front_padding$} {}",
            "Y-Axis".bold().bright_blue(),
            format!("~{:.1}/step", interval_size).italic()
        );
        eprintln!(
            "{:>front_padding$.0} {:-<graph_width$}",
            top_line.bold().bright_yellow(),
            "".bright_black()
        );
        for i in (0..interval_count).rev() {
            let interval_start = i as f64 * interval_size;
            let line = BucketLine::new(
                bar_width,
                interval_start,
                interval_size,
                reference_index,
                self.buckets.iter().cloned(),
            );
            eprintln!(
                "{:>front_padding$.0} {line}",
                (interval_start + interval_size / 2.0)
                    .bold()
                    .bright_yellow()
            );
        }
        eprintln!(
            "{:>front_padding$.0} {}",
            "",
            BucketLine::base(bar_width, self.buckets.len(), reference_index)
        );

        eprintln!(
            "{:>front_padding$.0} {}",
            "",
            self.get_x_axis(bar_width).bold().bright_yellow()
        );
    }
}

#[derive(Debug, Clone)]
struct Percentile {
    value: i64,
    size: usize,
    greater_than_count: usize,
    data_size: usize,
}

impl Percentile {
    fn greater_than_percentage(&self) -> f64 {
        self.greater_than_count as f64 / self.data_size as f64 * 100f64
    }

    fn less_than_percentage(&self) -> f64 {
        (self.data_size - self.greater_than_count - self.size) as f64 / self.data_size as f64
            * 100f64
    }
}

#[derive(Debug, Clone)]
struct PercentileRange {
    start: i64,
    end: i64,
    size: usize,
    data_size: usize,
}

impl PercentileRange {}

#[derive(Debug, Clone)]
struct Percentiles {
    percentiles: Vec<Percentile>,
    data_size: usize,
}

impl Percentiles {
    fn from_data(mut data: Vec<i64>) -> Self {
        data.sort();
        let data_size = data.len();
        let mut percentiles: Vec<Percentile> = vec![];
        for (i, value) in data.into_iter().enumerate() {
            if percentiles.is_empty() || percentiles.last().unwrap().value != value {
                percentiles.push(Percentile {
                    value,
                    size: 0,
                    greater_than_count: i,
                    data_size,
                });
            }
            percentiles.last_mut().unwrap().size += 1;
        }

        Self {
            percentiles,
            data_size,
        }
    }

    fn get_percentile_by_value(&self, value: i64) -> Percentile {
        self.percentiles
            .binary_search_by(|element| match element.value.cmp(&value) {
                Ordering::Equal => Ordering::Less,
                ord => ord,
            })
            .map_err(|i| {
                if i == 0 {
                    return Percentile {
                        value,
                        size: 0,
                        greater_than_count: 0,
                        data_size: self.data_size,
                    };
                }

                self.percentiles
                    .get(i - 1)
                    .map(|p| {
                        if p.value == value {
                            p.clone()
                        } else {
                            Percentile {
                                value,
                                size: 0,
                                greater_than_count: p.greater_than_count + p.size,
                                data_size: self.data_size,
                            }
                        }
                    })
                    .unwrap()
            })
            .unwrap_err()
    }

    fn get_greater_range_by_percentage(&self, percentage: usize) -> Option<PercentileRange> {
        if percentage >= 100 || self.percentiles.len() == 0 {
            return None;
        }

        let value = (percentage as f64 / 100f64 * self.data_size as f64).round() as usize;

        let idx = self
            .percentiles
            .binary_search_by(|element| match element.greater_than_count.cmp(&value) {
                Ordering::Equal => Ordering::Greater,
                ord => ord,
            })
            .unwrap_err();

        if idx >= self.percentiles.len() {
            return None;
        }

        self.percentiles
            .get(idx)
            .map(|l| self.percentiles.last().map(|r| (l, r)))
            .flatten()
            .map(|(l, r)| PercentileRange {
                start: l.value,
                end: r.value,
                size: self.data_size - l.greater_than_count,
                data_size: self.data_size,
            })
    }
}

struct Statistics {
    buckets: Buckets,
    sorted_data: Vec<i64>,
    reference: Option<i64>,
}

impl Statistics {
    fn from_data(mut data: Vec<i64>) -> Self {
        data.sort();
        Self {
            buckets: Buckets::from_data(&data),
            sorted_data: data,
            reference: None,
        }
    }

    fn get_percentile(normalized_percent: f64) -> Option<Percentile> {
        if normalized_percent < 0.0 || normalized_percent >= 1.0 {
            return None;
        }

        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_printed_width() {
        assert_eq!(printed_width(0), 1);
        assert_eq!(printed_width(-1), 2);
        assert_eq!(printed_width(12321313), 8);
        assert_eq!(printed_width(000010), 2);
        assert_eq!(printed_width(10002), 5);
    }

    #[test]
    fn test_which() {
        let buckets = Buckets::from_data(&vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
        assert_eq!(buckets.which(1), Some(0));
        assert_eq!(buckets.which(0), None);
        assert_eq!(buckets.which(2), Some(1));
    }

    #[test]
    fn test_percentiles_get_by_value() {
        let percentiles = Percentiles::from_data(vec![1, 1, 3, 3, 4, 4, 5, 5]);
        assert_matches!(
            percentiles.get_percentile_by_value(1),
            Percentile {
                value: 1,
                size: 2,
                greater_than_count: 0,
                data_size: 8
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(3),
            Percentile {
                value: 3,
                size: 2,
                greater_than_count: 2,
                data_size: 8
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(-1),
            Percentile {
                value: -1,
                size: 0,
                greater_than_count: 0,
                data_size: 8
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(2),
            Percentile {
                value: 2,
                size: 0,
                greater_than_count: 2,
                data_size: 8
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(6),
            Percentile {
                value: 6,
                size: 0,
                greater_than_count: 8,
                data_size: 8
            }
        );
    }

    #[test]
    fn test_percentiles_get_gt_by_percentage() {
        let percentiles = Percentiles::from_data(vec![1, 1, 3, 3, 4, 4, 5, 5]);
        assert_matches!(
            percentiles.get_greater_range_by_percentage(0),
            Some(PercentileRange {
                start: 1,
                end: 5,
                size: 8,
                data_size: 8
            })
        );
        assert_matches!(percentiles.get_greater_range_by_percentage(100), None);
        assert_matches!(
            percentiles.get_greater_range_by_percentage(50),
            Some(PercentileRange {
                start: 4,
                end: 5,
                size: 4,
                data_size: 8
            })
        );
        assert_matches!(
            percentiles.get_greater_range_by_percentage(25),
            Some(PercentileRange {
                start: 3,
                end: 5,
                size: 6,
                data_size: 8
            })
        );
        assert_matches!(
            percentiles.get_greater_range_by_percentage(75),
            Some(PercentileRange {
                start: 5,
                end: 5,
                size: 2,
                data_size: 8
            })
        );
    }

    #[test]
    fn test_percentile_percentages() {
        let percentiles = Percentiles::from_data(vec![0, 2, 4, 6, 7, 8, 10, 12, 14, 20]);
        let percentile = percentiles.get_percentile_by_value(9);
        println!("{:?}", percentile.greater_than_percentage());
        println!("{:?}", percentile.less_than_percentage());
        // assert_matches!(
        //     percentiles.get_percentile_by_value(1),
        //     Percentile {
        //         value: 1,
        //         size: 2,
        //         greater_than_count: 0,
        //         data_size: 8
        //     }
        // );
    }
}
