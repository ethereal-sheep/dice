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
                .arg(
                    arg!(
                        --seed <SEED> "Seeds the PRNG with SEED"
                    )
                    .value_parser(value_parser!(u64)),
                )
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
                        --seed <SEED> "Seeds the PRNG with SEED"
                    )
                    .value_parser(value_parser!(u64)),
                )
                .arg(
                    arg!(
                        --reference <REF> "Outputs test data in reference to REF"
                    )
                    .value_parser(value_parser!(i64)).allow_hyphen_values(true).allow_negative_numbers(true),
                ),
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
                    }
                    let seed = matches.get_one::<u64>("seed");
                    let mut rng: StdRng = if let Some(seed) = seed {
                        StdRng::seed_from_u64(*seed)
                    } else {
                        StdRng::from_entropy()
                    };
                    if is_debug {
                        if let Some(seed) = seed {
                            println!(
                                "{:>start_width$} with seeded {}; Seed {}",
                                "Rolling".bold().bright_cyan(),
                                "Std RNG".bold().bright_cyan(),
                                seed.bold().bright_yellow(),
                            );
                        } else {
                            println!(
                                "{:>start_width$} with {}",
                                "Rolling".bold().bright_cyan(),
                                "Std RNG".bold().bright_cyan()
                            );
                        }
                    }
                    let result = dice.roll(&mut rng, RollOptions { is_debug });
                    match result {
                        Ok(result) => {
                            if is_debug {
                                println!("{:start_width$}", result);
                            }
                            println!("{}", result.as_value())
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
                    }
                    let seed = matches.get_one::<u64>("seed");
                    let mut rng: StdRng = if let Some(seed) = seed {
                        StdRng::seed_from_u64(*seed)
                    } else {
                        StdRng::from_entropy()
                    };
                    if is_debug {
                        if let Some(seed) = seed {
                            println!(
                                "{:>start_width$} with seeded {}; Seed {}",
                                "Testing".bold().bright_cyan(),
                                "Std RNG".bold().bright_cyan(),
                                seed.bold().bright_yellow(),
                            );
                        } else {
                            println!(
                                "{:>start_width$} with {}",
                                "Testing".bold().bright_cyan(),
                                "Std RNG".bold().bright_cyan()
                            );
                        }
                    }
                    let result = dice.test(&mut rng, options);
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

                            let percentiles = Percentiles::from_data(result.output).unwrap();
                            percentiles.print(start_width, matches.get_one("reference").copied());
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

const VERTICAL_BAR_SECTIONS: &[char] = &['┄', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
// const HORIZONTAL_BAR_SECTIONS: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

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
    let index = (normalized_position * 8.0).round() as usize;
    VERTICAL_BAR_SECTIONS[index]
}

fn printed_width(n: impl Into<i64>) -> usize {
    let number = n.into();
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

    fn empty(bar_width: usize, buckets_size: usize) -> Self {
        Self {
            chars: vec![VERTICAL_BAR_SECTIONS[0]; buckets_size],
            bar_width,
            reference_index: None,
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
                // if i % 1 == 0 {
                //     write!(f, "{:^bar_width$}", self.buckets.buckets_start + i as i64)?;
                // } else {
                //     write!(f, "{:^bar_width$}", "")?;
                // }
                write!(f, "{:^bar_width$}", self.buckets.buckets_start + i as i64)?;
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
        let reference_index = reference.and_then(|i| self.which(i));
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
            "{:>front_padding$.0} {}",
            top_line.bold().bright_yellow(),
            BucketLine::empty(bar_width, self.buckets.len())
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
    first: i64,
    value: i64,
    last: i64,
    size: usize,
    greater_than_count: usize,
    data_size: usize,
}

impl Percentile {
    fn size(&self) -> usize {
        self.size
    }

    fn percentage(&self) -> f64 {
        self.size() as f64 / self.data_size as f64 * 100f64
    }

    fn greater_than_count(&self) -> usize {
        self.greater_than_count
    }

    fn less_than_count(&self) -> usize {
        self.data_size - self.greater_than_count - self.size
    }

    fn greater_than_percentage(&self) -> f64 {
        self.greater_than_count() as f64 / self.data_size as f64 * 100f64
    }

    fn less_than_percentage(&self) -> f64 {
        self.less_than_count() as f64 / self.data_size as f64 * 100f64
    }

    fn inverse_count(&self, cmp: PercentileCompare) -> usize {
        match cmp {
            PercentileCompare::Greater => self.greater_than_count(),
            PercentileCompare::Less => self.less_than_count(),
        }
    }

    fn inverse_percentage(&self, cmp: PercentileCompare) -> f64 {
        match cmp {
            PercentileCompare::Greater => self.greater_than_percentage(),
            PercentileCompare::Less => self.less_than_percentage(),
        }
    }
}

#[derive(Debug, Clone)]
struct Percentiles {
    percentiles: Vec<Percentile>,
    data_size: usize,
    first: i64,
    last: i64,
    mean: f64,
    median: f64,
}

#[derive(Debug, Clone, Copy)]
enum PercentileCompare {
    Greater,
    Less,
}

impl Percentiles {
    fn from_data(mut data: Vec<i64>) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        data.sort();
        let median = if data.len() % 2 == 0 {
            (data.get(data.len() / 2).unwrap() + data.get(data.len() / 2 - 1).unwrap()) as f64 / 2.0
        } else {
            *data.get(data.len() / 2).unwrap() as f64
        };

        let mean = data.iter().sum::<i64>() as f64 / data.len() as f64;

        let data_size = data.len();

        let mut percentiles: Vec<(i64, usize, usize)> = vec![];
        for (i, value) in data.iter().enumerate() {
            if percentiles.is_empty() || percentiles.last().unwrap().0 != *value {
                percentiles.push((*value, 0, i));
            }
            percentiles.last_mut().unwrap().1 += 1;
        }

        Some(Self {
            percentiles: percentiles
                .into_iter()
                .map(|(value, size, greater_than_count)| Percentile {
                    value,
                    size,
                    greater_than_count,
                    first: *data.first().unwrap(),
                    last: *data.last().unwrap(),
                    data_size,
                })
                .collect(),
            data_size,
            first: *data.first().unwrap_or(&0),
            last: *data.last().unwrap_or(&0),
            mean,
            median,
        })
    }

    fn mode(&self) -> i64 {
        self.percentiles
            .iter()
            .max_by(|a, b| a.size.cmp(&b.size))
            .unwrap()
            .value
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
                        first: self.first,
                        last: self.last,
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
                                first: p.first,
                                last: p.last,
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

    fn get_percentile_by_percentage(
        &self,
        percentage: impl Into<f64>,
        cmp: PercentileCompare,
    ) -> Option<&Percentile> {
        let percentage = percentage.into();
        if percentage >= 100f64 {
            return None;
        }

        let value = (percentage / 100f64 * self.data_size as f64).round() as usize;

        let idx = self
            .percentiles
            .binary_search_by(|element| match cmp {
                PercentileCompare::Greater => match element.greater_than_count().cmp(&value) {
                    Ordering::Equal => Ordering::Greater,
                    ord => ord,
                },
                PercentileCompare::Less => match value.cmp(&element.less_than_count()) {
                    Ordering::Equal => Ordering::Less,
                    ord => ord,
                },
            })
            .unwrap_err();

        match cmp {
            PercentileCompare::Greater => {
                if idx >= self.percentiles.len() {
                    return None;
                }
                self.percentiles.get(idx)
            }
            PercentileCompare::Less => {
                if idx == 0 {
                    return self.percentiles.get(idx).and_then(|p| {
                        if p.less_than_percentage() >= percentage {
                            Some(p)
                        } else {
                            None
                        }
                    });
                }
                self.percentiles.get(idx - 1)
            }
        }
    }

    fn last(&self, cmp: PercentileCompare) -> &Percentile {
        match cmp {
            PercentileCompare::Greater => self.percentiles.last().unwrap(),
            PercentileCompare::Less => self.percentiles.first().unwrap(),
        }
    }
    fn print(&self, front_padding: usize, reference: Option<i64>) {
        eprintln!(
            "{:>front_padding$} percentile tables; data contains {} unique values",
            "Drawing".bright_cyan().bold(),
            self.percentiles.len().bright_yellow().bold(),
        );
        self.print_percentile_table(front_padding, PercentileCompare::Greater, reference);
        self.print_percentile_table(front_padding, PercentileCompare::Less, reference);
        if let Some(reference_value) = reference {
            self.print_reference_stat_table(front_padding, reference_value);
        }
        self.print_stat_table(front_padding);
    }

    fn print_percentile_table(
        &self,
        front_padding: usize,
        cmp: PercentileCompare,
        reference: Option<i64>,
    ) {
        enum PercentileLabel {
            Percent(f64, usize),
            Text(&'static str),
        }

        impl PercentileLabel {
            fn len(&self) -> usize {
                match self {
                    PercentileLabel::Percent(_, precision) => {
                        if *precision == 0 {
                            2
                        } else {
                            precision + 3
                        }
                    }
                    PercentileLabel::Text(s) => s.len(),
                }
            }

            fn precision(&self) -> usize {
                match self {
                    PercentileLabel::Percent(_, precision) => *precision,
                    PercentileLabel::Text(_) => 0,
                }
            }
        }

        impl fmt::Display for PercentileLabel {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let width = f.width().unwrap_or(0);
                match self {
                    PercentileLabel::Percent(float, precision) => {
                        write!(f, "{float:<width$.precision$}")
                    }
                    PercentileLabel::Text(s) => s.fmt(f),
                }
            }
        }

        let reference_title = "Ref";
        let reference_percentile = reference.map(|v| self.get_percentile_by_value(v));

        let percentiles = {
            let mut percentiles: Vec<(PercentileLabel, &Percentile)> = vec![];
            let mut value_column_width = 1;
            for i in 0..=3 {
                let percentage = i * 25;
                if let Some(p) = self.get_percentile_by_percentage(percentage, cmp) {
                    value_column_width = value_column_width.max(printed_width(p.value));
                    percentiles.push((PercentileLabel::Percent(percentage as f64, 0), p));
                } else {
                    break;
                }
            }

            let mut iteration = 0;
            let mut curr_highest = 99f64;
            while let Some(p) = self.get_percentile_by_percentage(curr_highest, cmp) {
                value_column_width = value_column_width.max(printed_width(p.value));
                percentiles.push((PercentileLabel::Percent(curr_highest, iteration), p));
                curr_highest = 90.0 + curr_highest / 10.0;
                iteration += 1;
            }

            percentiles.push((PercentileLabel::Text("Last"), self.last(cmp)));

            if let Some(p) = &reference_percentile {
                let idx = percentiles
                    .binary_search_by(|(_, element)| {
                        match element.inverse_count(cmp).cmp(&p.inverse_count(cmp)) {
                            Ordering::Equal => match element.value.cmp(&p.value) {
                                Ordering::Equal => Ordering::Less,
                                ord => ord,
                            },
                            ord => ord,
                        }
                    })
                    .unwrap_err();
                percentiles.insert(idx, (PercentileLabel::Text(reference_title), p));
            }

            percentiles
        };

        let percentage_header = "%";
        let range_header = "Range";
        let inverse_header = if let PercentileCompare::Greater = cmp {
            ">"
        } else {
            "<"
        };
        let inverse_percentage_header = if let PercentileCompare::Greater = cmp {
            ">%"
        } else {
            "<%"
        };

        let max_precision = {
            let mut max_precision = 0;
            for (label, _) in percentiles.iter() {
                max_precision = max_precision.max(label.precision());
            }
            max_precision + 1
        };

        let (
            percentage_column_width,
            range_column_width,
            inverse_column_width,
            inverse_percentage_column_width,
        ) = {
            let mut percentage_column_width = percentage_header.len();
            let mut range_column_width = range_header.len();
            let mut inverse_column_width = inverse_header.len();
            for (label, percentile) in percentiles.iter() {
                percentage_column_width = percentage_column_width.max(label.len());
                range_column_width = range_column_width.max(printed_width(percentile.value));
                inverse_column_width =
                    inverse_column_width.max(printed_width(percentile.inverse_count(cmp) as u32));
            }
            (
                percentage_column_width,
                range_column_width,
                inverse_column_width,
                4 + max_precision,
            )
        };

        println!(
            "{:>front_padding$}─{:─<percentage_column_width$}─┬─{:─<range_column_width$}─┬─{:─<inverse_column_width$}─┬─{:─<inverse_percentage_column_width$}─┬",
            "┎".bright_yellow(),
            "", "", "", "",
        );
        println!(
            "{:>front_padding$} {:^percentage_column_width$} │ {:^range_column_width$} │ {:^inverse_column_width$} │ {:^inverse_percentage_column_width$} │",
            "┃".bright_yellow(),
            percentage_header,
            range_header,
            inverse_header,
            inverse_percentage_header,
        );
        println!(
            "{:>front_padding$}─{:─<percentage_column_width$}─┼─{:─<range_column_width$}─┼─{:─<inverse_column_width$}─┼─{:─<inverse_percentage_column_width$}─┼",
            "┠".bright_yellow(),
            "", "", "", "",
        );
        for (label, percentile) in percentiles.iter() {
            if let PercentileLabel::Text("Ref") = label {
                println!(
                    "{:>front_padding$} {:<percentage_column_width$} │ {:>range_column_width$} │ {:>inverse_column_width$} │ {:>inverse_percentage_column_width$.max_precision$} │",
                    "┃".bright_yellow(),
                    label.bright_magenta().bold(),
                    percentile.value.bright_magenta().bold(),
                    percentile.inverse_count(cmp).bright_magenta().bold(),
                    percentile.inverse_percentage(cmp).bright_magenta().bold(),
                );
            } else {
                println!(
                    "{:>front_padding$} {:<percentage_column_width$} │ {:>range_column_width$} │ {:>inverse_column_width$} │ {:>inverse_percentage_column_width$.max_precision$} │",
                    "┃".bright_yellow(),
                    label,
                    percentile.value,
                    percentile.inverse_count(cmp),
                    percentile.inverse_percentage(cmp),
                );
            }
        }
        println!(
            "{:>front_padding$}─{:─<percentage_column_width$}─┴─{:─<range_column_width$}─┴─{:─<inverse_column_width$}─┴─{:─<inverse_percentage_column_width$}─┴",
            "┖".bright_yellow(),
            "", "", "", "",
        );
    }

    fn print_reference_stat_table(&self, front_padding: usize, reference_value: i64) {
        let row_header_column_width = 10;
        let less_than_column_width = 11;
        let reference_column_width = 11;
        let greater_than_column_width = 11;

        let percentile = self.get_percentile_by_value(reference_value);

        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┬─{:─<less_than_column_width$}─┬─{:─<reference_column_width$}─┬─{:─<greater_than_column_width$}─┬",
            "┎".bright_yellow(),
            "", "", "", "",
        );
        println!(
            "{:>front_padding$} {:^row_header_column_width$} │ {:^less_than_column_width$} │ {:^reference_column_width$} │ {:^greater_than_column_width$} │",
            "┃".bright_yellow(),
            "",
            "x<",
            "Ref".bold().magenta(),
            "<x",
        );
        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┼─{:─<less_than_column_width$}─┼─{:─<reference_column_width$}─┼─{:─<greater_than_column_width$}─┼",
            "┠".bright_yellow(),
            "", "", "", "",
        );
        println!(
            "{:>front_padding$} {:^row_header_column_width$} │ {:^less_than_column_width$.2} │ {:^reference_column_width$.2} │ {:^greater_than_column_width$.2} │",
            "┃".bright_yellow(),
            "P%",
            percentile.greater_than_percentage(),
            percentile.percentage(),
            percentile.less_than_percentage(),
        );
        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┼─{:─<less_than_column_width$}─┼─{:─<reference_column_width$}─┼─{:─<greater_than_column_width$}─┼",
            "┠".bright_yellow(),
            "", "", "", "",
        );
        println!(
            "{:>front_padding$} {:^row_header_column_width$} │ {:^less_than_column_width$} │ {:^reference_column_width$} │ {:^greater_than_column_width$} │",
            "┃".bright_yellow(),
            "size",
            percentile.greater_than_count(),
            percentile.size(),
            percentile.less_than_count(),
        );
        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┴─{:─<less_than_column_width$}─┴─{:─<reference_column_width$}─┴─{:─<greater_than_column_width$}─┴",
            "┖".bright_yellow(),
            "", "", "", "",
        );
    }

    fn print_stat_table(&self, front_padding: usize) {
        let row_header_column_width = 10;
        let less_than_column_width = 11;
        let reference_column_width = 11;
        let greater_than_column_width = 11;

        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┬─{:─<less_than_column_width$}─┬─{:─<reference_column_width$}─┬─{:─<greater_than_column_width$}─┬",
            "┎".bright_yellow(),
            "", "", "", "",
        );
        println!(
            "{:>front_padding$} {:^row_header_column_width$} │ {:^less_than_column_width$} │ {:^reference_column_width$} │ {:^greater_than_column_width$} │",
            "┃".bright_yellow(),
            "",
            "Mean",
            "Median",
            "Mode",
        );
        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┼─{:─<less_than_column_width$}─┼─{:─<reference_column_width$}─┼─{:─<greater_than_column_width$}─┼",
            "┠".bright_yellow(),
            "", "", "", "",
        );
        println!(
            "{:>front_padding$} {:^row_header_column_width$} │ {:^less_than_column_width$.2} │ {:^reference_column_width$.1} │ {:^greater_than_column_width$} │",
            "┃".bright_yellow(),
            "Value",
            self.mean,
            self.median,
            self.mode(),
        );
        println!(
            "{:>front_padding$}─{:─<row_header_column_width$}─┴─{:─<less_than_column_width$}─┴─{:─<reference_column_width$}─┴─{:─<greater_than_column_width$}─┴",
            "┖".bright_yellow(),
            "", "", "", "",
        );
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

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
        let percentiles = Percentiles::from_data(vec![1, 1, 3, 3, 4, 4, 5, 5]).unwrap();
        assert_matches!(
            percentiles.get_percentile_by_value(1),
            Percentile {
                value: 1,
                size: 2,
                greater_than_count: 0,
                data_size: 8,
                first: 1,
                last: 5
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(3),
            Percentile {
                value: 3,
                size: 2,
                greater_than_count: 2,
                data_size: 8,
                first: 1,
                last: 5
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(-1),
            Percentile {
                value: -1,
                size: 0,
                greater_than_count: 0,
                data_size: 8,
                first: 1,
                last: 5
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(2),
            Percentile {
                value: 2,
                size: 0,
                greater_than_count: 2,
                data_size: 8,
                first: 1,
                last: 5
            }
        );
        assert_matches!(
            percentiles.get_percentile_by_value(6),
            Percentile {
                value: 6,
                size: 0,
                greater_than_count: 8,
                data_size: 8,
                first: 1,
                last: 5
            }
        );
    }

    #[test]
    fn test_percentiles_get_gt_by_percentage() {
        let percentiles: Percentiles =
            Percentiles::from_data(vec![1, 1, 3, 3, 4, 4, 5, 5]).unwrap();
        assert_matches!(
            percentiles.get_percentile_by_percentage(0, PercentileCompare::Greater),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 1,
                greater_than_count: 0
            })
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(100, PercentileCompare::Greater),
            None
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(50, PercentileCompare::Greater),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 4,
                greater_than_count: 4
            })
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(25, PercentileCompare::Greater),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 3,
                greater_than_count: 2
            })
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(75, PercentileCompare::Greater),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 5,
                greater_than_count: 6
            })
        );
    }

    #[test]
    fn test_percentiles_get_lt_by_percentage() {
        let percentiles: Percentiles =
            Percentiles::from_data(vec![1, 1, 3, 3, 4, 4, 5, 5]).unwrap();
        assert_matches!(
            percentiles.get_percentile_by_percentage(0, PercentileCompare::Less),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 5,
                greater_than_count: 6
            })
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(100, PercentileCompare::Less),
            None
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(50, PercentileCompare::Less),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 3,
                greater_than_count: 2
            })
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(25, PercentileCompare::Less),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 4,
                greater_than_count: 4
            })
        );
        assert_matches!(
            percentiles.get_percentile_by_percentage(75, PercentileCompare::Less),
            Some(Percentile {
                size: 2,
                data_size: 8,
                first: 1,
                last: 5,
                value: 1,
                greater_than_count: 0
            })
        );
    }

    #[test]
    fn test_percentile_percentages() {
        let percentiles = Percentiles::from_data(vec![0, 2, 4, 6, 7, 8, 10, 12, 14, 20]).unwrap();
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
