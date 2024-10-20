use core::fmt;

use clap::{arg, command, value_parser, Command};
use dice::{Dice, RollOptions, TestOptions};
use owo_colors::OwoColorize;
use rand::{rngs::StdRng, SeedableRng};

pub fn main() {
    let matches = command!()
        .subcommand(
            Command::new("roll")
                .about("Parses a dice script and rolls it")
                .arg(arg!(<SCRIPT> "Dice script to parse and roll"))
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
                .arg(arg!(<SCRIPT> "Dice script to parse and test"))
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
                            let buckets = Buckets::percentile(&result.output);
                            // if options.is_debug {
                            //     println!("{:start_width$}", result);
                            // }
                            buckets.print_graph(
                                start_width,
                                *matches.get_one::<u64>("height").unwrap_or(&10) as usize,
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
}

impl BucketLine {
    fn new(
        bar_width: usize,
        interval_start: f64,
        interval_size: f64,
        it: impl Iterator<Item = usize>,
    ) -> Self {
        Self {
            chars: it
                .flat_map(|value| {
                    vec![
                        choose_char_for_bucket_at_interval(
                            interval_start,
                            interval_size,
                            value as f64
                        );
                        bar_width
                    ]
                })
                .collect(),
        }
    }
}

impl fmt::Display for BucketLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for c in &self.chars {
            if *c == VERTICAL_BAR_SECTIONS[0] {
                write!(f, "{}", c.bright_black())?;
            } else {
                write!(f, "{}", c)?;
            }
        }
        write!(f, "")
    }
}

struct Bucket {
    start: i64,
    size: usize,
}

impl Bucket {
    fn printed_width(&self) -> usize {
        if self.size == 1 {
            printed_width(self.start)
        } else if self.size == 2 {
            printed_width(self.start) + printed_width(self.start + self.size as i64 - 1) + 2
        } else {
            printed_width(self.start) + printed_width(self.start + self.size as i64 - 1) + 3
        }
    }
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

impl Buckets {
    fn percentile(data: &Vec<i64>) -> Self {
        let min_value = *data.iter().min().unwrap_or(&0);
        let max_value = *data.iter().max().unwrap_or(&100);
        let value_range = (max_value - min_value + 1) as usize;
        let (bucket_count, bucket_size, extra_range) =
            choose_bucket_count_size_and_extra_range(value_range);
        let bucket_min = min_value - (extra_range / 2) as i64;
        eprintln!(
            "{:>12} {} data points",
            "Bucketing".bright_cyan().bold(),
            data.len().bright_yellow().bold()
        );

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

    fn get_graph_width(&self, bar_width: usize) -> usize {
        self.buckets.len() * bar_width
    }

    fn get_x_axis_string(&self, bar_width: usize) -> String {
        // try fit numbers to all buckets
        if self.bucket_size == 1 && bar_width > printed_width(self.max_value) {
            return (0..self.buckets.len())
                .map(|i| {
                    if i % 1 == 0 {
                        format!("{:^bar_width$}", self.buckets_start + i as i64)
                    } else {
                        format!("{:^bar_width$}", "")
                    }
                })
                .collect::<String>();
        }

        let graph_width = self.get_graph_width(bar_width);

        // for i in (2usize..=3usize).rev() {
        //     if (graph_width - 1) % i != 0 {
        //         continue;
        //     }
        //     let section_size = (graph_width - 1) / i;
        //     //|    |    |    |
        // }

        let thirds = graph_width / 3;
        let middle_third = graph_width - 2 * thirds;
        let middle_bucket_index = self.buckets.len() / 2;
        let middle_value = if self.buckets.len() % 2 == 0 {
            let r_middle_bucket_start =
                self.buckets_start + (middle_bucket_index * self.bucket_size) as i64;
            format!("{}", r_middle_bucket_start as f64 - 0.5)
        } else {
            format!(
                "{}",
                Bucket {
                    start: self.buckets_start + (middle_bucket_index * self.bucket_size) as i64,
                    size: self.bucket_size
                }
            )
        };
        let ret = format!(
            "{:<thirds$}{:^middle_third$}{:>thirds$}",
            self.min_value, middle_value, self.max_value
        );

        if ret.len() == graph_width {
            return ret;
        }

        let halves = graph_width / 2;
        let second_half = graph_width - halves;
        format!(
            "{:<halves$}{:>second_half$}",
            self.min_value, self.max_value
        )
    }

    fn print_graph(&self, front_padding: usize, graph_height: usize) {
        let bar_width = get_max_width(self.buckets.len());
        let interval_count = graph_height.clamp(1, 200);
        let graph_min = 0;
        let graph_max = {
            let max_bucket = *self.buckets.iter().max().unwrap_or(&100);
            max_bucket
        };
        let interval_size = (graph_max - graph_min) as f64 / interval_count as f64;

        let graph_width = self.get_graph_width(bar_width);
        let top_line = (interval_count as f64 + 0.5) * interval_size;
        println!(
            "{:>front_padding$} histogram of {} buckets of size {}; occupying {} ({} + 4) rows and {} columns (bar width {})",
            "Drawing".bright_cyan().bold(),
            self.buckets.len().bright_yellow().bold(),
            self.bucket_size.bright_yellow().bold(),
            (graph_height + 4).bright_yellow().bold(),
            graph_height,
            graph_width.bright_yellow().bold(),
            bar_width.bright_yellow().bold(),
        );
        println!(
            "{:>front_padding$} {}",
            "Y-Axis".bold().bright_blue(),
            format!("~{:.1}/step", interval_size).italic()
        );
        println!(
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
                self.buckets.iter().cloned(),
            );
            println!(
                "{:>front_padding$.0} {line}",
                (interval_start + interval_size / 2.0)
                    .bold()
                    .bright_yellow()
            );
        }
        println!("{:>front_padding$.0} {:▔<graph_width$}", "", "");

        println!(
            "{:>front_padding$.0} {}",
            "",
            self.get_x_axis_string(bar_width).bold().bright_yellow()
        );
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_printed_width() {
        assert_eq!(printed_width(0), 1);
        assert_eq!(printed_width(-1), 2);
        assert_eq!(printed_width(12321313), 8);
        assert_eq!(printed_width(000010), 2);
    }
}
