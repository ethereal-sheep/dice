#![feature(assert_matches)]
use core::fmt;
use std::{cmp::Ordering, rc::Rc};

use clap::{arg, command, value_parser, Command};
use dice::{Dice, ExecOutput, RollOptions, TestOptions};
use num_bigint::BigUint;
use owo_colors::OwoColorize;
use rand::{rngs::SmallRng, SeedableRng};

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
                    --raw "Prints a delimetered list of values if result is an array"
                ))
                .arg(arg!(
                    --experimental "Allows experimental features; might not work correctly!"
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
                )
                .arg(arg!(
                    --experimental "Allows experimental features; might not work correctly!"
                )),
        ).subcommand(
            Command::new("doc")
                .about("Shows the dice script documentation")
                .arg(arg!(
                    --nologo "Suppresses logo"
                ))
                .arg(arg!(
                    --experimental "Show experimental features in documentation"
                )),
        )
        .get_matches();

    let start_width = 12;
    if let Some(matches) = matches.subcommand_matches("roll") {
        let is_debug = matches.get_flag("debug");
        let is_raw = matches.get_flag("raw");
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
                    let mut rng: SmallRng = if let Some(seed) = seed {
                        SmallRng::seed_from_u64(*seed)
                    } else {
                        SmallRng::from_entropy()
                    };
                    if is_debug {
                        if let Some(seed) = seed {
                            println!(
                                "{:>start_width$} with seeded {}; Seed {}",
                                "Rolling".bold().bright_cyan(),
                                "Small RNG".bold().bright_cyan(),
                                seed.bold().bright_yellow(),
                            );
                        } else {
                            println!(
                                "{:>start_width$} with {}",
                                "Rolling".bold().bright_cyan(),
                                "Small RNG".bold().bright_cyan()
                            );
                        }
                    }
                    let result = dice.roll(
                        &mut rng,
                        RollOptions {
                            include_line_details: is_debug,
                        },
                    );
                    match result {
                        Ok(result) => {
                            if is_debug {
                                if let Some(middle_width) = result
                                    .details
                                    .iter()
                                    .map(|line| line.operation.len())
                                    .max()
                                    .map(|w| w.max(15))
                                {
                                    for line in result.details.iter() {
                                        println!(
                                            "{:>start_width$} {:middle_width$} => {}",
                                            line.name.bold().bright_yellow(),
                                            line.operation.bright_magenta(),
                                            line.output
                                        );
                                    }
                                    print!(
                                        "{:>start_width$} {:.<middle_width$} => ",
                                        "Result".bold().bright_green(),
                                        ""
                                    );
                                    if let ExecOutput::Array(v) = &result.output {
                                        if !is_raw && v.len() > 1 {
                                            print!("+(...) => ");
                                        }
                                    }
                                } else {
                                    print!("{:>start_width$} => ", "Result".bold().bright_cyan());
                                }

                                println!("{}", result.result_string().bold());
                            }

                            if is_raw {
                                println!("{}", result.raw_string())
                            } else {
                                println!("{}", result.value())
                            }
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
                    let test_size = if let Some(n) = matches.get_one::<u64>("size") {
                        *n as usize
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

                    if is_debug {
                        println!(
                            "{:>start_width$} {}",
                            "Compiled".bold().yellow(),
                            dice.bright_magenta()
                        );
                    }
                    let seed = matches.get_one::<u64>("seed");
                    let mut rng = if let Some(seed) = seed {
                        SmallRng::seed_from_u64(*seed)
                    } else {
                        SmallRng::from_entropy()
                    };
                    if is_debug {
                        if let Some(seed) = seed {
                            println!(
                                "{:>start_width$} with seeded {}; Seed {}",
                                "Testing".bold().bright_cyan(),
                                "Small RNG".bold().bright_cyan(),
                                seed.bold().bright_yellow(),
                            );
                        } else {
                            println!(
                                "{:>start_width$} with {}",
                                "Testing".bold().bright_cyan(),
                                "Small RNG".bold().bright_cyan()
                            );
                        }
                    }

                    let middle_width = 15;

                    if is_debug {
                        eprintln!(
                            "{:>start_width$} {:<middle_width$} => {}",
                            "Running".bold().bright_cyan(),
                            "test of size",
                            test_size.bold().bright_yellow()
                        );
                        eprintln!(
                            "{:>start_width$} {:<middle_width$} => {}",
                            "",
                            "search space",
                            big_uint_to_scientific(dice.search_space(), None, None)
                                .bold()
                                .bright_yellow()
                        );
                        eprintln!(
                            "{:>start_width$} {:<middle_width$} => {}",
                            "",
                            "step count",
                            dice.step_count().bold().bright_yellow()
                        );
                    }

                    let options = TestOptions {
                        is_debug,
                        test_size: test_size as usize,
                        interval_callback: Some(Rc::new(move |info| {
                            let interval = if info.test_size() < 100 {
                                1
                            } else {
                                info.test_size() / 100
                            };

                            let debug_message = if is_debug {
                                let message = format!(
                                    "{:>start_width$} {:<middle_width$} =>",
                                    info.current_test_info()
                                        .operation_code()
                                        .bright_yellow()
                                        .bold(),
                                    info.current_test_info().operation_name().bright_magenta()
                                );
                                Some(message)
                            } else {
                                None
                            };

                            if info.current_test_info().iteration_index() % interval == 0 {
                                let interval = if test_size < 100 { 1 } else { test_size / 100 };
                                if let Some(message) = &debug_message {
                                    let percent =
                                        (info.current_test_info().iteration_index() / interval) + 1;
                                    eprint!(
                                        "\x1b[2K\r{message} {}▏{percent:>2}% | ~{} steps ",
                                        progress_string(
                                            info.test_size() as f64,
                                            20,
                                            info.current_test_info().iteration_index() as f64
                                        ),
                                        info.current_test_info().step_count()
                                    );
                                } else {
                                    let interval = if info.total_test_count() < 100 {
                                        1
                                    } else {
                                        info.total_test_count() / 100
                                    };
                                    let percent = info.total_test_index() / interval + 1;
                                    eprint!(
                                    "\x1b[2K\r{:>start_width$} {:<middle_width$} => {}▏{percent:3}% {:<6.2}s ",
                                    "Testing".bold().bright_cyan(),
                                    info.current_test_info().operation_name().bright_magenta(),
                                    progress_string(info.total_test_count() as f64, 20, info.total_test_index() as f64),
                                    info.start_time().elapsed().as_secs_f32().bold()
                                );
                                }
                            }

                            if info.current_test_info().is_last() {
                                if let Some(message) = &debug_message {
                                    eprintln!(
                                        "\x1b[2K\r{message} {:<5}ms | ~{} steps ",
                                        info.current_test_info().start_time().elapsed().as_millis(),
                                        info.current_test_info().step_count()
                                    );
                                }
                            }
                            if info.is_last() {
                                if !is_debug {
                                    eprint!("\x1b[2K\r");
                                }

                                eprintln!(
                                    "{:>start_width$} {:.<middle_width$} => {:<6.2}s",
                                    "Finished".bold().bright_green(),
                                    "",
                                    info.start_time().elapsed().as_secs_f32().bold()
                                );

                                if is_debug {
                                    eprintln!(
                                        "{:>start_width$} {:<middle_width$} => {:<5}ns",
                                        "",
                                        "avg. step time",
                                        info.start_time().elapsed().as_nanos()
                                            / (info.total_step_count() * test_size) as u128
                                    );
                                }
                            }
                        })),
                    };

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
                            let reference = matches.get_one("reference").copied();
                            let buckets = Buckets::from_data(&result.output);
                            buckets.print_histogram(
                                start_width,
                                *matches.get_one::<u64>("height").unwrap_or(&10) as usize,
                                reference,
                                is_debug,
                            );

                            let percentiles = Percentiles::from_data(result.output).unwrap();
                            percentiles.print_table(start_width, reference, is_debug);
                        }
                        Err(err) => {
                            eprintln!("\n{:>start_width$} {}", "Error".red().bold(), err);
                        }
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
    } else if let Some(matches) = matches.subcommand_matches("doc") {
        if !matches.get_flag("nologo") {
            println!("{:>start_width$}", Logo {});
        } else {
            println!("{:>start_width$}", SimpleLogo {});
        }

        println!(
            "{:>start_width$}",
            Doc::new(&[
                ("DOCS", &[Content::paragraph("A documentation on the scripting language of dice.")]),
                (
                    "VARIABLES",
                    &[
                        Content::subheader("VAL", "An integer value."),
                        Content::subheader(
                            "ARR", 
                            "An array of VALs. Can be declared by enclosing values in \nparenthesis, e.g. (1,2,3). ARR will eagerly transform into a VAL \nby way of summation if its operation requires it."
                        ),
                        Content::subheader(
                            "NUM",
                            "An integer value that *MUST* be provided by the user."
                        ),
                        Content::subheader("COUNT?", "An unsigned NUM that defaults to 1 if omitted."),
                        Content::subheader("INDICES", "One or more VAL or RANGE, COMMA delimited"),
                    ]
                ),
                (
                    "OPERATORS",
                    &[
                        Content::subheader("Tokens", "+,-,*"),
                        Content::subheader("Usage", "<VAL><+,-,*><VAL> => VAL"),
                        Content::subheader(
                            "Desc.",
                            "Behaves as expected of math operators. Division is not available."
                        ),
                    ]
                ),
                (
                    "DICE",
                    &[
                        Content::subheader("Keyword", "d/D"),
                        Content::subheader("Usage", "<COUNT?>D<NUM> => ARR"),
                        Content::subheader("Example", "3d4, 2D8, D20, .."),
                        Content::subheader(
                            "Desc.",
                            "Rolls 1 or more dice and returns an array of the results."
                        ),
                    ]
                ),
                (
                    "RANGE",
                    &[
                        Content::subheader("Keyword", ".."),
                        Content::subheader("Usage", "<START>..<END> => ARR"),
                        Content::paragraph("where START and END are NUMs"),
                        Content::subheader("Example", "1..20, 0..5, -1..-10, .."),
                        Content::subheader(
                            "Desc.",
                            "Creates an array of values from START to END inclusively."
                        ),
                    ]
                ),
                (
                    "ADVANTAGE",
                    &[
                        Content::subheader("Keyword", "a/A"),
                        Content::subheader("Usage", "<COUNT?>A<ARR> => ARR"),
                        Content::paragraph("where COUNT <= size of input ARR"),
                        Content::subheader("Example", "A2D20, 2A(1,2,3) => (3,2), .."),
                        Content::subheader(
                            "Desc.",
                            "From an array of values, returns an array of the highest \nCOUNT values."
                        ),
                    ]
                ),
                (
                    "DISADVAN.",
                    &[
                        Content::subheader("Keyword", "z/Z"),
                        Content::subheader("Usage", "<COUNT?>Z<ARR> => ARR"),
                        Content::paragraph("where COUNT <= size of input ARR"),
                        Content::subheader("Example", "Z2D20, 2Z(1,2,3) => (1,2), .."),
                        Content::subheader(
                            "Desc.",
                            "From an array of values, returns an array of the lowest \nCOUNT values."
                        ),
                    ]
                ),
                (
                    "CHOOSE",
                    &[
                        Content::subheader("Keyword", "c/C"),
                        Content::subheader("Usage", "<COUNT?>C<ARR> => ARR"),
                        Content::subheader("Example", "2C(1,2,3)"),
                        Content::subheader(
                            "Desc.",
                            "From an array of values, select, with replacement, COUNT \nvalues and returns a containing array."
                        ),
                    ]
                ),
                (
                    "PICK",
                    &[
                        Content::subheader("Keyword", "p/P"),
                        Content::subheader("Usage", "<COUNT?>P<ARR> => ARR"),
                        Content::paragraph("where COUNT <= size of input ARR"),
                        Content::subheader("Example", "2P(1,2,3)"),
                        Content::subheader(
                            "Desc.",
                            "From an array of values, select, without replacement, COUNT \nvalues and returns a containing array."
                        ),
                    ]
                ),
                (
                    "SELECT",
                    &[
                        Content::subheader("Usage", "[<ARR>|<INDICES>] => ARR"),
                        Content::subheader("Example", "[10d6|1,3..6]"),
                        Content::subheader(
                            "Desc.",
                            "From an array of values I, construct a new array O of size \nsizeOf(INDICES), where O[i] = I[INDICES[i] mod sizeOf(I)]",
                        ),
                    ]
                ),
            ])
        );
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

fn big_uint_to_scientific(
    value: &BigUint,
    limit: Option<usize>,
    precision: Option<usize>,
) -> String {
    let mut value_str = value.to_str_radix(10);
    let exp = value_str.len() - 1;

    if exp < limit.unwrap_or(9) {
        return value_str;
    }

    let trimmed = value_str.trim_end_matches('0');
    let precision = precision.unwrap_or(3).min(trimmed.len());

    value_str.truncate(precision + 2); // 1 (for leading) + 1 (for rounding)
    let value = value_str.parse::<u64>().unwrap() as f64 / 10f64.powf((precision + 1) as f64);

    format!("{:.precision$}e+{}", value, exp)
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

struct Paragraph<'a> {
    sentences: std::str::Split<'a, char>,
}

impl<'a> Paragraph<'a> {
    fn new(string: &'a str) -> Paragraph<'a> {
        Paragraph {
            sentences: string.split('\n'),
        }
    }
}

enum Content<'a> {
    SubHeader(&'a str, Paragraph<'a>),
    Paragraph(Paragraph<'a>),
}

impl<'a> Content<'a> {
    fn paragraph(string: &'a str) -> Content<'a> {
        Content::Paragraph(Paragraph::new(string))
    }
    fn subheader(string: &'a str, paragraph: &'a str) -> Content<'a> {
        Content::SubHeader(string, Paragraph::new(paragraph))
    }
}

struct Doc<'a> {
    contents: &'a [(&'a str, &'a [Content<'a>])],
}

impl<'a> Doc<'a> {
    fn new(contents: &'a [(&'a str, &'a [Content<'a>])]) -> Doc<'a> {
        Doc { contents }
    }
}

impl<'a> fmt::Display for Doc<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(0);
        for (header, contents) in self.contents.iter() {
            let styled_header = header.bold();
            for (i, content) in contents.iter().enumerate() {
                match &content {
                    Content::SubHeader(sub, paragraph) => {
                        if i == 0 {
                            writeln!(f, "{:>width$}", styled_header)?;
                        }
                        for (j, sentence) in paragraph.sentences.clone().enumerate() {
                            if j == 0 {
                                writeln!(f, "{:>width$} {}", sub, sentence)?;
                            } else {
                                writeln!(f, "{:>width$} {}", "", sentence)?;
                            }
                        }
                    }
                    Content::Paragraph(paragraph) => {
                        for (j, sentence) in paragraph.sentences.clone().enumerate() {
                            if j == 0 && i == 0 {
                                writeln!(f, "{:>width$} {}", styled_header, sentence)?;
                            } else {
                                writeln!(f, "{:>width$} {}", "", sentence)?;
                            }
                        }
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

const VERTICAL_BAR_SECTIONS: &[char] = &['┄', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

struct Buckets {
    buckets: Vec<usize>,
    bucket_size: usize,
    min: i64,
    max: i64,
    size: usize,
}

fn get_max_bucket_count() -> usize {
    termsize::get()
        .map(|size| size.cols)
        .and_then(|cols| (cols > 15).then(|| cols as usize - 15))
        .unwrap_or(1)
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

fn printed_width_i64(n: impl Into<i64>) -> usize {
    let number = n.into();
    if number < 0 {
        return printed_width_i64(number.abs()) + 1;
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

fn printed_width_f64(n: f64) -> usize {
    format!("{}", n).len()
}

fn precision_floor(x: f64, digits: u32) -> f64 {
    if x == 0. || digits == 0 {
        0.
    } else {
        let shift = digits as i32 - x.abs().log10().ceil() as i32;
        let shift_factor = 10_f64.powi(shift);

        (x * shift_factor).floor() / shift_factor
    }
}

fn precision_round(x: f64, digits: u32) -> f64 {
    if x == 0. || digits == 0 {
        0.
    } else {
        let shift = digits as i32 - x.abs().log10().ceil() as i32;
        let shift_factor = 10_f64.powi(shift);

        (x * shift_factor).round() / shift_factor
    }
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
        it: impl Iterator<Item = f64>,
    ) -> Self {
        Self {
            chars: it
                .map(|value| {
                    choose_char_for_bucket_at_interval(interval_start, interval_size, value)
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
        if self.buckets.bucket_size == 1 && bar_width > printed_width_i64(self.buckets.max) {
            for i in 0..self.buckets.buckets.len() {
                write!(f, "{:^bar_width$}", self.buckets.min + i as i64)?;
            }
            return write!(f, "");
        }

        let graph_width = self.buckets.get_graph_width(bar_width);

        let front_length = printed_width_i64(self.buckets.min);
        let back_length = printed_width_i64(self.buckets.max);

        let mid_value = self.buckets.min + (self.buckets.max - self.buckets.min) / 2;
        let mid_length = printed_width_i64(mid_value);
        let quartile_value = self.buckets.min + (self.buckets.max - self.buckets.min) / 4;
        let quartile_length = printed_width_i64(quartile_value);
        let triquartile_value = mid_value + (mid_value - quartile_value);
        let triquartile_length = printed_width_i64(triquartile_value);

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
                self.buckets.min,
                "",
                quartile_value,
                "",
                mid_value,
                "",
                triquartile_value,
                "",
                self.buckets.max
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
                self.buckets.min, "", mid_value, "", self.buckets.max
            );
        }

        // for 2 numbers
        if graph_width > back_length + front_length {
            let padding = graph_width - back_length;

            return write!(f, "{}{:padding$}{}", self.buckets.min, "", self.buckets.max);
        }

        write!(f, "")
    }
}

impl Buckets {
    fn from_range(min: i64, max: i64) -> Self {
        let value_range = if min > max {
            1
        } else {
            (max - min + 1) as usize
        };
        let (bucket_count, bucket_size, _extra_range) =
            choose_bucket_count_size_and_extra_range(value_range);
        let buckets: Vec<usize> = vec![0; bucket_count];

        Self {
            buckets,
            bucket_size,
            min,
            max,
            size: 0,
        }
    }

    fn from_data(data: &Vec<i64>) -> Self {
        let min = data.iter().min().copied().unwrap_or(0);
        let max = data.iter().max().copied().unwrap_or(0);

        let mut buckets = Buckets::from_range(min, max);
        buckets.fill(data);
        buckets
    }

    fn fill(&mut self, data: &Vec<i64>) {
        for num in data {
            if *num < self.min || *num > self.max {
                continue;
            }
            let index = (*num - self.min) as usize / self.bucket_size;
            self.buckets[index] += 1;
            self.size += 1;
        }
    }

    fn range(&self) -> usize {
        self.buckets.len() * self.bucket_size
    }

    fn which(&self, value: i64) -> Option<usize> {
        if value < self.min || value > self.max {
            return None;
        }
        let normalized_value = value - self.min;
        if normalized_value < 0 || normalized_value as usize >= self.range() {
            return None;
        }
        Some(normalized_value as usize / self.bucket_size)
    }

    fn get_modal_bucket_index(&self) -> Option<usize> {
        self.buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, &val)| val)
            .map(|(i, _)| i)
    }

    fn get_bucket(&self, index: usize) -> Option<Bucket> {
        if index >= self.buckets.len() {
            return None;
        }
        let start = self.min + (index * self.bucket_size) as i64;
        let size = if start + self.bucket_size as i64 > self.max {
            (self.max - start + 1) as usize
        } else {
            self.bucket_size
        };
        Some(Bucket { start, size })
    }

    fn get_normalized_bucket_value(&self, index: usize) -> Option<f64> {
        if let Some(bucket) = self.get_bucket(index) {
            return Some(self.buckets[index] as f64 / bucket.size as f64);
        }
        None
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

    fn print_histogram(
        &self,
        front_padding: usize,
        graph_height: usize,
        reference: Option<i64>,
        is_debug: bool,
    ) {
        let bar_width = get_max_width(self.buckets.len());
        let reference_index = reference.and_then(|i| self.which(i));
        let interval_count = graph_height.clamp(1, 200);
        let graph_min = 0.0;
        let modal_bucket_normalized_value = self
            .get_modal_bucket_index()
            .and_then(|i| self.get_normalized_bucket_value(i))
            .map(|v| v / self.size as f64)
            .unwrap();
        let graph_max = precision_round(modal_bucket_normalized_value, 2) * self.size as f64;
        let interval_size = (graph_max - graph_min) / interval_count as f64;

        let graph_width = self.get_graph_width(bar_width);
        if is_debug {
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
                    println!(
                        "{:>front_padding$} value of {} belongs in bucket {} ({})",
                        "Reference".bright_magenta().bold(),
                        reference_value.bright_yellow().bold(),
                        index.bright_yellow().bold(),
                        bucket,
                    );
                } else {
                    println!(
                        "{:>front_padding$} value of {} is outside graph range",
                        "Reference".bright_magenta().bold(),
                        reference_value.bright_yellow().bold(),
                    );
                }
            }
        }
        println!("{:>front_padding$} (%)", "Y-Axis".bold().bright_blue(),);
        let increment = interval_size * 100.0 / self.size as f64;
        let top_line = (interval_count as f64 + 0.5) * increment;
        let print_label = |v: f64, line: Option<BucketLine>| {
            if top_line < 0.1 {
                eprint!(
                    "{:>front_padding$.2e}",
                    precision_floor(v, 3).bold().bright_yellow()
                );
            } else {
                let digits = if top_line < 1.0 {
                    3
                } else if top_line < 10.0 {
                    2
                } else {
                    1
                };
                eprint!(
                    "{:>front_padding$.digits$}",
                    precision_floor(v, 3).bold().bright_yellow()
                );
            }

            if let Some(line) = line {
                eprintln!(" {line}");
            } else {
                eprintln!(" {}", BucketLine::empty(bar_width, self.buckets.len()));
            }
        };
        print_label(top_line, None);
        for i in (0..interval_count).rev() {
            let interval_start = i as f64 * interval_size;
            let line = BucketLine::new(
                bar_width,
                interval_start,
                interval_size,
                reference_index,
                (0..self.buckets.len()).map(|i| self.get_normalized_bucket_value(i).unwrap()),
            );
            print_label(
                (interval_start + interval_size / 2.0) * 100.0 / self.size as f64,
                Some(line),
            );
        }
        println!(
            "{:>front_padding$.0} {}",
            "",
            BucketLine::base(bar_width, self.buckets.len(), reference_index)
        );

        println!(
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
    fn value(&self) -> i64 {
        self.value
    }

    fn size(&self) -> usize {
        self.size
    }

    fn greater_than_count(&self) -> usize {
        self.greater_than_count
    }

    fn less_than_count(&self) -> usize {
        self.data_size - self.greater_than_count - self.size
    }

    fn percentage(&self) -> f64 {
        self.size() as f64 / self.data_size as f64 * 100f64
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
}

#[derive(Debug, Clone, Copy)]
enum PercentileCompare {
    Greater,
    Less,
}

impl PercentileCompare {
    fn iterator() -> impl Iterator<Item = PercentileCompare> {
        [Self::Greater, Self::Less].iter().copied()
    }

    fn column_name(&self) -> &'static str {
        match *self {
            PercentileCompare::Greater => "Greater than",
            PercentileCompare::Less => "Less than",
        }
    }
}

impl Percentiles {
    fn from_data(mut data: Vec<i64>) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        data.sort();
        // let median = if data.len() % 2 == 0 {
        //     (data.get(data.len() / 2).unwrap() + data.get(data.len() / 2 - 1).unwrap()) as f64 / 2.0
        // } else {
        //     *data.get(data.len() / 2).unwrap() as f64
        // };

        // let mean = data.iter().sum::<i64>() as f64 / data.len() as f64;

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
        })
    }

    // fn mode(&self) -> i64 {
    //     self.percentiles
    //         .iter()
    //         .max_by(|a, b| a.size.cmp(&b.size))
    //         .unwrap()
    //         .value
    // }

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
        if percentage >= 100f64 || self.percentiles.is_empty() {
            return None;
        }

        let value = (percentage / 100f64 * self.data_size as f64).round() as usize;

        let mut idx = self
            .percentiles
            .binary_search_by(|element| match cmp {
                PercentileCompare::Greater => match element.greater_than_count().cmp(&value) {
                    Ordering::Equal => Ordering::Greater,
                    ord => ord,
                },
                PercentileCompare::Less => match value.cmp(&element.less_than_count()) {
                    Ordering::Equal => Ordering::Greater,
                    ord => ord,
                },
            })
            .unwrap_err();

        if let PercentileCompare::Less = cmp {
            if idx < self.percentiles.len() && self.percentiles[idx].less_than_count() < value {
                idx = idx.overflowing_sub(1).0;
            }
        }

        if idx >= self.percentiles.len() {
            return None;
        }

        self.percentiles.get(idx)
    }

    fn last(&self, cmp: PercentileCompare) -> &Percentile {
        match cmp {
            PercentileCompare::Greater => self.percentiles.last().unwrap(),
            PercentileCompare::Less => self.percentiles.first().unwrap(),
        }
    }

    fn print_table(&self, start_width: usize, reference: Option<i64>, _is_debug: bool) {
        eprintln!(
            "{:>start_width$} percentile tables; data contains {} unique values",
            "Drawing".bright_cyan().bold(),
            self.percentiles.len().bright_yellow().bold(),
        );

        let reference_percentile = reference.map(|i| self.get_percentile_by_value(i));

        if let Some(percentile) = &reference_percentile {
            println!(
                "{:>start_width$} Value of {} has a likelihood of {}; Greater than {}; Less than {}",
                "Reference".bright_magenta().bold(),
                percentile.value()
                    .bold()
                    .bright_yellow(),
                format_args!("{}%", precision_floor(percentile.percentage(), 3))
                    .bold()
                    .bright_yellow(),
                format_args!("{}%", precision_floor(percentile.greater_than_percentage(), 3))
                    .bold()
                    .bright_yellow(),
                format_args!("{}%", precision_floor(percentile.less_than_percentage(), 3))
                    .bold()
                    .bright_yellow(),
            );
        }

        let map_percentile =
            |percentile: Option<&Percentile>, cmp: PercentileCompare, digits: Option<u32>| {
                percentile.map(|p| {
                    (
                        p.value(),
                        precision_floor(p.inverse_percentage(cmp), digits.unwrap_or(3)),
                        p.inverse_count(cmp),
                    )
                })
            };

        const FIXED_PERCENTILES: &[f64] =
            &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut percentiles = FIXED_PERCENTILES
            .iter()
            .map(|p| {
                (
                    *p,
                    PercentileCompare::iterator()
                        .map(|cmp| {
                            map_percentile(self.get_percentile_by_percentage(*p, cmp), cmp, Some(3))
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let mut dynamic_percentile = 99.0;
        let mut loop_count = 0;
        loop {
            if dynamic_percentile > 99.999 {
                break;
            }
            let v = PercentileCompare::iterator()
                .map(|cmp| {
                    map_percentile(
                        self.get_percentile_by_percentage(dynamic_percentile, cmp),
                        cmp,
                        Some(loop_count + 3),
                    )
                })
                .collect::<Vec<_>>();
            let should_stop = v.iter().all(|o| o.is_none());
            if should_stop {
                break;
            }
            percentiles.push((dynamic_percentile, v));
            dynamic_percentile = dynamic_percentile / 10.0 + 90.0;
            loop_count += 1;
        }

        let v = PercentileCompare::iterator()
            .map(|cmp| map_percentile(Some(self.last(cmp)), cmp, Some(7)))
            .collect::<Vec<_>>();
        let last_values = v.iter().map(|o| Some(o.unwrap().0)).collect::<Vec<_>>();
        let last_percentile_values = percentiles
            .last()
            .map(|(_, v)| v.iter().map(|o| o.map(|(v, _, _)| v)).collect::<Vec<_>>())
            .unwrap();
        if last_percentile_values
            .into_iter()
            .zip(last_values.into_iter())
            .any(|(lhs, rhs)| lhs == rhs)
        {
            percentiles.last_mut().unwrap().0 = f64::MAX;
            percentiles.last_mut().unwrap().1 = v;
        } else {
            percentiles.push((f64::MAX, v));
        }

        if let Some(p) = &reference_percentile {
            PercentileCompare::iterator()
                .enumerate()
                .for_each(|(i, cmp)| {
                    let inverse_count = p.inverse_count(cmp);
                    let mut idx = percentiles
                        .binary_search_by(|(_, element)| {
                            if let Some((v, _, count)) = element[i] {
                                return match count.cmp(&inverse_count) {
                                    Ordering::Equal => match cmp {
                                        PercentileCompare::Greater => match v.cmp(&p.value) {
                                            Ordering::Equal => Ordering::Less,
                                            ord => ord,
                                        },
                                        PercentileCompare::Less => match v.cmp(&p.value) {
                                            Ordering::Equal => Ordering::Greater,
                                            Ordering::Less => Ordering::Greater,
                                            Ordering::Greater => Ordering::Less,
                                        },
                                    },
                                    ord => ord,
                                };
                            }
                            Ordering::Greater
                        })
                        .unwrap_err();

                    if idx < percentiles.len() && percentiles[idx].0.is_nan() {
                        let (_, _, c) = percentiles[idx]
                            .1
                            .iter()
                            .cloned()
                            .find(|v| v.is_some())
                            .flatten()
                            .unwrap();
                        if c < inverse_count {
                            idx += 1;
                        }
                    }

                    let mut v: Vec<Option<(i64, f64, usize)>> = vec![None; 2];
                    v[i] = map_percentile(Some(p), cmp, Some(3));
                    percentiles.insert(idx, (f64::NAN, v));
                });
        }

        const INNER_COLUMN_NAMES: &[&str] = &["Value", "Actual(%)"];
        let column_widths = PercentileCompare::iterator()
            .enumerate()
            .map(|(i, cmp)| {
                let inner_column_widths = vec![
                    std::cmp::max(
                        INNER_COLUMN_NAMES[0].len(),
                        percentiles
                            .iter()
                            .map(|(_, v)| v[i].map_or(0, |c| printed_width_i64(c.0)))
                            .max()
                            .unwrap_or(0),
                    ),
                    std::cmp::max(
                        INNER_COLUMN_NAMES[1].len(),
                        percentiles
                            .iter()
                            .map(|(_, v)| v[i].map_or(0, |c| printed_width_f64(c.1)))
                            .max()
                            .unwrap_or(0),
                    ),
                ];
                (
                    std::cmp::max(
                        cmp.column_name().len(),
                        inner_column_widths.iter().cloned().sum::<usize>()
                            + (inner_column_widths.len() - 1) * 3,
                    ),
                    inner_column_widths,
                )
            })
            .collect::<Vec<_>>();

        print!("{:>start_width$}   ", "",);
        for (i, cmp) in PercentileCompare::iterator().enumerate() {
            let (width, _) = &column_widths[i];
            print!("{:^width$}   ", cmp.column_name().bright_yellow().bold(),);
        }
        println!();
        // let separator = "+".bright_black();
        print!("{:>start_width$}   ", "P%".bright_cyan().bold());
        for (i, _) in PercentileCompare::iterator().enumerate() {
            let (_, v) = &column_widths[i];
            for (width, name) in v.iter().zip(INNER_COLUMN_NAMES.iter()) {
                print!("{:>width$}   ", name.bright_yellow().bold());
            }
        }
        println!();

        for (percentile, v) in percentiles {
            if percentile == f64::MAX {
                print!("{:>start_width$}   ", "Last".bright_cyan().bold(),);
            } else if percentile.is_nan() {
                print!("{:>start_width$}   ", "Ref".bright_magenta().bold(),);
            } else {
                print!(
                    "{:>start_width$}   ",
                    format!("{percentile}%").bright_cyan().bold(),
                );
            };
            for (i, o) in v.into_iter().enumerate() {
                let (full_width, inner_widths) = &column_widths[i];
                if let Some((col1, col2, _)) = o {
                    let first_inner_column_width = inner_widths[0];
                    let second_inner_column_width = inner_widths[1];
                    if percentile.is_nan() {
                        print!(
                            "{:>full_width$}   ",
                            format!(
                                "{:>first_inner_column_width$}   {:>second_inner_column_width$}",
                                col1.bold().bright_magenta(),
                                col2.bold().bright_magenta()
                            ),
                        );
                    } else {
                        print!(
                            "{:>full_width$}   ",
                            format!(
                                "{:>first_inner_column_width$}   {:>second_inner_column_width$}",
                                col1, col2
                            ),
                        );
                    }
                } else {
                    print!("{:->full_width$}   ", "-".bright_black());
                }
            }
            println!();
        }
    }
}

// struct Statistics {
//     mean: f64,
//     median: i64,
//     mode: i64,
// }

// impl Statistics {
//     fn from_data(mut data: Vec<i64>) -> Self {
//         data.sort();
//         Self {
//             buckets: Buckets::from_data(&data),
//             sorted_data: data,
//             reference: None,
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use super::*;

    #[test]
    fn test_printed_width() {
        assert_eq!(printed_width_i64(0), 1);
        assert_eq!(printed_width_i64(-1), 2);
        assert_eq!(printed_width_i64(12321313), 8);
        assert_eq!(printed_width_i64(10), 2);
        assert_eq!(printed_width_i64(10002), 5);
    }

    #[test]
    fn test_which() {
        let mut buckets = Buckets::from_range(1, 5);
        buckets.fill(&vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 7]);
        assert_matches!(buckets.which(1), Some(_));
        assert_matches!(buckets.which(2), Some(_));
        assert_eq!(buckets.which(7), None);
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
        assert_matches!(
            percentiles.get_percentile_by_value(1),
            Percentile {
                value: 1,
                size: 0,
                greater_than_count: 1,
                data_size: 10,
                first: 0,
                last: 20
            }
        );
    }
}
