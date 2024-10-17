use core::fmt;

use clap::{arg, command, Command};
use dice::{Dice, RollOptions};
use owo_colors::OwoColorize;
use rand::thread_rng;

pub fn main() {
    let matches = command!()
        .subcommand(
            Command::new("roll")
                .about("Parse a dice script and roll it")
                .arg(arg!(<SCRIPT> "Dice script to parse and roll"))
                .arg(arg!(
                    -d --debug "Turn debugging information on"
                ))
                .arg(arg!(
                    --nologo "Suppresses logo in debug mode"
                )),
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

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    let start_width = 12;
    if let Some(matches) = matches.subcommand_matches("roll") {
        // You can check the value provided by positional arguments, or option arguments
        let options = RollOptions {
            is_debug: matches.get_flag("debug"),
        };

        if options.is_debug {
            if !matches.get_flag("nologo") {
                println!("{:>start_width$}", Logo {});
            } else {
                println!("{:>start_width$}", SimpleLogo {});
            }
        }

        if let Some(script) = matches.get_one::<String>("SCRIPT") {
            if options.is_debug {
                println!(
                    "{:>start_width$} {}",
                    "Compiling".bold().bright_green(),
                    script
                );
            }
            let result = script.parse::<Dice>();
            match result {
                Ok(dice) => {
                    if options.is_debug {
                        println!(
                            "{:>start_width$} {}",
                            "Compiled".bold().yellow(),
                            dice.bright_magenta()
                        );
                        println!(
                            "{:>start_width$} {} {}",
                            "Rolling".bold().bright_green(),
                            "with",
                            "thread RNG".bold().bright_cyan()
                        );
                    }
                    let result = dice.roll(&mut thread_rng(), &options);
                    match result {
                        Ok(result) => {
                            if options.is_debug {
                                println!("{:start_width$}", result);
                            }
                            println!("{}", result.into_value())
                        }
                        Err(err) => println!("{:>start_width$} {}", "Error".red().bold(), err),
                    }
                }
                Err(err) => {
                    if options.is_debug {
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
