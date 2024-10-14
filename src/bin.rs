use clap::{arg, command, Command};
use dice::Dice;
use rand::thread_rng;

pub fn main() {
    let matches = command!()
        .subcommand(
            Command::new("roll")
                .about("Parse a dice script and roll it")
                .arg(arg!(<SCRIPT> "Dice script to parse and roll"))
                .arg(arg!(
                    -d --debug "Turn debugging information on"
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
    if let Some(matches) = matches.subcommand_matches("roll") {
        // You can check the value provided by positional arguments, or option arguments

        if let Some(script) = matches.get_one::<String>("SCRIPT") {
            // if matches.get_flag("debug") {
            //     // "$ myapp test -l" was run
            //     println!("debugging...");
            // }
            let result = script.parse::<Dice>();
            match result {
                Ok(dice) => {
                    let result = dice.roll(&mut thread_rng());
                    match result {
                        Ok(result) => println!("{:?}", result),
                        Err(err) => println!("Runtime Error: {}", err),
                    }
                }
                Err(err) => println!("{}", err),
            }
        }
    }
}
