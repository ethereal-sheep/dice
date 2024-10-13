use dice::Dice;
use rand::thread_rng;

pub fn main() {
    let x = "$220";
    match x.parse::<Dice>() {
        Ok(dice) => println!("{:?}", dice.roll(&mut thread_rng())),
        Err(err) => println!("{}", err),
    }
}
