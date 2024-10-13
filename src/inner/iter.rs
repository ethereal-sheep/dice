#[derive(Debug, Clone)]
pub struct Countable<T: Iterator + Clone> {
    it: T,
    n: usize,
}

impl<T: Iterator + Clone> Countable<T> {
    pub fn new(it: T) -> Self {
        Self { it, n: 0 }
    }

    pub fn curr_index(&self) -> usize {
        self.n
    }

    pub fn peek(&mut self) -> Option<T::Item> {
        let mut temp = self.it.clone();
        temp.next()
    }

    pub fn advance_by(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        self.nth(n - 1);
    }

    pub fn diff(&self, rhs: &Self) -> usize {
        if self.n > rhs.n {
            self.n - rhs.n
        } else {
            rhs.n - self.n
        }
    }
}

impl<T: Iterator + Clone> Iterator for Countable<T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.it.next() {
            self.n += 1;
            return Some(x);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_countable_independent() {
        let x = "abcdefg";
        let mut chars = x.chars();
        let mut countable = Countable::new(chars.clone());

        assert_eq!(countable.next(), Some('a'));
        assert_eq!(countable.next(), Some('b'));
        assert_eq!(countable.next(), Some('c'));
        assert_eq!(countable.next(), Some('d'));

        // assert chars unchanged
        assert_eq!(chars.next(), Some('a'));
        assert_eq!(chars.next(), Some('b'));
        assert_eq!(chars.next(), Some('c'));
        assert_eq!(chars.next(), Some('d'));
    }

    #[test]
    fn test_countable_count() {
        let x = "abcdefg";
        let mut countable = Countable::new(x.chars().clone());

        let n = 4;
        assert_eq!(countable.nth(n - 1), Some('d'));
        assert_eq!(countable.curr_index(), n);
    }

    #[test]
    fn test_countable_peek() {
        let x = "abcdefg";
        let mut countable = Countable::new(x.chars().clone());

        assert_eq!(countable.peek(), Some('a'));
        assert_eq!(countable.curr_index(), 0);
    }

    #[test]
    fn test_countable_advance_by() {
        let x = "abcdefg";
        let mut countable = Countable::new(x.chars().clone());

        let n = 5;
        countable.advance_by(n);
        assert_eq!(countable.curr_index(), n);
    }

    #[test]
    fn test_countable_diff() {
        let x = "abcdefghijk";
        let mut lhs = Countable::new(x.chars().clone());
        let mut rhs = Countable::new(x.chars().clone());

        let l = 5;
        lhs.advance_by(l);
        let r = 10;
        rhs.advance_by(r);
        assert_eq!(rhs.diff(&lhs), r - l);
    }
}
