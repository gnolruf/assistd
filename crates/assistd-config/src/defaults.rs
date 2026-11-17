//! Builders that check a default literal at compile time, and the fixed
//! values other crates read.

use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};

use crate::values::{Bounded, Temperature};
use crate::voice::UnitInterval;

/// How long a confirmation prompt waits for its answer before it is denied.
pub const CONFIRMATION_PROMPT_SECS: u64 = 120;

/// Builds a `NonZero` from a literal; a zero fails at compile time.
pub const fn nz16(v: u16) -> NonZeroU16 {
    match NonZeroU16::new(v) {
        Some(n) => n,
        None => panic!("default must be non-zero"),
    }
}

/// See [`nz16`].
pub const fn nz32(v: u32) -> NonZeroU32 {
    match NonZeroU32::new(v) {
        Some(n) => n,
        None => panic!("default must be non-zero"),
    }
}

/// See [`nz16`].
pub const fn nz64(v: u64) -> NonZeroU64 {
    match NonZeroU64::new(v) {
        Some(n) => n,
        None => panic!("default must be non-zero"),
    }
}

/// Builds a [`UnitInterval`] from a literal; a value outside `0.0..=1.0`
/// fails at compile time.
pub const fn unit(v: f32) -> UnitInterval {
    match UnitInterval::new(v) {
        Some(u) => u,
        None => panic!("default must be in 0.0..=1.0"),
    }
}

/// Builds a [`Temperature`] from a literal; a value outside `0.0..=2.0`
/// fails at compile time.
pub const fn temperature(v: f32) -> Temperature {
    match Temperature::new(v) {
        Some(t) => t,
        None => panic!("default must be in 0.0..=2.0"),
    }
}

/// Builds a [`Bounded`] from a literal; a value outside `MIN..=MAX` fails at
/// compile time.
pub const fn bounded<const MIN: u32, const MAX: u32>(v: u32) -> Bounded<MIN, MAX> {
    match Bounded::new(v) {
        Some(b) => b,
        None => panic!("default must be in MIN..=MAX"),
    }
}
