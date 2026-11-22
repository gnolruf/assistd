//! Field types shared across tables, each holding one key's range or format
//! rule so a value that parses is valid.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use assistd_utils::hf::HfFileId;
use assistd_utils::path::expand_tilde_from_env;
use serde::{Deserialize, Serialize};

use crate::errors::{
    AbsoluteDirError, BlankStringError, EmptyListError, HfFileError, IntegerRangeError,
};

/// Declares an `f32` setting that holds only the values its predicate accepts, with
/// `new`, `get` and the `f32` conversions serde reads and writes it through. `$expected`
/// completes "must be …" and should rule out NaN along with the predicate.
macro_rules! decimal_range {
    ($(#[$attribute:meta])* $name:ident, |$value:ident| $accepted:expr, $expected:literal) => {
        $(#[$attribute])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(try_from = "f32", into = "f32")]
        pub struct $name(f32);

        impl $name {
            #[doc = concat!("Returns `None` unless `value` is ", $expected, ".")]
            pub const fn new($value: f32) -> Option<Self> {
                if $accepted { Some(Self($value)) } else { None }
            }

            pub const fn get(self) -> f32 {
                self.0
            }
        }

        impl TryFrom<f32> for $name {
            type Error = $crate::errors::DecimalRangeError;

            fn try_from(value: f32) -> Result<Self, Self::Error> {
                Self::new(value).ok_or($crate::errors::DecimalRangeError {
                    value,
                    expected: $expected,
                })
            }
        }

        impl From<$name> for f32 {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

/// Implements the `String` conversions serde reads and writes a setting through:
/// `TryFrom<String>` by the type's `FromStr`, and `From<T> for String` by the closure.
/// A trailing `display` adds a `Display` that delegates to a newtype's field.
macro_rules! string_setting {
    (@display $setting:ty) => {
        impl ::std::fmt::Display for $setting {
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                ::std::fmt::Display::fmt(&self.0, formatter)
            }
        }
    };
    ($setting:ty $(where $server:ident: $kind:path)?, |$value:ident| $text:expr
        $(, $display:ident)?) => {
        impl$(<$server: $kind>)? TryFrom<String> for $setting {
            type Error = <Self as ::std::str::FromStr>::Err;

            fn try_from(text: String) -> Result<Self, Self::Error> {
                text.parse()
            }
        }

        impl$(<$server: $kind>)? From<$setting> for String {
            fn from($value: $setting) -> Self {
                $text
            }
        }

        $($crate::values::string_setting!(@$display $setting);)?
    };
}

decimal_range! {
    /// A sampling temperature: an `f32` in `0.0..=2.0`; never NaN.
    Temperature, |value| value >= 0.0 && value <= 2.0, "in the range 0.0..=2.0"
}

decimal_range! {
    /// A presence penalty: an `f32` in `-2.0..=2.0`; never NaN.
    PresencePenalty, |value| value >= -2.0 && value <= 2.0, "in the range -2.0..=2.0"
}

decimal_range! {
    /// A speaking-rate scale: a positive, finite `f32`.
    LengthScale, |value| value > 0.0 && value < f32::INFINITY, "a positive, finite number"
}

/// A `u32` in `MIN..=MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct Bounded<const MIN: u32, const MAX: u32>(u32);

impl<const MIN: u32, const MAX: u32> Bounded<MIN, MAX> {
    /// Returns `None` when `value` is outside `MIN..=MAX`.
    pub const fn new(value: u32) -> Option<Self> {
        if value >= MIN && value <= MAX {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl<const MIN: u32, const MAX: u32> fmt::Display for Bounded<MIN, MAX> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl<const MIN: u32, const MAX: u32> TryFrom<u32> for Bounded<MIN, MAX> {
    type Error = IntegerRangeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(IntegerRangeError {
            value,
            min: MIN,
            max: MAX,
        })
    }
}

impl<const MIN: u32, const MAX: u32> From<Bounded<MIN, MAX>> for u32 {
    fn from(value: Bounded<MIN, MAX>) -> Self {
        value.0
    }
}

/// A string with at least one non-whitespace character.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct NonBlankString(String);

impl NonBlankString {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for NonBlankString {
    type Err = BlankStringError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.trim().is_empty() {
            Err(BlankStringError)
        } else {
            Ok(Self(text.to_owned()))
        }
    }
}

string_setting!(NonBlankString, |text| text.0, display);

/// A list with at least one entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<T>", into = "Vec<T>")]
#[serde(bound(
    serialize = "T: Serialize + Clone",
    deserialize = "T: Deserialize<'de>"
))]
pub struct NonEmpty<T>(Vec<T>);

impl<T> NonEmpty<T> {
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> TryFrom<Vec<T>> for NonEmpty<T> {
    type Error = EmptyListError;

    fn try_from(entries: Vec<T>) -> Result<Self, Self::Error> {
        if entries.is_empty() {
            Err(EmptyListError)
        } else {
            Ok(Self(entries))
        }
    }
}

impl<T> From<NonEmpty<T>> for Vec<T> {
    fn from(entries: NonEmpty<T>) -> Self {
        entries.0
    }
}

/// An absolute directory path. Parsing expands a leading `~` to `$HOME` and
/// refuses a path that is still relative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "PathBuf")]
pub struct AbsoluteDir(PathBuf);

impl AbsoluteDir {
    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

impl FromStr for AbsoluteDir {
    type Err = AbsoluteDirError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let dir = expand_tilde_from_env(raw);
        if dir.is_absolute() {
            Ok(Self(dir))
        } else {
            Err(AbsoluteDirError(raw.to_owned()))
        }
    }
}

impl TryFrom<String> for AbsoluteDir {
    type Error = AbsoluteDirError;

    fn try_from(raw: String) -> Result<Self, Self::Error> {
        raw.parse()
    }
}

impl From<AbsoluteDir> for PathBuf {
    fn from(dir: AbsoluteDir) -> Self {
        dir.0
    }
}

/// A Hugging Face file written as `<owner>/<repo>[@<revision>]:<file>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct HfFile(HfFileId);

impl HfFile {
    pub fn id(&self) -> &HfFileId {
        &self.0
    }
}

impl FromStr for HfFile {
    type Err = HfFileError;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        HfFileId::parse(id).map(Self).map_err(|reason| HfFileError {
            id: id.to_owned(),
            reason,
        })
    }
}

string_setting!(HfFile, |file| file.to_string(), display);

pub(crate) use {decimal_range, string_setting};
