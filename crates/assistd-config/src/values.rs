//! Field types shared across tables, each holding one key's range or format
//! rule so a value that parses is valid.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use assistd_utils::hf::HfFileId;
use assistd_utils::path::expand_tilde_from_env;
use serde::{Deserialize, Serialize};

use crate::errors::{
    AbsoluteDirError, BlankStringError, DecimalRangeError, EmptyListError, HfFileError,
    IntegerRangeError, LoopbackAddrError,
};

/// A sampling temperature: an `f32` in `0.0..=2.0`; never NaN.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Temperature(f32);

impl Temperature {
    /// Returns `None` when `value` is outside `0.0..=2.0` or NaN.
    pub const fn new(value: f32) -> Option<Self> {
        if value >= 0.0 && value <= 2.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Temperature {
    type Error = DecimalRangeError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(DecimalRangeError {
            value,
            expected: "in the range 0.0..=2.0",
        })
    }
}

impl From<Temperature> for f32 {
    fn from(value: Temperature) -> Self {
        value.0
    }
}

/// A presence penalty: an `f32` in `-2.0..=2.0`; never NaN.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct PresencePenalty(f32);

impl PresencePenalty {
    /// Returns `None` when `value` is outside `-2.0..=2.0` or NaN.
    pub const fn new(value: f32) -> Option<Self> {
        if value >= -2.0 && value <= 2.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for PresencePenalty {
    type Error = DecimalRangeError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(DecimalRangeError {
            value,
            expected: "in the range -2.0..=2.0",
        })
    }
}

impl From<PresencePenalty> for f32 {
    fn from(value: PresencePenalty) -> Self {
        value.0
    }
}

/// A speaking-rate scale: a positive, finite `f32`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct LengthScale(f32);

impl LengthScale {
    /// Returns `None` when `value` is zero, negative, infinite or NaN.
    pub const fn new(value: f32) -> Option<Self> {
        if value > 0.0 && value < f32::INFINITY {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for LengthScale {
    type Error = DecimalRangeError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(DecimalRangeError {
            value,
            expected: "a positive, finite number",
        })
    }
}

impl From<LengthScale> for f32 {
    fn from(value: LengthScale) -> Self {
        value.0
    }
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

/// A loopback address, `127.0.0.0/8` or `::1`, for a server whose traffic
/// must stay on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "IpAddr", into = "IpAddr")]
pub struct LoopbackAddr(IpAddr);

impl LoopbackAddr {
    pub const LOCALHOST: Self = Self(IpAddr::V4(Ipv4Addr::LOCALHOST));

    pub const fn get(self) -> IpAddr {
        self.0
    }
}

impl fmt::Display for LoopbackAddr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl TryFrom<IpAddr> for LoopbackAddr {
    type Error = LoopbackAddrError;

    fn try_from(addr: IpAddr) -> Result<Self, Self::Error> {
        if addr.is_loopback() {
            Ok(Self(addr))
        } else {
            Err(LoopbackAddrError(addr))
        }
    }
}

impl From<LoopbackAddr> for IpAddr {
    fn from(addr: LoopbackAddr) -> Self {
        addr.0
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

impl fmt::Display for NonBlankString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for NonBlankString {
    type Err = BlankStringError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::try_from(text.to_owned())
    }
}

impl TryFrom<String> for NonBlankString {
    type Error = BlankStringError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        if text.trim().is_empty() {
            Err(BlankStringError)
        } else {
            Ok(Self(text))
        }
    }
}

impl From<NonBlankString> for String {
    fn from(text: NonBlankString) -> Self {
        text.0
    }
}

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

impl fmt::Display for HfFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
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

impl TryFrom<String> for HfFile {
    type Error = HfFileError;

    fn try_from(id: String) -> Result<Self, Self::Error> {
        id.parse()
    }
}

impl From<HfFile> for String {
    fn from(file: HfFile) -> Self {
        file.0.to_string()
    }
}
