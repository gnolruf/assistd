//! What the readers of the XML inside an archive share.

use std::borrow::Cow;

use quick_xml::XmlVersion;
use quick_xml::events::BytesStart;

use crate::protocol::SkipReason;

impl From<quick_xml::Error> for SkipReason {
    fn from(error: quick_xml::Error) -> Self {
        Self::ExtractorFailed {
            message: error.to_string(),
        }
    }
}

/// The value of `element`'s attribute `name`, its character references resolved.
pub(crate) fn attribute(
    element: &BytesStart<'_>,
    name: &str,
) -> Result<Option<String>, quick_xml::Error> {
    element
        .try_get_attribute(name)?
        .map(|attribute| {
            attribute
                .normalized_value(XmlVersion::Implicit1_0)
                .map(Cow::into_owned)
        })
        .transpose()
}
