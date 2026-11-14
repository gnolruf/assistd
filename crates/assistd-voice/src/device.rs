//! cpal device naming shared by capture and playback.

use cpal::traits::DeviceTrait;

/// The name a device is configured by: its driver name, else its description.
pub(crate) fn device_name(device: &cpal::Device) -> String {
    device.description().map_or_else(
        |_| "<unknown>".to_string(),
        |description| {
            description
                .driver()
                .map_or_else(|| description.name().to_string(), str::to_string)
        },
    )
}
