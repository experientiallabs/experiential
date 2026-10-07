//! Measure an uploaded audio file's duration from its coded frames.
//!
//! A transcription is billed by the seconds the provider decodes, so its
//! reservation must bound them. Neither the upload size (low-bitrate Opus or
//! long silence carries far more seconds per byte than speech codecs) nor any
//! container timing (headers, timestamps, and granule positions are all
//! caller-authored metadata) is such a bound. The coded frames are: this
//! demuxes the file without decoding and sums each packet's duration as its
//! codec defines it:
//!
//! - Opus: the packet's TOC byte (RFC 6716 section 3.1) states its frame size
//!   and frame count;
//! - AAC: every frame counts as 2,048 samples (HE-AAC's size, the larger of the
//!   profiles) at the stream's sample rate;
//! - MP3 and FLAC in their own containers: the demuxer derives each packet's
//!   length from the frame header the decoder itself reads;
//! - PCM in WAV: the packet's sample count follows from its byte length.
//!
//! MP3, FLAC, or PCM wrapped in MP4 or Matroska would take its lengths from a
//! caller-authored sample table, so those pairings are refused.
//!
//! Any other codec (Vorbis, for one, whose packet sizes need its setup
//! header), a container that fails to demux, or a stream with no audio has no
//! trustworthy duration and is refused rather than reserved on a guess.

use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};

use symphonia::core::codecs::audio::well_known::{
    CODEC_ID_AAC, CODEC_ID_FLAC, CODEC_ID_MP3, CODEC_ID_OPUS, CODEC_ID_PCM_ALAW,
    CODEC_ID_PCM_F32LE, CODEC_ID_PCM_F64LE, CODEC_ID_PCM_MULAW, CODEC_ID_PCM_S16BE,
    CODEC_ID_PCM_S16LE, CODEC_ID_PCM_S24LE, CODEC_ID_PCM_S32LE, CODEC_ID_PCM_U8,
};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

/// Opus always counts samples at 48 kHz (RFC 6716).
const OPUS_SAMPLE_RATE: u64 = 48_000;

/// The most output samples one AAC frame decodes to: 1,024 for AAC-LC and 2,048
/// for HE-AAC (SBR doubles the rate). The container's profile signaling is
/// metadata, so every frame is counted at the larger size: the reservation may
/// cover up to twice an AAC-LC file's length, but never less than any profile.
const AAC_SAMPLES_PER_FRAME: u64 = 2_048;

/// How one codec's packet durations are derived.
enum Meter {
    /// Parse each packet's TOC byte; samples at 48 kHz.
    Opus,
    /// Count frames; samples at the stream's sample rate.
    Aac(u64),
    /// Trust the demuxer's frame-header-derived packet duration (time base ticks).
    FrameHeaders,
}

/// The audio's duration in milliseconds, or None when it cannot be measured.
///
/// `extension` is a probe hint (the upload's file extension), never trusted:
/// the probe identifies the container from its bytes. A demuxer panic on a
/// malformed file is caught and reported as unmeasurable.
pub(crate) fn duration_milli(bytes: Vec<u8>, extension: &str) -> Option<u64> {
    let hint_extension = extension.to_ascii_lowercase();
    catch_unwind(AssertUnwindSafe(move || measure(bytes, &hint_extension))).ok()?
}

fn measure(bytes: Vec<u8>, extension: &str) -> Option<u64> {
    let mut hint = Hint::new();
    hint.with_extension(extension);
    let stream = MediaSourceStream::new(
        Box::new(Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );
    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .ok()?;
    // A provider may decode any audio track of a multi-track container, and
    // only one could be measured; such uploads have no single duration.
    let audio_tracks = reader
        .tracks()
        .iter()
        .filter(|track| matches!(track.codec_params, Some(CodecParameters::Audio(_))))
        .count();
    if audio_tracks != 1 {
        return None;
    }
    let track = reader.first_track(TrackType::Audio)?;
    let track_id = track.id;
    let time_base = track.time_base;
    let Some(CodecParameters::Audio(params)) = track.codec_params.as_ref() else {
        return None;
    };
    // MP3, FLAC, and PCM packet lengths are frame-derived only in their own
    // containers, whose demuxers parse each frame (raw MP3 frame headers, FLAC
    // frame headers, WAV sample data). Wrapped in MP4 or Matroska the length
    // comes from a caller-authored sample table instead, so that pairing is
    // refused.
    let frame_parsed = matches!(
        reader.format_info().short_name,
        "mp1" | "mp2" | "mp3" | "flac" | "wave"
    );
    let meter = match params.codec {
        CODEC_ID_OPUS => Meter::Opus,
        CODEC_ID_AAC => Meter::Aac(u64::from(params.sample_rate.filter(|rate| *rate > 0)?)),
        CODEC_ID_MP3 | CODEC_ID_FLAC | CODEC_ID_PCM_S16LE | CODEC_ID_PCM_S16BE
        | CODEC_ID_PCM_S24LE | CODEC_ID_PCM_S32LE | CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F64LE
        | CODEC_ID_PCM_U8 | CODEC_ID_PCM_ALAW | CODEC_ID_PCM_MULAW
            if frame_parsed =>
        {
            Meter::FrameHeaders
        }
        _ => return None,
    };
    // Summed in nanoseconds so every meter shares one exact accumulator.
    let mut total_nanos: u128 = 0;
    loop {
        let packet = match reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(_) => return None,
        };
        if packet.track_id != track_id {
            continue;
        }
        let nanos = match meter {
            Meter::Opus => {
                u128::from(opus_packet_samples(&packet.data)?) * 1_000_000_000
                    / u128::from(OPUS_SAMPLE_RATE)
            }
            Meter::Aac(rate) => {
                u128::from(AAC_SAMPLES_PER_FRAME) * 1_000_000_000 / u128::from(rate)
            }
            Meter::FrameHeaders => {
                let base = time_base?;
                u128::from(packet.dur.get()) * u128::from(base.numer.get()) * 1_000_000_000
                    / u128::from(base.denom.get())
            }
        };
        total_nanos = total_nanos.checked_add(nanos)?;
    }
    // A sub-millisecond total rounds to zero: as unmeasurable as no packets.
    u64::try_from(total_nanos / 1_000_000)
        .ok()
        .filter(|milli| *milli > 0)
}

/// Samples (at 48 kHz) one Opus packet decodes to, from its TOC byte.
///
/// The frame size follows from the configuration number (RFC 6716 table 2)
/// and the frame count from the code: one frame, two, or an explicit count in
/// the second byte. A packet longer than the 120 ms the format allows is
/// malformed.
fn opus_packet_samples(packet: &[u8]) -> Option<u64> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let frame_samples: u64 = match config {
        0..=11 => [480, 960, 1_920, 2_880][usize::from(config % 4)],
        12..=15 => [480, 960][usize::from(config % 2)],
        _ => [120, 240, 480, 960][usize::from(config % 4)],
    };
    let frames: u64 = match toc & 0x03 {
        0 => 1,
        1 | 2 => 2,
        _ => u64::from(*packet.get(1)? & 0x3F),
    };
    let samples = frames * frame_samples;
    (frames > 0 && samples <= 5_760).then_some(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(seconds: u32) -> Vec<u8> {
        let data_bytes = 32_000 * seconds;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&32_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, data_bytes as usize));
        bytes
    }

    #[test]
    fn measures_a_wav_from_its_samples() {
        assert_eq!(duration_milli(wav(2), "wav"), Some(2_000));
    }

    #[test]
    fn opus_toc_states_each_packets_samples() {
        // config 1 (SILK 20 ms), code 0: one 960-sample frame.
        assert_eq!(opus_packet_samples(&[1 << 3]), Some(960));
        // config 31 (CELT 20 ms), code 1: two frames.
        assert_eq!(opus_packet_samples(&[(31 << 3) | 1]), Some(1_920));
        // config 16 (CELT 2.5 ms), code 3 with 48 frames = 120 ms, the maximum.
        assert_eq!(opus_packet_samples(&[(16 << 3) | 3, 48]), Some(5_760));
        // 3 x 60 ms exceeds 120 ms: malformed. Zero frames and empty packets too.
        assert_eq!(opus_packet_samples(&[(3 << 3) | 3, 3]), None);
        assert_eq!(opus_packet_samples(&[(16 << 3) | 3, 0]), None);
        assert_eq!(opus_packet_samples(&[]), None);
    }

    #[test]
    fn unmeasurable_or_garbage_uploads_have_no_duration() {
        assert_eq!(duration_milli(b"not audio at all".to_vec(), "mp3"), None);
        assert_eq!(duration_milli(Vec::new(), "wav"), None);
        let mut truncated = wav(1);
        truncated.truncate(30);
        assert_eq!(duration_milli(truncated, "wav"), None);
    }
}
