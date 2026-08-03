# Mathematics of Voice and Audio Pipelines

Building voice agents requires understanding the digital representation of sound and the metrics used to evaluate speech pipelines.

## 1. Audio Sample Rate and Quantization
Sound is an analog wave, which must be digitized through sampling. The data rate of an audio stream dictates the required bandwidth and buffer sizes.

The bit rate or data rate \( S \) (in bits per second) is calculated as:
$$ S = \text{samples/sec} \times \text{bit\_depth} \times \text{channels} $$

- **Samples/sec (Sample Rate):** How many times per second the analog wave is measured (e.g., 16,000 Hz or 16 kHz is standard for speech).
- **Bit depth:** The precision of each sample (e.g., 16-bit audio provides 65,536 possible amplitude values).
- **Channels:** 1 for mono, 2 for stereo. (Voice agents typically use 1 channel).

For standard voice agent audio (16kHz, 16-bit, Mono):
$$ S = 16,000 \times 16 \times 1 = 256,000 \text{ bits/sec} = 32 \text{ kB/sec} $$
A 20ms audio frame thus contains exactly 640 bytes of data.

## 2. Word Error Rate (WER)
Word Error Rate is the standard metric for evaluating Speech-to-Text (STT) accuracy. It behaves like the Levenshtein distance, but operates at the word level rather than the character level.

$$ \text{WER} = \frac{S + D + I}{N} $$

Where:
- \( S \) = Number of **Substitutions** (words incorrectly recognized).
- \( D \) = Number of **Deletions** (words spoken but missing from the transcript).
- \( I \) = Number of **Insertions** (words added to the transcript that were not spoken).
- \( N \) = Total number of words in the reference (ground truth) transcript.

A lower WER indicates a more accurate STT system. Note that because of insertions, WER can theoretically exceed 100%.

## 3. Energy-Based Voice Activity Detection (VAD)
The simplest way to detect speech (though modern systems use neural networks like Silero) is analyzing the energy of an audio frame.

For a frame containing \( N \) discrete audio samples \( x_1, x_2, \dots, x_N \), the average energy \( E \) is the mean of the squared amplitudes:
$$ E = \frac{1}{N} \sum_{i=1}^{N} (x_i)^2 $$

To detect speech, this energy is compared against a threshold \( \theta_{\text{vad}} \):
- If \( E > \theta_{\text{vad}} \), the frame is classified as **Speech**.
- If \( E \le \theta_{\text{vad}} \), the frame is classified as **Silence/Noise**.

In robust systems, this threshold adapts dynamically to the background noise floor to prevent false positives.
