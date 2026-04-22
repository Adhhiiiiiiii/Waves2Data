import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import tempfile


# =========================
# App Header / Branding
# =========================
st.set_page_config(page_title="📈 Waves2Data", layout="centered")

st.title("📈 Waves2Data")
st.markdown("""
Wave2Data is an audio analysis tool that converts raw sound into meaningful visual and statistical representations.

**What you can do:**
- Upload or record audio directly from your browser
- Visualize waveform and frequency components
- Analyze spectral patterns and MFCC features
""")

st.divider()


# =========================
# Core Functions
# =========================
def convert_to_wav(input_file, output_file):
    data, sample_rate = librosa.load(input_file, sr=None)
    sf.write(output_file, data, sample_rate)


def analyze_audio(file_path):
    base_name, ext = os.path.splitext(file_path)

    if ext.lower() != '.wav':
        wav_file = base_name + '.wav'
        convert_to_wav(file_path, wav_file)
        file_path = wav_file

    try:
        audio_data, sample_rate = librosa.load(file_path)
    except Exception as e:
        return f"Error: {e}", None

    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    info_text = f"""
File Information:
- Duration: {duration:.2f} seconds
- Sample Rate: {sample_rate} Hz
"""

    plt.figure(figsize=(12, 6))

    # Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title('Waveform')

    # Spectrogram
    plt.subplot(3, 1, 2)
    spectrogram = np.abs(librosa.stft(audio_data))
    librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram, ref=np.max),
        sr=sample_rate, x_axis='time', y_axis='log'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Power Spectrum
    plt.subplot(3, 2, 5)
    power_spectrum = np.mean(spectrogram, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    plt.plot(freqs, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectrum')

    # MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    plt.subplot(3, 2, 6)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    plt.tight_layout()

    plot_file = 'audio_analysis.png'
    plt.savefig(plot_file)
    plt.close()

    return info_text, plot_file


# =========================
# Input Section
# =========================
st.subheader("🎧 Input Audio")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "flac", "ogg"]
)

mic_audio = st.audio_input("Or record using microphone")

file_path = None

if mic_audio is not None:
    st.audio(mic_audio)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(mic_audio.read())
        file_path = tmp_file.name

elif uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name


# =========================
# Analysis Section
# =========================
if file_path:
    st.divider()
    st.subheader("📊 Analysis Results")

    info, plot_path = analyze_audio(file_path)

    st.text(info)

    if plot_path:
        st.image(plot_path, caption="Audio Analysis", use_container_width=True)


# =========================
# Footer
# =========================

st.markdown("""
---
### 🎶 About Waves2Data

Wave2Data transforms raw audio signals into structured data insights by applying digital signal processing techniques.

**It performs:**
- Time-domain analysis (Waveform)
- Frequency-domain analysis (Spectrogram & Power Spectrum)
- Feature extraction (MFCC – widely used in speech and audio recognition)

This tool is useful for:
- Audio research
- Signal processing learning
- Speech and sound pattern analysis
""")
