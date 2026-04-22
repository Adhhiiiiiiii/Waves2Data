import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf
import webrtcvad
import torch
import torchaudio
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import tempfile

# =========================
# Speaker Embedding Model (SAFE INIT)
# =========================
model = None
EMBEDDING_MODE = "mfcc"

try:
    if hasattr(torchaudio, "pipelines") and hasattr(torchaudio.pipelines, "SUPERB_XVECTOR"):
        bundle = torchaudio.pipelines.SUPERB_XVECTOR
        model = bundle.get_model()
        EMBEDDING_MODE = "xvector"
        print("Using XVector model")
    else:
        print("SUPERB_XVECTOR not available, using MFCC fallback")
except Exception as e:
    print("Model load failed:", e)


def extract_embedding(waveform, sr):
    # Ensure 16kHz
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    # Use xvector if available
    if EMBEDDING_MODE == "xvector" and model is not None:
        waveform_t = torch.tensor(waveform).unsqueeze(0)
        with torch.no_grad():
            emb = model(waveform_t)
        return emb.squeeze().numpy()

    # Fallback: MFCC-based embedding
    mfcc = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13)
    return np.mean(mfcc, axis=1)


# =========================
# Audio Processor (REAL-TIME)
# =========================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.vad = webrtcvad.Vad(2)
        self.sample_rate = 16000

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0

        # 🔥 CRITICAL FIX: WebRTC gives ~48kHz → resample to 16kHz
        try:
            audio = librosa.resample(audio, orig_sr=48000, target_sr=16000)
        except:
            pass

        # -------- VAD --------
        pcm16 = (audio * 32768).astype(np.int16).tobytes()
        is_speech = False
        try:
            is_speech = self.vad.is_speech(pcm16, 16000)
        except:
            pass

        # -------- Noise Profiling --------
        noise_level = float(np.mean(np.abs(audio)))

        # -------- Speaker Embedding --------
        try:
            embedding = extract_embedding(audio, 16000)
            speaker_id = float(np.linalg.norm(embedding))
        except:
            speaker_id = 0.0

        # Store in session
        st.session_state["waveform"] = audio[:1000]
        st.session_state["vad"] = is_speech
        st.session_state["noise"] = noise_level
        st.session_state["speaker"] = speaker_id

        return frame


# =========================
# Visualization
# =========================
def plot_waveform(audio):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(audio)
    ax.set_title("Real-time Waveform")
    return fig


# =========================
# Streamlit UI
# =========================
st.title("Real-Time Audio Intelligence Dashboard")

# Init session state
for key in ["waveform", "vad", "noise", "speaker"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------- WebRTC Stream --------
webrtc_streamer(
    key="audio",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# -------- Live Metrics --------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Voice Activity", "Speech" if st.session_state["vad"] else "Silence")

with col2:
    if st.session_state["noise"] is not None:
        st.metric("Noise Level", f"{st.session_state['noise']:.4f}")

with col3:
    if st.session_state["speaker"] is not None:
        st.metric("Speaker Fingerprint", f"{st.session_state['speaker']:.2f}")

# -------- Waveform --------
if st.session_state["waveform"] is not None:
    fig = plot_waveform(st.session_state["waveform"])
    st.pyplot(fig)


# =========================
# FILE UPLOAD ANALYSIS
# =========================
st.divider()
st.subheader("Upload Audio for Deep Analysis")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    y, sr = librosa.load(path)

    duration = librosa.get_duration(y=y, sr=sr)
    st.text(f"Duration: {duration:.2f}s | Sample Rate: {sr}")

    # Spectrogram
    D = np.abs(librosa.stft(y))
    fig, ax = plt.subplots()
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max),
        sr=sr,
        x_axis='time',
        y_axis='log',
        ax=ax
    )
    st.pyplot(fig)

    # Speaker embedding
    emb = extract_embedding(y, sr)
    st.write("Speaker Fingerprint Vector (first 10 values):")
    st.write(emb[:10])
