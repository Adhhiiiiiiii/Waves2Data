import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import gradio as gr

def convert_to_wav(input_file, output_file):
    data, sample_rate = librosa.load(input_file, sr=None)
    sf.write(output_file, data, sample_rate)

def analyze_audio(file_path):
    # Convert to WAV if not already
    base_name, ext = os.path.splitext(file_path)
    if ext.lower() != '.wav':
        wav_file = base_name + '.wav'
        convert_to_wav(file_path, wav_file)
        file_path = wav_file

    # Load the audio file
    try:
        audio_data, sample_rate = librosa.load(file_path)
    except Exception as e:
        return f"Error: {e}"

    # Basic file information
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    info_text = f"File Information:\nDuration: {duration:.2f} seconds\nSample Rate: {sample_rate} Hz"

    # Waveform visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title('Waveform')
    plt.tight_layout()

    # Spectrogram
    plt.subplot(3, 1, 2)
    spectrogram = np.abs(librosa.stft(audio_data))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                             sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    # Power Spectrum
    plt.subplot(3, 2, 5)
    power_spectrum = np.mean(spectrogram, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    plt.plot(freqs, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.tight_layout()

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    plt.subplot(3, 2, 6)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    plt.tight_layout()

    # Save the plot as an image
    plot_file = 'audio_analysis.png'
    plt.savefig(plot_file)
    plt.close()  # Close the plot to free memory

    return info_text, plot_file

# Gradio interface
iface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Audio Info"), gr.Image(type="filepath", label="Analysis Plot")],
    title="Audio Analysis Tool",
    description="Upload an audio file to analyze its waveform, spectrogram, power spectrum, and MFCCs."
)

if __name__ == "__main__":
    iface.launch(share=True)