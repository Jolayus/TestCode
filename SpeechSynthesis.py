import os
import joblib
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Function to synthesize a phoneme sequence into features using HMM models
def synthesize_phoneme_sequence(phoneme_sequence, model_dir, output_lengths):
    synthesized_features = []

    for idx, phoneme_name in enumerate(phoneme_sequence):
        # Load the HMM model and scaler for each phoneme
        model_file = os.path.join(model_dir, f"{phoneme_name}_hmm_model.pkl")
        scaler_file = os.path.join(model_dir, f"{phoneme_name}_scaler.pkl")

        hmm_model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        # Ensure output length is at least 1 to avoid empty feature generation
        output_length = max(output_lengths[idx], 1)

        # Generate the state sequences
        n_components = hmm_model.n_components
        states = np.random.choice(n_components, output_length, p=hmm_model.startprob_)

        # Generate features for each state
        phoneme_features = []
        for state in states:
            mean = hmm_model.means_[state]
            covar = hmm_model.covars_[state]
            feature = np.random.multivariate_normal(mean, covar)
            phoneme_features.append(feature)

        phoneme_features = np.array(phoneme_features)

        # Reverse the scaling
        scaled_features = phoneme_features * scaler.scale_ + scaler.mean_
        synthesized_features.append(scaled_features)

    # Concatenate features from all phonemes in the sequence
    return np.vstack(synthesized_features)

# Function to convert generated features into audio using Griffin-Lim
def features_to_audio(features, sample_rate=16000, n_mfcc=13, n_delta=13, n_delta_delta=13, n_energy=1):
    # Extract individual features from the synthesized features
    mfcc_features = features[:, :n_mfcc]  # MFCCs
    delta_features = features[:, n_mfcc:n_mfcc+n_delta]  # Delta
    delta_delta_features = features[:, n_mfcc+n_delta:n_mfcc+n_delta+n_delta_delta]  # Delta-Delta
    energy_features = features[:, n_mfcc+n_delta+n_delta_delta:]  # Energy

    # Combine MFCC, delta, and delta-delta features for synthesis (optional, but depends on usage)
    combined_features = np.hstack((mfcc_features, delta_features, delta_delta_features))

    # Reconstruct Mel spectrogram from MFCCs
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mfcc_features)

    # Convert the Mel spectrogram to a linear spectrogram (STFT)
    linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sample_rate)

    # Reconstruct the audio using the Griffin-Lim algorithm
    audio = librosa.griffinlim(linear_spectrogram, n_iter=32, hop_length=512)

    # Adjust audio using energy scaling
    audio *= np.mean(energy_features)  # Scale by the average energy

    return audio

if __name__ == "__main__":
    # Load the average durations from file
    with open('average_phoneme_durations.txt', 'r') as f:
        avg_durations = {}
        for line in f:
            phoneme, duration = line.strip().split('\t')
            avg_durations[phoneme] = float(duration)
            
    # Example phoneme sequence to synthesize
    phoneme_sequence = ["a", "k", "o"]  # Example sequence: "ako" (meaning "I" in Filipino)
    model_dir = "hmm_models/"

    # Calculate the output lengths based on the average durations
    output_lengths = []
    for phoneme in phoneme_sequence:
        duration = avg_durations[phoneme]
        # Convert duration from milliseconds to frames (assuming 16 kHz sample rate)
        output_length = int(duration / 1000 * 16000 / 512)
        output_lengths.append(output_length)
    
    # Synthesize features for the phoneme sequence
    synthesized_features = synthesize_phoneme_sequence(phoneme_sequence, model_dir, output_lengths)
    print(f"Synthesized features for phoneme sequence {phoneme_sequence}: {synthesized_features.shape}")

    # Convert synthesized features into audio
    synthesized_audio = features_to_audio(synthesized_features)

    # Save the synthesized audio
    output_file = "synthesized_speech.wav"
    sf.write(output_file, synthesized_audio, 16000)
    print(f"Synthesized speech saved as '{output_file}' for phoneme sequence {phoneme_sequence}")
