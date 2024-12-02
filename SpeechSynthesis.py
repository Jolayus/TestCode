import os
import joblib
import numpy as np
import librosa
import soundfile as sf

# Function to synthesize a phoneme sequence into features using HMM models
def synthesize_phoneme_sequence(phoneme_sequence, model_dir, output_lengths):
    synthesized_features = []

    for idx, phoneme_name in enumerate(phoneme_sequence):
        # Load the HMM model and scaler for each phoneme
        model_file = os.path.join(model_dir, f"{phoneme_name}_hmm_model.pkl")
        scaler_file = os.path.join(model_dir, f"{phoneme_name}_scaler.pkl")

        try:
            hmm_model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
        except FileNotFoundError as e:
            print(f"Error loading model or scaler for {phoneme_name}: {e}")
            continue

        # Generate state sequence based on transition probabilities
        output_length = max(output_lengths[idx], 1)
        states = [np.random.choice(hmm_model.n_components, p=hmm_model.startprob_)]
        
        for _ in range(output_length - 1):
            next_state = np.random.choice(hmm_model.n_components, p=hmm_model.transmat_[states[-1]])
            states.append(next_state)

        # Generate features for each state
        phoneme_features = []
        for state in states:
            mean = hmm_model.means_[state]
            covar = hmm_model.covars_[state]

            # Ensure the covariance matrix is positive semi-definite
            covar_matrix = np.diag(covar) if covar.ndim == 1 else covar
            if not np.all(np.linalg.eigvals(covar_matrix) >= 0):
                covar_matrix += np.eye(covar_matrix.shape[0]) * 1e-6  # Stabilize

            feature = np.random.multivariate_normal(mean, covar_matrix)
            phoneme_features.append(feature)

        phoneme_features = np.array(phoneme_features)

        # Reverse the scaling
        if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
            scaled_features = scaler.inverse_transform(phoneme_features)
        else:
            scaled_features = phoneme_features  # Fall back if no scaler attributes

        synthesized_features.append(scaled_features)

    # Concatenate features from all phonemes in the sequence
    return np.vstack(synthesized_features)

# Function to convert generated features into audio using Griffin-Lim
def features_to_audio(features, sample_rate=16000, n_mfcc=13, n_delta=13, n_delta_delta=13):
    mfcc_features = features[:, :n_mfcc]

    # Reconstruct Mel spectrogram directly from MFCCs
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mfcc_features)
    audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sample_rate, n_iter=60)

    # Normalize audio to prevent clipping
    audio = audio / np.max(np.abs(audio))

    return audio

if __name__ == "__main__":
    try:
        # Load the average durations from file
        avg_durations = {}
        with open('average_phoneme_durations.txt', 'r') as f:
            for line in f:
                phoneme, duration = line.strip().split('\t')
                avg_durations[phoneme] = float(duration)
    except FileNotFoundError:
        print("Error: 'average_phoneme_durations.txt' file not found.")
        avg_durations = {}

    # Example phoneme sequence to synthesize
    phoneme_sequence = ["b", "a", "h", "a", "y"]  # Example sequence: "ako"
    model_dir = "hmm_models/"

    output_lengths = []
    for phoneme in phoneme_sequence:
        duration = avg_durations.get(phoneme, 100)  # Default to 100 ms if not in file
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
