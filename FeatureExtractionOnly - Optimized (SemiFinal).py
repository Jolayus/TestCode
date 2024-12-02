import os
import librosa
import joblib
import logging
import numpy as np
import pyworld as pw

from scipy.fftpack import dct

from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Extraction Class ---
class AudioFeatureExtractor:
    def __init__(self, sample_rate, frame_duration, frame_overlap, num_filters, num_ceps):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_overlap = frame_overlap
        self.num_filters = num_filters
        self.num_ceps = num_ceps
        
    def frame_audio(self, signal, frame_size=None, hop_length=None):
        if signal is None or len(signal) == 0:
            logging.warning("Audio data is empty or None.")
            return np.zeros((1, frame_size))

        if frame_size is None:
            frame_size = int(self.frame_duration * self.sample_rate)
        if hop_length is None:
            hop_length = int(frame_size * (1 - self.frame_overlap))

        # Check if the signal is too short
        if len(signal) < frame_size:
            padding = np.zeros(frame_size - len(signal))
            signal = np.concatenate((signal, padding))

        frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_length).T
        if frames.shape[1] == 0:
            logging.warning("Generated empty frames from audio data.")
            return np.zeros((1, frame_size))
        return frames

    def apply_hamming_window(self, frames):
        if frames is None or frames.size == 0:
            logging.warning("Empty frames received for window application.")
            return None
        hamming_window = np.hamming(frames.shape[1])
        return frames * hamming_window

    def hz_to_mel(self, frequency):
        return 2595 * np.log10(1 + frequency / 700)

    def hz_to_bin(self, frequency, fft_size):
        return int(frequency * fft_size / self.sample_rate)

    def bin_to_hz(self, bin_index, fft_size):
        return bin_index * self.sample_rate / fft_size

    def mel_to_hz(self, mel_value):
        return 700 * (10 ** (mel_value / 2595) - 1)

    def mel_filterbank(self, fft_size):
        mel_filter_bank = np.zeros((self.num_filters, fft_size // 2 + 1))
        mel_points = np.linspace(self.hz_to_mel(0), self.hz_to_mel(self.sample_rate / 2), self.num_filters + 2)

        for i in range(1, self.num_filters + 1):
            lower, center, upper = mel_points[i - 1], mel_points[i], mel_points[i + 1]
            lower_bin = int(self.hz_to_bin(self.mel_to_hz(lower), fft_size))
            center_bin = int(self.hz_to_bin(self.mel_to_hz(center), fft_size))
            upper_bin = int(self.hz_to_bin(self.mel_to_hz(upper), fft_size))

            mel_filter_bank[i - 1, lower_bin:center_bin] = (np.arange(lower_bin, center_bin) - lower_bin) / (center_bin - lower_bin)
            mel_filter_bank[i - 1, center_bin:upper_bin] = (upper_bin - np.arange(center_bin, upper_bin)) / (upper_bin - center_bin)

        return mel_filter_bank

    def apply_fft_to_windowed_frames(self, frames, zero_padding=True):
        if zero_padding:
            frames = np.pad(frames, ((0, 0), (0, frames.shape[1] // 2)), mode='constant')
        return np.fft.rfft(frames)

    def compute_delta_coefficients(self, features, order=1):
        delta_features = librosa.feature.delta(features, order=order)
        return delta_features

    def compute_mfcc_with_dct(self, dft_frames, windowed_frames):
        if dft_frames.size == 0 or windowed_frames.size == 0:
            raise ValueError("Error: Invalid DFT frames or windowed frames.")

        # Create mel filterbank
        mel_filter_bank = self.mel_filterbank(len(dft_frames[0]))

        mfcc_features = []

        for dft_frame in dft_frames:
            if len(dft_frame) == 0:
                print("Warning: Empty DFT frame encountered.")
                continue

            # Calculate mel spectrum and log mel spectrum
            mel_spectrum = np.dot(mel_filter_bank, np.abs(dft_frame[:len(mel_filter_bank[0])]) ** 2)
            log_mel_spectrum = np.log(mel_spectrum + 1e-10)

            # Compute MFCCs using DCT
            mfcc = dct(log_mel_spectrum, type=2, axis=-1, norm='ortho')[:self.num_ceps]
            mfcc_features.append(mfcc)

        if not mfcc_features:
            print("Error: No valid MFCC features.")
            return []

        # Subtract mean energy for normalization
        mean_energy = np.mean(self.compute_energy(windowed_frames))
        mfcc_features = np.array(mfcc_features) - mean_energy

        return np.array(mfcc_features)

    def compute_energy(self, frames):
        energy = np.sum(frames**2, axis=1)
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)  # Stable normalization
        return energy
    
    def compute_f0_aperiodicity(self, audio_data, sr):
        _f0, t = pw.dio(audio_data, sr)
        f0 = pw.stonemask(audio_data, _f0, t, sr)
        ap = pw.d4c(audio_data, f0, t, sr)
        return f0, ap

    def extract_features_from_audio(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            print("Warning: Audio data is empty or invalid. Skipping processing.")
            return {}  # or continue to the next iteration, depending on your context

        frames = self.frame_audio(audio_data)
        print(f"Frames shape: {frames.shape}")

        if frames.shape[0] == 0 or frames.shape[1] == 0:
            print("Frames content:", frames)
            raise ValueError("Error: Invalid frames after framing.")

        if np.any(frames == 0):
            print("Frames contain zeros. Min:", np.min(frames), "Max:", np.max(frames))

        windowed_frames = self.apply_hamming_window(frames)
        dft_frames = self.apply_fft_to_windowed_frames(windowed_frames)

        if dft_frames.size == 0:
            raise ValueError("Error: Invalid DFT frames.")

        mfcc_dct_features = self.compute_mfcc_with_dct(dft_frames, windowed_frames)

        if mfcc_dct_features.size == 0:
            print("Error: MFCC DCT features are empty or invalid.")
            return {}

        delta_features = self.compute_delta_coefficients(mfcc_dct_features, order=1)
        delta_delta_features = self.compute_delta_coefficients(delta_features, order=2)

        energy_features = self.compute_energy(windowed_frames)

        # Initialize lists for F0 and aperiodicity
        f0_features = []
        aperiodicity_features = []

        # Use pyworld to extract F0 and aperiodicity
        # Convert each frame to a continuous signal (since pyworld processes the whole signal)
        # Use the compute_f0_aperiodicity function to extract F0 and aperiodicity
        for frame in windowed_frames:
            frame = frame.astype(np.float64)  # PyWorld requires float64
            
            # Compute F0 and aperiodicity using the function
            f0, ap = self.compute_f0_aperiodicity(frame, self.sample_rate)
            
            # Append the mean values per frame to align with other features (1 value per frame)
            f0_features.append(np.mean(f0))
            aperiodicity_features.append(np.mean(ap))

        return {
            "mfcc_features": mfcc_dct_features,                                         # MFCC (1st to 13th coefficients)
            "delta_features": delta_features,                                           # Delta features (1st to 13th MFCC)
            "delta_delta_features": delta_delta_features,                               # Delta-Delta features (1st to 13th MFCC)
            "energy_features": energy_features[:, np.newaxis],                          # Energy feature (1 per frame)
            "f0_features": np.array(f0_features)[:, np.newaxis],                        # F0 feature (1 per frame)
            "aperiodicity_features": np.array(aperiodicity_features)[:, np.newaxis]     # Aperiodicity feature (1 per frame)
        }

# Helper functions
def load_audio_file(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None
    audio_data, sr = librosa.load(filepath, sr=None)
    if audio_data is None or len(audio_data) == 0:
        logging.warning(f"Empty or invalid audio data in file: {filepath}")
        return None
    return audio_data, sr

def load_alignment_file(alignment_file):
    with open(alignment_file, "r") as f:
        alignments = []
        for line in f.readlines():
            parts = line.strip().split('\t')
            phoneme = parts[0]
            start = float(parts[1])
            end = float(parts[2])
            alignments.append((phoneme, start, end))
    return alignments

def time_to_samples(start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return start_sample, end_sample

def extract_audio_segment(audio_data, start_sample, end_sample):
    return audio_data[start_sample:end_sample]

# --- Main Code ---
if __name__ == "__main__":
    
    # Initialize feature extraction map for phonemes
    phoneme_features_map = defaultdict(list)
    
    # Initialize the feature extractor
    feature_extractor = AudioFeatureExtractor(
        sample_rate=16000, 
        frame_duration=0.025, 
        frame_overlap=0.5, 
        num_filters=26, 
        num_ceps=13
    )

    # Loop through all phoneme folders
    phoneme_dir = "Phonemes"
    for folder_name in os.listdir(phoneme_dir):
        folder_path = os.path.join(phoneme_dir, folder_name)
        
        # Check if directory exists before processing
        if not os.path.exists(folder_path):
            print(f"Error: Directory {folder_path} does not exist.")
        
        if os.path.isdir(folder_path):
            for phoneme_file in os.listdir(folder_path):
                if phoneme_file.endswith('.wav'):
                    phoneme_file_path = os.path.join(folder_path, phoneme_file)
                    
                    print(phoneme_file_path)
                    
                    # Load the phoneme audio file
                    phoneme_data, phoneme_sample_rate = load_audio_file(phoneme_file_path)
                    
                    if phoneme_data is not None:
                        # Extract features from the phoneme audio
                        phoneme_features = feature_extractor.extract_features_from_audio(phoneme_data)
                        
                        # Get phoneme from filename (e.g., "a.wav" -> "a")
                        phoneme = phoneme_file.split(".")[0]
                        
                        # Store features in the phoneme features map
                        phoneme_features_map[phoneme].append(phoneme_features)
                    
    # Loop through the dataset and alignment directories
    dataset_dir = "Audio WAV Files"
    alignment_dir = "Alignment"
    for dataset_folder in os.listdir(dataset_dir):
        dataset_folder_path = os.path.join(dataset_dir, dataset_folder)
        
        if not os.path.exists(dataset_folder_path):
            print(f"Error: Directory {dataset_folder_path} does not exist.")
        
        alignment_folder_path = os.path.join(alignment_dir, dataset_folder)
        
        if not os.path.exists(alignment_folder_path):
            print(f"Error: Directory {alignment_folder_path} does not exist.")
        
        if os.path.isdir(dataset_folder_path) and os.path.isdir(alignment_folder_path):
            for audio_file in os.listdir(dataset_folder_path):
                if audio_file.endswith('.wav'):
                    # Load the WAV file
                    audio_file_path = os.path.join(dataset_folder_path, audio_file)
                    print(audio_file_path)
                    audio_data, sample_rate = load_audio_file(audio_file_path)

                    if audio_data is not None:
                        # Find the corresponding alignment folder for this audio file
                        alignment_subfolder_path = os.path.join(alignment_folder_path, audio_file.split(".")[0])

                        if os.path.isdir(alignment_subfolder_path):
                            # Loop through all alignment files for the audio file
                            for alignment_file in os.listdir(alignment_subfolder_path):
                                if alignment_file.endswith('.txt'):
                                    # Load the alignment file
                                    alignment_file_path = os.path.join(alignment_subfolder_path, alignment_file)
                                    print(alignment_file_path)
                                    alignments = load_alignment_file(alignment_file_path)
                                    
                                    # Loop through alignments and extract features
                                    for i, (phoneme, start_time, end_time) in enumerate(alignments):
                                        start_sample, end_sample = time_to_samples(start_time, end_time, sample_rate)
                                        
                                        # Extract the phoneme audio segment
                                        audio_segment = extract_audio_segment(audio_data, start_sample, end_sample)

                                        # Extract features from the audio segment
                                        features = feature_extractor.extract_features_from_audio(audio_segment)
                                        
                                        # Append features to the phoneme in the phoneme features map
                                        phoneme_features_map[phoneme].append(features)

# create a folder to store the pkl files
folder_name = 'phoneme_features_folder'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# loop over phoneme features and dump each one into a separate pkl file
for phoneme, features in phoneme_features_map.items():
    filename = f'{phoneme}.pkl'
    filepath = os.path.join(folder_name, filename)
    joblib.dump(features, filepath)