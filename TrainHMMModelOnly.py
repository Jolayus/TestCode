import os
import joblib
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# --- HMM Training ---
class TrainHMMModel:
    def __init__(self, n_components=5, n_iter=750, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.n_iter, random_state=random_state)

    def train(self, X, patience=50, tolerance=1e-3):
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input features contain NaN or infinite values.")

        best_log_likelihood = -np.inf
        no_improvement_count = 0

        for i in range(self.n_iter):
            self.model.fit(X)

            # Check for NaN or inf in model parameters
            if np.any(np.isnan(self.model.startprob_)) or np.any(np.isinf(self.model.startprob_)):
                print("Warning: Start probabilities contain NaN or inf after fitting.")
                return

            current_log_likelihood = self.model.score(X)
            print(f"Iteration {i+1}/{self.n_iter}, Log-Likelihood: {current_log_likelihood}")

            # Check for improvement
            if current_log_likelihood - best_log_likelihood > tolerance:
                best_log_likelihood = current_log_likelihood
                no_improvement_count = 0  # Reset counter if we have an improvement
            else:
                no_improvement_count += 1
            
            # Check if we should stop
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)
        
# --- Main Training Process ---
if __name__ == "__main__":
    
    phoneme_features_dir = "phoneme_features_folder/"
    save_model_dir = "hmm_models/"
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Loop through phoneme pickle files
    for phoneme_file in os.listdir(phoneme_features_dir):
        if phoneme_file.endswith(".pkl"):
            phoneme_name = phoneme_file.split(".")[0]  # Extract the phoneme name from the file
            
            # Load phoneme features
            phoneme_features_path = os.path.join(phoneme_features_dir, phoneme_file)
            with open(phoneme_features_path, 'rb') as f:
                phoneme_features = joblib.load(f)  # List of dictionaries
            
            # Initialize empty lists to store concatenated features across occurrences
            all_mfcc_0th = []  # New list for 0th MFCC
            all_mfcc = []
            all_delta = []
            all_delta_delta = []
            all_energy = []
            all_f0 = []
            all_aperiodicity = []

            # Loop through each dictionary (representing an occurrence of the phoneme)
            for occurrence_num, occurrence in enumerate(phoneme_features, start=1):
                if not occurrence:
                    print(f"Skipping empty occurrence {occurrence_num} in phoneme: {phoneme_name}")
                    continue
                
                mfcc_0th = occurrence.get("mfcc_0th")  # 0th MFCC
                mfcc_features = occurrence.get("mfcc_features")
                delta_features = occurrence.get("delta_features")
                delta_delta_features = occurrence.get("delta_delta_features")
                energy_features = occurrence.get("energy_features")
                f0_features = occurrence.get("f0_features")
                aperiodicity_features = occurrence.get("aperiodicity_features")

                # Ensure all features exist before processing
                if (mfcc_features is None or delta_features is None or delta_delta_features is None 
                        or energy_features is None or f0_features is None or aperiodicity_features is None or mfcc_0th is None):
                    print(f"Some features are missing in phoneme: {phoneme_name}")
                    continue

                # Append to respective lists
                all_mfcc_0th.append(mfcc_0th)  # Append 0th MFCC
                all_mfcc.append(mfcc_features)
                all_delta.append(delta_features)
                all_delta_delta.append(delta_delta_features)
                all_energy.append(energy_features)
                all_f0.append(f0_features)
                all_aperiodicity.append(aperiodicity_features)

            # Skip phoneme if no valid occurrences were found
            if not all_mfcc:
                print(f"No valid data found for phoneme: {phoneme_name}, skipping.")
                continue
            
            # Concatenate features across occurrences (combine the lists)
            mfcc_0th_features = np.concatenate(all_mfcc_0th, axis=0)  # Concatenate 0th MFCC
            mfcc_features = np.concatenate(all_mfcc, axis=0)
            delta_features = np.concatenate(all_delta, axis=0)
            delta_delta_features = np.concatenate(all_delta_delta, axis=0)
            energy_features = np.concatenate(all_energy, axis=0)
            f0_features = np.concatenate(all_f0, axis=0)
            aperiodicity_features = np.concatenate(all_aperiodicity, axis=0)

            # Concatenate features along the last axis (time frames)
            X_phoneme = np.concatenate([mfcc_0th_features, mfcc_features, delta_features, delta_delta_features, 
                                        energy_features, f0_features, aperiodicity_features], axis=1)

            # Normalize features using StandardScaler
            scaler = StandardScaler()
            X_phoneme = scaler.fit_transform(X_phoneme)

            # Save the scaler for later use
            scaler_file = os.path.join(save_model_dir, f"{phoneme_name}_scaler.pkl")
            joblib.dump(scaler, scaler_file)
            
            # Train HMM model for this phoneme
            print(f"Training HMM model for phoneme: {phoneme_name}")
            hmm_trainer = TrainHMMModel(n_components=5, n_iter=750, random_state=64)
            hmm_trainer.train(X_phoneme, patience=20, tolerance=1e-3)

            # Save the trained HMM model
            model_file = os.path.join(save_model_dir, f"{phoneme_name}_hmm_model.pkl")
            hmm_trainer.save_model(model_file)
            print(f"Saved HMM model for phoneme: {phoneme_name}")

    print("All phoneme models have been trained and saved.")
