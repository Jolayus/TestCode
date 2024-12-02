import os
import joblib
import logging
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- HMM Training ---
class TrainHMMModel:
    def __init__(self, n_components, n_iter, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.n_iter, random_state=random_state, init_params="stmc")
        
    def check_nan_inf_in_model(self):
        parameters = {
            "startprob_": self.model.startprob_,
            "transmat_": self.model.transmat_,
            "means_": getattr(self.model, "means_", None),
            "covars_": getattr(self.model, "covars_", None),
        }
        for param_name, param in parameters.items():
            if param is not None and (np.any(np.isnan(param)) or np.any(np.isinf(param))):
                logger.error(f"{param_name} contains NaN or infinite values.")  # Log the message


    def train(self, X, patience=50):
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.error("Input features contain NaN or infinite values.")
        
        # Adaptive tolerance
        best_log_likelihood = -np.inf
        no_improvement_count = 0
        initial_tolerance = 1e-2
        final_tolerance = 1e-4
        adaptive_tolerance = lambda i: initial_tolerance * (1 - i / self.n_iter) + final_tolerance * (i / self.n_iter)

        for i in range(self.n_iter):
            current_tolerance = adaptive_tolerance(i)

            self.model.fit(X)

            # Check for NaN or inf in model parameters
            self.check_nan_inf_in_model()

            current_log_likelihood = self.model.score(X)
            
            logger.info(f"Iteration {i+1}/{self.n_iter}, Log-Likelihood: {current_log_likelihood}")

            if current_log_likelihood - best_log_likelihood > current_tolerance:
                best_log_likelihood = current_log_likelihood
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                logger.info(f"Early stopping triggered at iteration {i+1}")
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
            all_mfcc = []
            all_delta = []
            all_delta_delta = []
            all_energy = []
            all_f0 = []
            all_aperiodicity = []

            # Loop through each dictionary (representing an occurrence of the phoneme)
            for occurrence_num, occurrence in enumerate(phoneme_features, start=1):
                if not occurrence:
                    logger.warning(f"Skipping empty occurrence {occurrence_num} in phoneme: {phoneme_name}")
                    continue
                
                mfcc_features = occurrence.get("mfcc_features")
                delta_features = occurrence.get("delta_features")
                delta_delta_features = occurrence.get("delta_delta_features")
                energy_features = occurrence.get("energy_features")
                f0_features = occurrence.get("f0_features")
                aperiodicity_features = occurrence.get("aperiodicity_features")

                # Ensure all features exist before processing
                if (mfcc_features is None or delta_features is None or delta_delta_features is None 
                        or energy_features is None or f0_features is None or aperiodicity_features is None):
                    logger.warning(f"Some features are missing in phoneme: {phoneme_name}, skipping occurrence.")
                    continue

                # Append to respective lists
                all_mfcc.append(mfcc_features)
                all_delta.append(delta_features)
                all_delta_delta.append(delta_delta_features)
                all_energy.append(energy_features)
                all_f0.append(f0_features)
                all_aperiodicity.append(aperiodicity_features)

            # Skip phoneme if no valid occurrences were found
            if not all_mfcc:
                logger.warning(f"No valid data found for phoneme: {phoneme_name}, skipping.")
                continue
            
            # Concatenate features across occurrences (combine the lists)
            mfcc_features = np.concatenate(all_mfcc, axis=0)
            delta_features = np.concatenate(all_delta, axis=0)
            delta_delta_features = np.concatenate(all_delta_delta, axis=0)
            energy_features = np.concatenate(all_energy, axis=0)
            f0_features = np.concatenate(all_f0, axis=0)
            aperiodicity_features = np.concatenate(all_aperiodicity, axis=0)

            # Concatenate features along the last axis (time frames)
            X_phoneme = np.concatenate([ mfcc_features, delta_features, delta_delta_features, 
                                        energy_features, f0_features, aperiodicity_features], axis=1)

            # Normalize features using StandardScaler
            scaler = StandardScaler()
            X_phoneme = scaler.fit_transform(X_phoneme)
            
            # Save the scaler for later use
            scaler_file = os.path.join(save_model_dir, f"{phoneme_name}_scaler.pkl")
            joblib.dump(scaler, scaler_file)
            
            # Train HMM model for this phoneme
            logger.info(f"Training HMM model for phoneme: {phoneme_name}")
            hmm_trainer = TrainHMMModel(n_components=5, n_iter=500, random_state=30)
            hmm_trainer.train(X_phoneme, patience=50)

            # Save the trained HMM model
            model_file = os.path.join(save_model_dir, f"{phoneme_name}_hmm_model.pkl")
            hmm_trainer.save_model(model_file)

            logger.info(f"Saved HMM model for phoneme: {phoneme_name}")

    logger.info("All phoneme models have been trained and saved.")
