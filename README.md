# Speaker Identification using Gaussian Mixture Models (GMM)

This project implements **speaker identification** using Gaussian Mixture Models (GMMs). It trains speaker models on normal speech data (TDSC control data) [Audio Data] (https://catalog.ldc.upenn.edu/LDC2021S04) and tests on samples from the KADBAT dataset. By modeling speaker-specific voice features, the system identifies speakers based on the statistical likelihood that their speech matches a trained GMM.


## How the GMM-based Speaker Identification Works

1. **Data Preparation:**
   - Each speaker’s audio data is organized into folders.
   - The data is split into training and testing sets.

2. **Silence Removal:**
   - Silence is removed from audio samples using FFmpeg’s `silenceremove` filter.

3. **Feature Extraction:**
   - MFCC (Mel Frequency Cepstral Coefficients) features, along with their delta and double-delta coefficients, are extracted from each training audio file.
   - Cepstral Mean Normalization (CMS) is applied to improve robustness.

4. **GMM Training:**
   - For each speaker, extracted features are used to train a separate GMM with 16 components.
   - Models are saved as `.gmm` files, each representing a unique speaker.

5. **Testing and Identification:**
   - Features are extracted from test audio samples.
   - The log-likelihood of the sample belonging to each speaker’s GMM is calculated.
   - The speaker whose GMM yields the highest total log-likelihood is selected as the identified speaker.
   - The identification process outputs:
     - Bar plots of log-likelihoods for visual inspection.
     - Accuracy statistics over the test set.
     - Optionally, identification results for a single random file.

## Repository Structure

Your repo should look like:
```
/SpeakerModels           # GMM model files saved after training
/TrainingData            # Organized folders with each speaker's training audio
/TestingData/audio       # Audio files used for testing identification
speaker_identification_task.py
```

## Instructions

1. **Prepare the Dataset:**
   - Place your TDSC control data (normal speech) into structured folders inside `/TrainingData`.
   - Place your test audio files from KADBAT or other data into `/TestingData/audio`.

2. **Run the Script:**
   - Open `speaker_identification_task.py`.
   - Replace all dataset paths in the script with your local or cloud environment paths:
     - E.g., `trainpath`, `testpath`, `gmm_destination`, etc.
   - Run the script:
     ```bash
     python speaker_identification_task.py
     ```

3. **View Results:**
   - The script will train GMM models, evaluate test samples, and print identification accuracy.
   - Log-likelihood bar plots will be displayed for each processed test file.
   - Models will be saved in `/SpeakerModels`.

4. **Test Random Files:**
   - At the end of the script, you can specify a random `.wav` file’s path to test it directly.

---

##  Important Notes

- All **paths must be adjusted** in the script:
  - Replace default `/content/...` or `/drive/MyDrive/...` with the actual locations of your audio data.
- The script uses the `python_speech_features` library — install it before running:
  ```bash
  pip install python_speech_features
  ```
- MFCC features with deltas and double-deltas give a robust representation of each speaker’s unique voiceprint.
- GMMs are trained with 16 components, diagonal covariances, and three initializations for better convergence.

