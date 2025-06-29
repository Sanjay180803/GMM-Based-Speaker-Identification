# -*- coding: utf-8 -*-
"""Speaker Identification Task"""

import os
import shutil
import math
import subprocess
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc, delta
from subprocess import Popen, PIPE
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class DataManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def make_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def copy_files(self, dst, group):
        for fname in group:
            shutil.copy2(fname, os.path.join(dst, os.path.basename(fname)))

    def get_fnames_from_dict(self, dataset_dict, key):
        training_data, testing_data = [], []
        total = len(dataset_dict[key])
        split_idx = math.trunc(total * 2 / 3)
        training_data += dataset_dict[key][:split_idx]
        testing_data += dataset_dict[key][split_idx:]
        return training_data, testing_data

    def manage(self):
        dataset_directory = self.dataset_path
        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist:", dataset_directory)
            return

        print("Checking dataset directory:", dataset_directory)
        file_names = []
        for speaker_folder in os.listdir(dataset_directory):
            speaker_path = os.path.join(dataset_directory, speaker_folder)
            if os.path.isdir(speaker_path):
                audio_subfolder_path = os.path.join(speaker_path, 'audio')
                if os.path.isdir(audio_subfolder_path):
                    audio_files = [f for f in os.listdir(audio_subfolder_path) if f.endswith('.wav')]
                    for file in audio_files:
                        file_path = os.path.join(audio_subfolder_path, file)
                        file_names.append((speaker_folder, file_path))

        if not file_names:
            print("No audio files found.")
            return

        dataset_dict = {}
        for speaker_id, file_path in file_names:
            dataset_dict.setdefault(speaker_id, []).append(file_path)

        self.make_folder("TrainingData")
        self.make_folder("TestingData")

        for key in dataset_dict.keys():
            training, testing = self.get_fnames_from_dict(dataset_dict, key)
            speaker_train_folder = os.path.join("TrainingData", key)
            speaker_test_folder = os.path.join("TestingData", key)
            self.make_folder(speaker_train_folder)
            self.make_folder(speaker_test_folder)
            self.copy_files(speaker_train_folder, training)
            self.copy_files(speaker_test_folder, testing)


class SilenceEliminator:
    def __init__(self):
        pass

    def ffmpeg_silence_eliminator(self, input_path, output_path):
        filter_command = [
            "ffmpeg", "-i", input_path,
            "-af", "silenceremove=1:0:0.05:-1:1:-36dB",
            "-ac", "1", "-ss", "0", "-t", "90", output_path, "-y"
        ]
        subprocess.run(filter_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with_silence_duration = os.popen(f"ffprobe -i '{input_path}' -show_format -v quiet | sed -n 's/duration=//p'").read()
        no_silence_duration = os.popen(f"ffprobe -i '{output_path}' -show_format -v quiet | sed -n 's/duration=//p'").read()
        load_command = ["ffmpeg", "-i", output_path, "-f", "wav", "-"]
        p = Popen(load_command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        data, error = p.communicate()
        if error:
            print("FFmpeg Load Error:", error.decode())
        audio_np = np.frombuffer(data[data.find(b'\x00data') + 9:], np.int16)
        os.remove(output_path)
        return audio_np, no_silence_duration


class FeaturesExtractor:
    def __init__(self):
        pass

    def extract_features(self, audio, rate):
        mfcc_feature = mfcc(audio, rate, winlen=0.025, winstep=0.01, numcep=20, nfilt=30, nfft=512, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        deltas = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined = np.hstack((mfcc_feature, deltas, double_deltas))
        return combined

    def accelerated_get_features_vector(self, input_wave_file, audio, sound_rate):
        try:
            return self.extract_features(audio, sound_rate)
        except Exception as e:
            print(f"Cannot extract features from {os.path.basename(input_wave_file)}: {e}")
            return np.array([])


def train_gmms(trainpath, gmm_destination):
    file_paths = []
    os.makedirs(gmm_destination, exist_ok=True)
    for root, dirs, files in os.walk(trainpath):
        speaker_files = [os.path.join(root, f) for f in files if f.endswith('.wav')]
        if speaker_files:
            file_paths.append(speaker_files)

    for files in file_paths:
        features = np.asarray(())
        for filepath in files:
            print("Processing:", filepath)
            rate, audio = wavfile.read(filepath)
            extractor = FeaturesExtractor()
            vector = extractor.accelerated_get_features_vector(filepath, audio, rate)
            if vector.size == 0:
                continue
            features = vector if features.size == 0 else np.vstack((features, vector))

        if features.size == 0 or features.ndim != 2:
            continue

        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)
        speaker_id = os.path.basename(filepath).split('_')[0]
        model_path = os.path.join(gmm_destination, f"{speaker_id}.gmm")
        with open(model_path, 'wb') as f:
            pickle.dump(gmm, f)
        print(f"Trained and saved GMM for speaker {speaker_id} at {model_path}")


def test_gmms(modelpath, testpath):
    db = {}
    for fname in [f for f in os.listdir(modelpath) if f.endswith('.gmm')]:
        speaker = fname.split('.')[0]
        model = pickle.load(open(os.path.join(modelpath, fname), 'rb'))
        db[speaker] = model

    file_paths = [os.path.join(root, f) for root, _, files in os.walk(testpath) for f in files if f.endswith('.wav')]
    error, total_sample = 0, 0

    for path in file_paths:
        expected_speaker = os.path.basename(path).split('_')[0]
        if expected_speaker in db:
            rate, audio = wavfile.read(path)
            extractor = FeaturesExtractor()
            vector = extractor.accelerated_get_features_vector(path, audio, rate)
            if vector.shape == (0,):
                continue
            total_sample += 1
            log_likelihood = {speaker: model.score(vector).sum() for speaker, model in db.items()}
            predicted_speaker = max(log_likelihood, key=log_likelihood.get)

            plt.figure(figsize=(10, 5))
            plt.bar(log_likelihood.keys(), log_likelihood.values())
            plt.title(f'Log-Likelihood Scores for {os.path.basename(path)}')
            plt.xlabel('Speakers')
            plt.ylabel('Log-Likelihood')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            if predicted_speaker != expected_speaker:
                error += 1
            print(f"Processed: {os.path.basename(path)} | Expected: {expected_speaker} | Identified: {predicted_speaker}")

    if total_sample > 0:
        accuracy = ((total_sample - error) / total_sample) * 100
        print(f"Overall Identification Accuracy: {accuracy:.2f}%")
    else:
        print("No valid samples were processed.")


if __name__ == "__main__":
    # Prepare the data splits
    data_manager = DataManager("/content/drive/MyDrive/SSN_TDSC_duplicate/data/control")
    data_manager.manage()

    # Train GMM models
    train_gmms("TrainingData", "SpeakerModels")

    # Test GMM models
    test_gmms("SpeakerModels", "TestingData/audio")
