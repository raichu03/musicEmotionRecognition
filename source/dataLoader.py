import os
import json

import torch
import torchaudio
from torch.utils.data import Dataset

### Train Datase Loader ###
def load_dataset(data_dir: str, save_label_map_path=None):
    """
    Loads the dataset for training form the directory and creates a label map.
    Args:
        data_dir: str: Path to the directory containing the dataset.
        save_label_map_path: str: Path to save the label map.
        Returns: 
            dataset: list: List of tuples containing the audio file path and the corresponding label.
            label_map: dict: Dictionary containing the mapping of labels to integers.
    """
    
    file_paths = []
    labels = []
    # Loop through the folders in the data directory
    label_map = {folder: idx for idx, folder in enumerate(os.listdir(data_dir))}
    
    if save_label_map_path:
        with open(save_label_map_path, 'w') as f:
            json.dump(label_map, f, indent=4)
            
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_paths.append(os.path.join(folder_path, file))
                    labels.append(label_map[folder])
    
    return file_paths, labels

### Test Datase Loader ###
def test_dataset(data_dir: str, label_map_path=None):
    """
    Loads the dataset for testing form the directory and creates a label map.
    Args:
        data_dir: str: Path to the directory containing the dataset.
        save_label_map_path: str: Path to save the label map.
        Returns: 
            dataset: list: List of tuples containing the audio file path and the corresponding label.
            label_map: dict: Dictionary containing the mapping of labels to integers.
    """
    file_paths = []
    labels = []
    
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
        
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            if folder not in label_map:
                print(f"Skipping folder '{folder}' as it is not in the label map.")
                continue
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_paths.append(os.path.join(folder_path, file))
                    labels.append(label_map[folder])
    
    return file_paths, labels
    
class AudioDataset(Dataset):
    
    def __init__(self, file_paths, labels, processor, target_length=30000):
        """
        Process the data and returns in suitable format.
        Args:
            file_paths: list: List of file paths.
            labels: list: List of labels.
            processor: Wav2vec2Processor: Processor to process the audio files.
            target_length: int: Target length of the audio files in milliseconds.
        """
        self.file_paths = file_paths   
        self.labels = labels
        self.processor = processor
        self.target_length = target_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_input, sample_rate = torchaudio.load(self.file_paths[idx])
        
        ## Convert stereo to mono if needed ##
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)
        
        ## Reample the audio to 16kHz ##
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_input = resampler(audio_input)
        
        ## Pad or truncate the audio to the target length ##
        # Truncate
        if audio_input.shape[1] > self.target_length:
            audio_input = audio_input[:, :self.target_length]
        
        # Pad with zeros
        elif audio_input.shape[1] < self.target_length:
            padding = self.target_length - audio_input.shape[1]
            audio_input = torch.nn.functional.pad(audio_input, (0, padding))
            
        audio_input = audio_input.squeeze(0)
        input_values = self.processor(audio_input, sampling_rate=16000, return_tensors='pt').input_values
        label = torch.tensor(self.labels[idx])
        
        return input_values.squeeze(), label

def collate_fn(batch):
    """
    Function to stack the input and labels in the batch.
    Args:
        batch: list: List of tuples containing the input and label.
    """
    
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels             