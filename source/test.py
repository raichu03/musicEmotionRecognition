import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataLoader import AudioDataset, collate_fn, test_dataset


def evaluate(test_dataloader, model, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    # Load label map from JSON file
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
    
    # Invert label map for human-readable labels
    idx_to_label = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for input_values, labels in tqdm(test_dataloader, desc="Evaluating"):
            input_values = input_values.to(device)
            labels = labels.to(device)

            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    report = classification_report(all_labels, all_predictions, output_dict=True)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Generate confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[idx_to_label[i] for i in range(len(label_map))], 
                yticklabels=[idx_to_label[i] for i in range(len(label_map))])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    
    return total_loss / len(test_dataloader), accuracy, report


def main( data_path, saved_model_path: str, label_map_path: str, batch_size: int):
    
    file_paths, labels = test_dataset(data_path, label_map_path)
    labels_num = len(set(labels))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(saved_model_path, map_location=device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=labels_num,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    test_dataset = AudioDataset(file_paths, labels, processor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    loss, accuracy, report = evaluate(test_dataloader, model, device)

    print("Confusion matrix saved as 'confusion_matrix.png'.")
    print(f"Model testing complete with loss: {loss}, accuracy: {accuracy} and report: {report}")
    

if __name__=="__main__":
    
    data = "" #Path to the test data
    model_path = "" #Path to the saved model
    label_map = "" #Path to the label map
    batch_size = 0 #Batch size for testing
    main()