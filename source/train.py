### Imports ###
import os
import numpy as np
import json

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataLoader import load_dataset, collate_fn, AudioDataset


def train(model, optimizer, train_dataloader, device):
    """
    Train the model.
    Args:
        model: Wav2Vec2ForSequenceClassification: Model to train.
        optimizer: AdamW: Optimizer to use for training.
        train_dataloader: DataLoader: Dataloader containing the training data.
        device: str: Device to use for training.
    """
    
    model.train()
    total_loss = 0
    
    for input_values, labels in tqdm(train_dataloader, desc="Training"):
        input_values = input_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_dataloader)

def evaluate(model, validation_dataloader, device):
    """
    Evaluate the model.
    Args:
        model: Wav2Vec2ForSequenceClassification: Model to evaluate.
        validation_dataloader: DataLoader: Dataloader containing the validation data.
        device: str: Device to use for evaluation.
    """
    
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for input_values, labels in tqdm(validation_dataloader, desc="Evaluating"):
            input_values = input_values.to(device)
            labels = labels.to(device)

            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    # Compute metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    return total_loss / len(validation_dataloader), accuracy, report
    

def main(data_dir, batch_size, epochs, label_map_path=None):
    
    if data_dir is None:
        raise ValueError("Data directory is not provided.")
    
    file_paths, labels = load_dataset(data_dir, label_map_path)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(file_paths, labels, test_size=0.5, random_state=42, stratify=labels)
    labels_num = len(set(labels))
    
    ## We will use the Wav2Vec2 processor and model for this task ##
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=labels_num)
    
    ## Create the dataset ##
    train_dataset = AudioDataset(train_paths, train_labels, processor)
    validation_dataset = AudioDataset(validation_paths, validation_labels, processor)
    
    ## Create the dataloaders ##
    tran_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    ## Define the optimizer and models ##
    optimizer = AdamW(model.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Training on {device}")
    
    best_val_loss = float('inf')
    model_path = "models/best_model.pt"
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_classification_reports = []
    
    for epoch in range(epochs):
        train_loss = train(model, optimizer, tran_dataloader, device)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        val_loss, accuracy, report = evaluate(model, validation_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        val_classification_reports.append(report)
        
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
        print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy}")
        print(f"Epoch {epoch+1}, Classification Report: {report}")
        
        if val_loss < best_val_loss:  # Save based on validation loss
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            }, model_path)
    
    print(f"Training complete. Best model saved at {model_path} with Validation Loss: {best_val_loss:.4f}")
    
    # Save metrics to file or display graph
    plt.figure(figsize=(10, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    
    plt.savefig("metrics_plot.png", dpi=300)
    plt.close()
    
    print("Training complete")
            
            
if __name__ == "__main__":
    
    data_dir = "" #Path to the data directory
    batch_size = 0 #Batch size for training
    epochs = 0 #Number of epochs
    label_map_path = "" #Path to the label map
    main(data_dir, batch_size, epochs, label_map_path) 
 