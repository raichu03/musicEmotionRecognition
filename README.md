# Music Genre Classification

## Introduction
This project is a music genre classification project. The goal is to classify music into different genres. The dataset used is the Nepali Music Dataset. The dataset contains 10 different genres of music. The genres are:
1. Asare
2. Deuda
3. Jhyaure
4. Kumari
5. Maruni
6. Sakela
7. Tamang
8. Salaijo
9. Sarangi
10. Tharu

The dataset contains ~7 hours of audio data. The dataset is divided into training and testing data. The training data contains 80% of the data and the testing data contains 20% of the data. The dataset is divided into 10 folders, each folder representing a genre. The audio files are in .wav format.

## Features
The model is capable of extracting the features form raw audio waveforms and classify the music into different  genres. Without the need of any manual feature extraction it might be able to classify the music into different genres more accurately.

## Installation
To run the project, you need to install all the requirements. The requirements are in the requirements.txt file. To install the requirements, run the following command:

Clone the repository;
```bash
git clone https://github.com/raichu03/musicEmotionRecognition.git
```
After cloning the repository, navigate to the home directory of the project and run the following command:

```bash
pip install -r requirements.txt
```

To train the model on your own dataset, you need to make some changes in the train.py file inside the source directory. You need to change the following variables:
1. **data_dir** = " Add your data path here " 
2. **batch_size** = 0 # Batch size
3. **epochs** = 0 #Number of epochs
4. **label_map_path** = " Add path to save the label_map file "

To test the model on your own dataset, you need to make changes similar to the test.py file. The changes are same as the train.py file.
