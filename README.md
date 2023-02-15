# CNN-Music-Genre-Classifier

A music genre classifier using CNN, RNN, and Mel spectrogram

Also provides a CNN feature extractor to be used with other machine learning models such as SVM / KN.

Course: Introduction to Machine Learning by C.C. Cheng (NYCU 2021 Fall)

## File description

### Report/
Contains all experiment results and details

### Demo-classify-youtube.py: 
Use trained model to classify youtube video.

### Hyper_parameters.py: 
Hyperparameter settings.

### audio_augmentation.py: 
Augments dataset.

### feature_extraction.py: 
Extracts feature from augmented dataset.

### XXX_Train_Test_Plot.py
Train + Test model {XXX}

## Usage
### Train and test on GTZAN dataset:
1. Download GTZAN dataset and place genre folders under dataset/gtzan
2. Augment data by running ```$ python audio_augmentation.py```
3. Extracts feature by running ```$ python feature_extraction```
4. Starts train by running ```$ python XXX_Train_Test_Plot.py```

### Use trained model to classify youtube video:
1. Make sure you have "Demo_CNN_model.pth" under the same directory.
2. Run ```$ python Demo-classify-youtube.py```

**Note:** YouTube tends to block python crawler, if the process is stucked, please enter ^C and restart the program.
