# NYCU-2021Fall-Introduction_to_Machine_Learning
NYCU 2021 Fall Introduction to Machine Learning Final project

# Requirements:
Python packages that you need.  
numpy==1.19.5  
pytube==10.8.2  
matplotlib==3.3.4  
SoundFile==0.10.3.post1  
librosa==0.8.0  
moviepy==1.0.3  
torch==1.8.0+cu111  
# File description
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

# Usage
### Train and test on GTZAN dataset:
1. Download GTZAN dataset and place genre folders under dataset/gtzan
2. Augment data by running ```$ python audio_augmentation.py```
3. Extracts feature by running ```$ python feature_extraction```
4. Starts train by running ```$ python XXX_Train_Test_Plot.py```

### Use trained model to classify youtube video:
1. Make sure you have "Demo_CNN_model.pth" under the same directory.
2. Run ```$ python Demo-classify-youtube.py```

**Note:** YouTube tends to block python crawler, if the process is stucked, please enter ^C and restart the program.
