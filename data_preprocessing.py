import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

def change_format(m4a_file,wav_file):
    track = AudioSegment.from_file(m4a_file, format='m4a')
    file_handle = track.export(wav_file, format='wav')

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features
# Now we iterate through every audio file and extract features
# using Mel-Frequency Cepstral Coefficients
def preprocess(path,csv_name):
    extracted_features=[]
    for file in tqdm(os.listdir(path)):
        if (os.path.isdir(f'{path}\\{file}')):
            for files in tqdm(os.listdir(f'{path}\\{file}')):
                # classes.append(file)
                data=features_extractor(f'{path}\\{file}\\{files}')
                extracted_features.append([data,file])
    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
    extracted_features_df.to_csv(csv_path,index=False)
    print(extracted_features_df.shape)

path=r'C:\\mine\\guitar_notes_classification\\audiofiles'
csv_path=r'C:\mine\guitar_notes_classification\audiofiles\preprocessed_total.csv'
# preprocess(path,csv_path)