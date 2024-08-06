from data_preprocessing import preprocess,features_extractor
from record_split import split_audio
from keras.models import load_model
from data_preprocessing import change_format
import os
import numpy as np
import pandas as pd
filename = "D:\\audio_pred"
df=pd.DataFrame(columns=['data'])
model = load_model(r'C:\mine\guitar_notes_classification\saved_models\audio_classification_lstm_all.hdf5')
for files in os.listdir(filename):
    df1=pd.DataFrame()
    # change_format(f'{filename}\{files}',f'{filename}\{files.split(".")[0]}.wav')
    split_audio(sound_file=f'{filename}\{files}',dest_path=r'C:\mine\guitar_notes_classification\output\out')
    predlist=[]
    for i in os.listdir(r'C:\mine\guitar_notes_classification\output\out'):
        data=features_extractor(file_name=f'C:\mine\guitar_notes_classification\output\out\{i}')
        data=data.reshape(1,1,40)
        pred=model.predict(data)
        predlist.append(np.argmax(pred))
        print(np.argmax(pred))
        os.remove(f'C:\mine\guitar_notes_classification\output\out\{i}')
    df1=pd.DataFrame({'data':[predlist]})
    df=df.append(df1,ignore_index=True)
    df.to_csv('dataset_for_ed.csv')