from pvrecorder import PvRecorder
import wave
import struct
import os
import numpy as np
import time
from pydub.playback import play
from pydub import AudioSegment
from data_preprocessing import preprocess,features_extractor
from record_split import split_audio,audio_merge
from keras.models import load_model
model = load_model(r'C:\mine\guitar_notes_classification\saved_models\audio_classification_lstm_all.hdf5')
for index, device in enumerate(PvRecorder.get_audio_devices()):
    print(f"[{index}] {device}")

path=r'sample_audio.wav'
recorder = PvRecorder(device_index=1, frame_length=512)
audio = []
print(3)
time.sleep(1)
print(2)
time.sleep(2)
print(1)
time.sleep(1)
try:
    recorder.start()

    while True:
        frame = recorder.read()
        audio.extend(frame)

except KeyboardInterrupt:
    print('recorded....starting music')
    recorder.stop()
    with wave.open(path, 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))

    split_audio(sound_file=path, dest_path=r'C:\mine\guitar_notes_classification\output\out')
    predlist = []
    for i in os.listdir(r'C:\mine\guitar_notes_classification\output\out'):
        data = features_extractor(file_name=f'C:\mine\guitar_notes_classification\output\out\{i}')
        data = data.reshape(1, 1, 40)
        pred = model.predict(data)
        predlist.append(np.argmax(pred))
        # print(np.argmax(pred))
    audio_merge(predlist)
    final1=AudioSegment.from_wav('final.wav')
    play(final1)


finally:
    recorder.delete()