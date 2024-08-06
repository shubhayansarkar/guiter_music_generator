from pydub import AudioSegment
sound_file = r"C:\mine\guitar_notes_classification\audiofiles\Ni.wav"
dest_path = r"C:\mine\guitar_notes_classification\audiofiles\Ni"


def split_audio(sound_file, dest_path):
    sound = AudioSegment.from_wav(sound_file)
    sound.duration_seconds == (len(sound) / 1000.0)
    # seconds to minutes conversion
    minutes_duartion = int(sound.duration_seconds // 60)
    seconds_duration = round((sound.duration_seconds % 60), 3)
    print(minutes_duartion, ':', round(seconds_duration, 0))
    print(minutes_duartion * 60 + round(seconds_duration, 0))
    for i in range(int(minutes_duartion * 60 + round(seconds_duration, 0))):
        StrtSec = i
        EndSec = i + 1
        StrtTime = StrtSec * 1000
        EndTime = EndSec * 1000
        extract = sound[StrtTime:EndTime]
        extract.export(f"{dest_path}\{dest_path[-2:]}_{i}.wav", format="wav")
def audio_merge(sound):
    from pydub import AudioSegment
    from pydub import effects
    velocidad_X = 2.2  # No puede estar por debajo de 1.0

    final = 0
    for i in range(len(sound)):
        final += AudioSegment.from_wav(f'real_music/{sound[i]}.wav')

        final_audio = final.export('final.wav', format='wav')

    sound = AudioSegment.from_wav('final.wav')
    so = sound.speedup(velocidad_X, 150, 25)
    so.export('final.wav', format='wav')
# split_audio(sound_file,dest_path)
