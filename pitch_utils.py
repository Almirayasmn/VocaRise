

import librosa
import numpy as np

def analyze_pitch(audio_path):
    """
    Mengambil file audio dan mengembalikan daftar pitch per frame.
    """
    y, sr = librosa.load(audio_path)
    
    # Menggunakan YIN algorithm untuk mendeteksi pitch
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), 
                     fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # Membersihkan nilai NaN dan ekstrim
    f0_clean = f0[~np.isnan(f0)]
    f0_clean = f0_clean[f0_clean > 0]
    
    return f0_clean
