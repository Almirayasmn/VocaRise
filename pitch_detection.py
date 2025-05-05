# pitch_detection.py

import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
file = 'sample_voice.wav'  # You can replace this with your own voice sample
y, sr = librosa.load(file)

# Extract pitch using librosa's piptrack
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Get the most prominent pitch in each frame
pitch_track = []
for i in range(pitches.shape[1]):
    index = magnitudes[:, i].argmax()
    pitch = pitches[index, i]
    pitch_track.append(pitch)

# Clean and smooth
pitch_track = np.array(pitch_track)
pitch_track = np.where(pitch_track < 50, np.nan, pitch_track)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(pitch_track, label='Pitch (Hz)', color='purple')
plt.title("Pitch Detection Result")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.tight_layout()
plt.show()
