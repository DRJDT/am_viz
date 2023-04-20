import librosa
import numpy as np
import soundfile 
import matplotlib.pyplot as plt

# Load sample audio
filename = librosa.ex('trumpet')
# filename = '../samples/Everbloom_2-4-23_Master_2.wav'

y, sr = librosa.load(filename)

#yt,index = librosa.effects.trim(y)

# soundfile.write('../samples/test_output.wav', y, sr, subtype='PCM_24') # Write out audio as 24bit PCM WAV
# sf.write('../samples/test_output.flac', y, sr, format='flac', subtype='PCM_24') #  Write out audio as 24bit Flac

# fig, ax = plt.subplots()
# librosa.display.waveshow(y, sr=sr, ax=ax)
# plt.show()

# hop_length = 1024
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),ref=np.max)
# # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)),ref=np.max)

# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# img = librosa.display.specshow(D, y_axis='log', sr=sr, x_axis='time', ax=ax[1], hop_length=hop_length)

# ax[1].set(title='Log-frequency power spectrogram')
# ax[1].label_outer()
# fig.colorbar(img, ax=ax, format="%+2.f dB")

# plt.show()

# stft = np.abs(librosa.stft(y, hop_length=512, n_fft=2048))  # Compute Short-Time Fourier Transform -> matrix contains amplitude vs frequency vs time indexes
# spectrogram = librosa.amplitude_to_db(stft, ref=np.max) # convert amplitudes to dB

# librosa.display.specshow(spectrogram,y_axis='log', x_axis='time')
# plt.title('Your title')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()




chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)

# For display purposes, let's zoom in on a 15-second chunk from the middle of the song
idx = tuple([slice(None), slice(*list(librosa.time_to_frames([45, 60])))])

# And for comparison, we'll show the CQT matrix as well.
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))


fig, ax = plt.subplots(nrows=2, sharex=True)
img1 = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max)[idx],
                                y_axis='cqt_note', x_axis='time', bins_per_octave=12*3,
                                ax=ax[0])
fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")
ax[0].label_outer()

img2 = librosa.display.specshow(chroma_orig[idx], y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img2, ax=[ax[1]])
ax[1].set(ylabel='Default chroma')