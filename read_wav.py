import numpy as np
from scipy.io import wavfile
from scipy import signal

import matplotlib.pyplot as plt

# Read the WAV file
filename = 'recordings/recording_20260120_170442.wav'  # Replace with your file path
sample_rate, data = wavfile.read(filename)

# Get number of channels
if len(data.shape) == 1:
  num_channels = 1
  data = data.reshape(-1, 1)
else:
  num_channels = data.shape[1]

print(f"Sample rate: {sample_rate} Hz")
print(f"Number of channels: {num_channels}")
print(f"Data shape: {data.shape}")

# Create subplots for spectrograms
fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels))

# Handle single channel case
if num_channels == 1:
  axes = [axes]

# Generate spectrogram for each channel
for i in range(num_channels):
  channel_data = data[:, i]
  
  # Compute spectrogram
  f, t, Sxx = signal.spectrogram(channel_data, sample_rate, nperseg=2048, noverlap=1024)
  
  # Plot spectrogram
  axes[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
  axes[i].set_ylabel(f'Ch {i+1}\nFreq (Hz)')
  axes[i].set_xlabel('Time (s)')
  axes[i].set_title(f'Channel {i+1} Spectrogram')

plt.tight_layout()
plt.show()