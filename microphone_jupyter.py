import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime
import os

# Select sounddevice with index 0
sd.default.device = 0

class AudioRecorder:
  def __init__(self, sample_rate=44100, channels=16):
    self.sample_rate = sample_rate
    self.channels = channels
    self.recording_data = []
    self.output_file = None
  
  def audio_callback(self, indata, frames, time_info, status):
    """Callback function to capture audio chunks in real-time."""
    if status:
      print(f"Status: {status}")
    self.recording_data.append(indata.copy())
  
  def record(self, duration=60):
    self.recording_data = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.output_file = f"recordings/recording_{timestamp}.wav"
    
    print(f"Recording for {duration} seconds... (Press Ctrl+C to stop early)")
    
    try:
      with sd.InputStream(
        samplerate=self.sample_rate,
        channels=self.channels,
        callback=self.audio_callback
      ):
        sd.sleep(duration * 1000)
    except KeyboardInterrupt:
      print("\nRecording interrupted by user")
    
    if self.recording_data:
      recording = np.concatenate(self.recording_data, axis=0)
      write(self.output_file, self.sample_rate, recording)
      duration = len(recording) / self.sample_rate
      print(f"Recording saved to {self.output_file} with {self.channels} channels")
      print(f"Duration: {duration:.2f} seconds")

if __name__ == "__main__":
  os.makedirs("recordings", exist_ok=True)
  recorder = AudioRecorder()
  recorder.record(duration=60)
