import sounddevice as sd
import os
from scipy.io.wavfile import write
from matplotlib.widgets import Button
import numpy as np
from datetime import datetime
import threading
import matplotlib.pyplot as plt

# Select sounddevice with index 0
sd.default.device = 0

class AudioRecorderUI:
  def __init__(self, sample_rate=48000, channels=16):
    self.sample_rate = sample_rate
    self.channels = channels
    self.is_recording = False
    self.recording_data = []
    self.output_file = None
    self.stream = None
    
    self.fig, self.ax = plt.subplots(figsize=(8, 6))
    self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    self.ax.set_xlim(0, 10)
    self.ax.set_ylim(-1, 1)
    self.ax.set_title("Press SPACEBAR to start/stop recording")
    self.ax.set_xlabel("Time (s)")
    self.ax.set_ylabel("Amplitude")
    
    self.status_text = self.ax.text(0.5, 0.5, "Ready - Press SPACEBAR", 
                     transform=self.ax.transAxes,
                     ha='center', va='center', fontsize=16)
    
  def audio_callback(self, indata, frames, time_info, status):
    """Callback function to capture audio chunks in real-time."""
    if status:
      print(f"Status: {status}")
    if self.is_recording:
      self.recording_data.append(indata.copy())
  
  def on_key_press(self, event):
    if event.key == ' ':
      if not self.is_recording:
        self.start_recording()
      else:
        self.stop_recording()
  
  def start_recording(self):
    self.is_recording = True
    self.recording_data = []
    self.status_text.set_text("Recording... Press SPACEBAR to stop")
    self.status_text.set_color('red')
    plt.draw()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.output_file = f"recordings/recording_{timestamp}.wav"
    
    print("Recording started...")
    self.stream = sd.InputStream(
      samplerate=self.sample_rate,
      channels=self.channels,
      callback=self.audio_callback
    )
    self.stream.start()
  
  def stop_recording(self):
    if not self.is_recording:
      return
      
    self.is_recording = False
    
    if self.stream:
      self.stream.stop()
      self.stream.close()
      self.stream = None
    
    if self.recording_data:
      recording = np.concatenate(self.recording_data, axis=0)
      write(self.output_file, self.sample_rate, recording)
      print(f"Recording saved to {self.output_file} with {self.channels} channels")
      print(f"Duration: {len(recording) / self.sample_rate:.2f} seconds")
    
    self.status_text.set_text("Stopped - Press SPACEBAR to record again")
    self.status_text.set_color('green')
    plt.draw()
  
  def show(self):
    plt.show()

if __name__ == "__main__":
  os.makedirs("recordings", exist_ok=True)
  ui = AudioRecorderUI()
  ui.show()