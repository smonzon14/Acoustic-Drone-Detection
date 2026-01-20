import sounddevice as sd
import os
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Select sounddevice with index 0
sd.default.device = 0

class AudioRecorderUI:
  def __init__(self, sample_rate=44100, channels=16):
    self.sample_rate = sample_rate
    self.channels = channels
    self.is_recording = False
    self.recording_data = []
    self.output_file = None
    self.stream = None
    
    # Create UI widgets
    self.record_button = widgets.Button(
      description='Start Recording',
      button_style='success',
      icon='microphone'
    )
    self.status_label = widgets.Label(value='Ready to record')
    
    self.record_button.on_click(self.toggle_recording)
    
    # Display UI
    self.ui = widgets.VBox([self.record_button, self.status_label])
  
  def audio_callback(self, indata, frames, time_info, status):
    """Callback function to capture audio chunks in real-time."""
    if status:
      print(f"Status: {status}")
    if self.is_recording:
      self.recording_data.append(indata.copy())
  
  def toggle_recording(self, button):
    if not self.is_recording:
      self.start_recording()
    else:
      self.stop_recording()
  
  def start_recording(self):
    self.is_recording = True
    self.recording_data = []
    self.record_button.description = 'Stop Recording'
    self.record_button.button_style = 'danger'
    self.status_label.value = '🔴 Recording...'
    
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
    
    self.record_button.description = 'Start Recording'
    self.record_button.button_style = 'success'
    self.status_label.value = '✅ Recording saved'
  
  def show(self):
    display(self.ui)

if __name__ == "__main__":
  os.makedirs("recordings", exist_ok=True)
  ui = AudioRecorderUI()
  ui.show()
