import sounddevice as sd

def list_audio_devices():
  """List all audio devices with their interface numbers."""
  devices = sd.query_devices()
  
  print("Available Audio Devices:")
  print("-" * 80)
  
  for idx, device in enumerate(devices):
    # Check if device has input channels (can be used as microphone)
    if device['max_input_channels'] > 0:
      print(f"Index: {idx}")
      print(f"Name: {device['name']}")
      print(f"Input Channels: {device['max_input_channels']}")
      print(f"Default Sample Rate: {device['default_samplerate']} Hz")
      print(f"Host API: {sd.query_hostapis(device['hostapi'])['name']}")
      print("-" * 80)

if __name__ == "__main__":
  list_audio_devices()
  