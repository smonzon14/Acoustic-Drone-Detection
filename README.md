## Audio Recording

### List audio devices
Run the device listing script to find the 16-channel microphone array:

```bash
python list_audio_devices.py
```

Look for an input device that reports 16 channels. Note its index.

### Record with the microphone array
1. Open `microphone.py` and set the device index:

```python
sd.default.device = <your_device_index>
```

2. Start the recorder:

```bash
python microphone.py
```

3. Press SPACEBAR to start/stop recording. Files are saved under `recordings/`.

## ML Detection Model

The detection pipeline uses a CNN trained on spectrograms to classify audio as
`0` (no drone) or `1` (drone).

Download the dataset from Hugging Face and place the parquet files under
`detection/local_dataset_dir/data`:

https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
