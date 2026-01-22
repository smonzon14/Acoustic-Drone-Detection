---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
  splits:
  - name: train
    num_bytes: 38218213221
    num_examples: 180320
  download_size: 6809972235
  dataset_size: 38218213221
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: mit
task_categories:
- audio-classification
language:
- en
pretty_name: DADS
---

### Dataset Description
Drone Audio Detection Samples (DADS) is currently the largest publicly available drone audio database, specifically designed for developing drone detection systems using deep learning techniques. All audio files are standardized to a sample rate of 16,000 Hz, 16-bit depth, mono-channel, and vary in length from 500 milliseconds to several minutes.
Most drone audio files were manually trimmed to ensure that a drone was always present in the recording. However, some datasets contain drone audio captured from distant microphones, making it necessary to carefully listen multiple times to detect drone sounds in the background.
### Dataset Structure
The dataset is structured into two main categories:
- **0 (no drone)**: Contains audio samples without drone sounds.
- **1 (drone)**: Contains audio samples of various drones.

### Dataset Statistics
| Category    | # Files | Total Duration (s) | Total Duration (h) | Avg. Duration (s) |
|-------------|---------|--------------------|---------------------|-------------------|
| No-Drone (0)| 16,729  | 121,714            | **33.81**           | 7.28              |
| Drone (1)   | 163,591 | 97,563.39          | **27.10**           | 0.60              |

All audio files share the following attributes:
- **Sample Rate**: 16 kHz
- **Bit Depth**: 16 bits
- **Channels**: Mono
- **Format**: WAV, PCM 16-bit

### Usage
DADS is intended for developing robust audio-based drone detection algorithms and models in security, surveillance, and related research fields.

**Drone Audio Sources:**
1. **Drone Audio Dataset:**
   Alemadi, S. (2019). *Drone Audio Dataset* [Data set]. GitHub. https://github.com/saraalemadi/DroneAudioDataset/tree/master

2. **SPCup 19 Egonoise Dataset:**
   Inria. (2019). *The SPCup19 Egonoise Dataset* [Data set]. Inria. http://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/

3. **DREGON Dataset:**
   Inria. (2019). *The DREGON Dataset* [Data set]. Inria. http://dregon.inria.fr/datasets/dregon/

4. **Drone Noise Database:**
   Ramos-Romero, C., Green, N., Asensio, C., & Torija Martinez, A. J. (2024). *DroneNoise Database* [Data set]. University of Salford. https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411

5. **AUDROK Drone Sound Data:**
   AUDROK. (2023). *AUDROK Drone Sound Data* [Data set]. Mobilithek. https://mobilithek.info/offers/605778370199691264

6. **Sound-Based Drone Fault Classification Using Multi-Task Learning:**
   Yi, W., Choi, J.-W., & Lee, J.-W. (2023). *Sound-Based Drone Fault Classification Using Multi-Task Learning* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7779574

**Non-Drone Audio Sources:**
1. **Urban Sound 8K:**
   Salamon, J., Jacoby, C., & Bello, J. P. (2014). *UrbanSound8K Dataset* [Data set]. Urban Sound Datasets. https://urbansounddataset.weebly.com

2. **TUT Acoustic Scenes 2017:**
   Mesaros, A., Heittola, T., & Virtanen, T. (2017). *TUT Acoustic Scenes 2017, Evaluation Dataset* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1040168

3. **Environmental Sound Classification (ESC)-50:**
   Piczak, K. J. (2015). *ESC-50: Dataset for Environmental Sound Classification* [Data set]. GitHub. https://github.com/karolpiczak/ESC-50

4. **Dataset for Noise Classification (DNC):**
   Zequeira, R. I. (2021). *Dataset for Noise Classification (DNC)* [Data set]. GitHub. https://github.com/zequeira/DNC

### Licensing
Data from various sources is licensed under Creative Commons Attribution and similar open licenses. Please verify individual licenses for commercial and non-commercial usage.