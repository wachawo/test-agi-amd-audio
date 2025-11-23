# Asterisk VAD and AMD Scripts

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/e128f1a5-67ac-4f80-96cf-3caf9caede8b" />

This project provides scripts for Voice Activity Detection (VAD) and Answering Machine Detection (AMD) for use with Asterisk.

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/a839f1f9-3b6c-4f07-bf22-ad9e2e0b2a37" />

### `amd_audio.py`

An Asterisk EAGI script that performs Voice Activity Detection (VAD) and Answering Machine Detection (AMD) on an audio stream.

### `test_amd_audio.py`

A utility script to test the `amd_audio.py` script with a local `.wav` file. It helps evaluate the accuracy of the VAD and AMD algorithms.

### Usage
```bash
python3 test_amd_audio.py -a <audio_file.wav> -b amd_audio.py [options]
```

**Options:**
- `-c, --channel <0|1>`: Select audio channel (0=Left, 1=Right). Default: 0.
- `-p, --play`: Play back the captured audio.
- `-i, --image`: Generate and display a waveform image of the captured audio.

### Examples
```bash
python3 test_amd_audio.py -a 2.wav -c 0 -p -i -b ./amd_audio.py
python3 test_amd_audio.py -a 2.wav -c 0 -p -i -b ./amd_audio
```
