# This project contains Asterisk EAGI script for Voice Activity Detection (VAD) and Answering Machine Detection (AMD).

## amd_audio.py
Example Asterisk EAGI script for Voice Activity Detection (VAD) and Answering Machine Detection (AMD).

## test_amd_audio.py
The script allows you to check the effectiveness of Voice Activity Detection (VAD) and Answering Machine Detection (AMD).

### Usage
```bash
python3 test_amd_audio.py -a <audio_file.wav> -b amd_audio.py [options]
```

**Options:**
- `-c, --channel <0|1>`: Select audio channel (0=Left, 1=Right). Default: 1.
- `-p, --play`: Play back the captured audio.
- `-i, --image`: Generate and display a waveform image of the captured audio.

### Examples
```bash
python3 test_amd_audio.py -a 2.wav -c 1 -p -i -b ./amd_audio.py
python3 test_amd_audio.py -a 2.wav -c 1 -p -i -b ./amd_audio
```