# AMD Audio Tools

## amd_audio.py
Designed for Asterisk integration. Its primary purpose is to detect the start of speech (after initial silence) and extract the first 2 seconds of audio. This segment is then analyzed to determine if the responder is a **Human** or an **Answering Machine**.

## test_amd_audio.py
A utility to test `amd_audio.py` locally using a WAV file. It simulates the Asterisk environment, feeds audio to the script, and captures the result for verification.

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
python3 test_amd_audio.py -a 2.wav -c 1 -p -i -b amd_audio.py
python3 test_amd_audio.py -a 2.wav -c 1 -p -i -b ./amd_audio
```