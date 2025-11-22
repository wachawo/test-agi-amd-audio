#!/usr/bin/env python3
"""
Simplified test script for amd_audio.py
- Runs amd_audio.py locally.
- Feeds audio to FD 3 (converting Stereo to Mono Left if needed).
- Captures output via local HTTP server.
- Analyzes offset (when voice started).
- Plays back received audio.
"""

import argparse
import http.server
from urllib.request import Request, urlopen
import json
import logging
import os
import socketserver
import subprocess
import sys
import threading
import time
import wave
import base64
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# 0 = Left, 1 = Right
CHANNEL_TO_USE = 0 # Deprecated, using args.channel
AMDAI_URL = os.getenv("AMDAI_URL", "http://127.0.0.1:9000/a/")
AUDIO_WAV = "audio.wav"
IMAGE_PNG = "waveform.png"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def display_waveform(audio_bytes, rate):
    try:
        data = np.frombuffer(audio_bytes, dtype=np.int16)
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title("Received Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(IMAGE_PNG)
        plt.close()
        logger.info(f"Saved waveform to {IMAGE_PNG}")
        
        # Try to open the image
        if sys.platform.startswith('linux'):
            subprocess.call(["xdg-open", IMAGE_PNG])
        elif sys.platform == 'darwin':
            subprocess.call(["open", IMAGE_PNG])
        elif sys.platform == 'win32':
            os.startfile(IMAGE_PNG)
            
    except Exception as e:
        logger.error(f"Failed to generate waveform: {e}")

class ServerData:
    def __init__(self):
        self.received_audio: bytes = b""
        self.received_json: dict = {}
        self.stop_event = threading.Event()

class AudioUploadHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/upload':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            response_data = b'{"answer": "UNKNOWN"}'
            try:
                data = json.loads(post_data.decode('utf-8'))
                self.server.server_data.received_json = data
                
                if "audio" in data:
                    audio_b64 = data["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    self.server.server_data.received_audio = audio_bytes
                    logger.info(f"Server: Received POST {content_length:,} bytes")
                    logger.info(f"Server: Decoded {len(audio_bytes):,} bytes of audio")
                    
                    # Proxy to API
                    try:
                        logger.info(f"Proxying to API: {AMDAI_URL}")
                        req = Request(
                            AMDAI_URL,
                            data=post_data, # Forward original JSON
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        )
                        with urlopen(req, timeout=10) as resp:
                            api_response = resp.read()
                            logger.info(f"API Response: {api_response.decode('utf-8')}")
                            response_data = api_response
                    except Exception as e:
                        logger.error(f"API Proxy Failed: {e}")
                        # Fallback to UNKNOWN is already set
                else:
                    logger.warning("Server: No 'audio' field in JSON")

            except Exception as e:
                logger.error(f"Server: Failed to decode JSON/Audio: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response_data)
            
            # Signal to stop
            self.server.server_data.stop_event.set()
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return # Suppress default logging

def run_http_server(port: int, server_data: ServerData):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), AudioUploadHandler) as httpd:
        httpd.server_data = server_data
        httpd.timeout = 0.5
        logger.info(f"HTTP server listening on {port}")
        while not server_data.stop_event.is_set():
            httpd.handle_request()

def read_wav_mono(path: str, channel: int = 0):
    """Reads WAV, returns mono raw bytes and framerate."""
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        logger.info(f"WAV: {params.nchannels}ch, {params.framerate}Hz, {params.sampwidth*8}bit")
        
        frames = wf.readframes(params.nframes)
        
        if params.sampwidth != 2:
            raise ValueError("Only 16-bit audio supported")
            
        if params.nchannels == 1:
            return frames, params.framerate
        elif params.nchannels == 2:
            # Stereo -> Mono
            data = np.frombuffer(frames, dtype=np.int16)
            data = data.reshape(-1, 2)
            if channel not in [0, 1]:
                logger.warning(f"Invalid channel {channel}, defaulting to 0")
                channel = 0
            mono = data[:, channel]
            logger.info(f"Converted Stereo to Mono (Channel {channel})")
            return mono.tobytes(), params.framerate
        else:
            raise ValueError(f"Unsupported channels: {params.nchannels}")

def save_wav(path: str, data: bytes, rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", required=True, help="Input WAV file")
    parser.add_argument("-b", "--binary", default="./amd_audio.py", help="Path to amd_audio script")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port")
    parser.add_argument("-p", "--play", action="store_true", help="Play received audio")
    parser.add_argument("-c", "--channel", type=int, default=CHANNEL_TO_USE, help="Audio channel to use (0=Left, 1=Right)")
    parser.add_argument("-i", "--image", action="store_true", help="Generate and show waveform image")
    args = parser.parse_args()

    if os.path.exists(AUDIO_WAV):
        os.remove(AUDIO_WAV)
    if os.path.exists(IMAGE_PNG):
        os.remove(IMAGE_PNG)

    # 1. Prepare Audio
    try:
        raw_audio, rate = read_wav_mono(args.audio, args.channel)
    except Exception as e:
        logger.error(f"Failed to read audio: {e}")
        sys.exit(1)

    # 2. Prepare Pipe (FD 3) BEFORE starting server to avoid clobbering socket
    read_fd, write_fd = os.pipe()
    
    # Function to set up fd/3 in the child process
    def make_setup_fd3(fd):
        def setup_fd3():
            import os
            try:
                # Copy the passed fd to fd/3 in the child process
                os.dup2(fd, 3)
                # Close the original fd (it's now available as fd/3)
                if fd != 3:
                    os.close(fd)
                # Make fd/3 inheritable (though this should already be done via pass_fds)
                os.set_inheritable(3, True)
            except OSError as e:
                import sys
                sys.stderr.write(f"Error setting up fd/3: {e}\n")
                sys.stderr.flush()
        return setup_fd3
    
    setup_fd3 = make_setup_fd3(read_fd)
    
    # Make read_fd inheritable
    os.set_inheritable(read_fd, True)

    # 3. Start Server
    server_data = ServerData()
    server_thread = threading.Thread(target=run_http_server, args=(args.port, server_data), daemon=True)
    server_thread.start()

    # 4. Prepare Command
    binary_path = args.binary
    local_url = f"http://127.0.0.1:{args.port}/upload"
    amd_duration_ms = "2000" # Hardcoded duration for amd_audio (ms)
    amd_timeout_sec = "10"    # Hardcoded timeout for amd_audio
    vad_enabled = "1"
    vad_aggressive = "3"

    cmd = [binary_path, local_url, amd_duration_ms, amd_timeout_sec, vad_enabled, vad_aggressive]

    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=setup_fd3,
            pass_fds=[read_fd]  # Pass read_fd to the child process
        )
    except Exception as e:
        logger.error(f"Failed to start process: {e}")
        sys.exit(1)

    # 5. Feed Audio - start IMMEDIATELY so the process doesn't hang waiting
    def feeder():
        # Small delay for process initialization (but very small)
        time.sleep(0.01)
        
        chunk_size = int(rate * 2 * 0.1)  # 100ms in bytes (2 bytes per sample for 16-bit)
        if chunk_size == 0:
            chunk_size = 1600  # Minimum chunk size (approximately 100ms at 8000Hz)
        
        offset = 0
        logger.info(f"Feeding audio ({len(raw_audio):,} bytes)...")
        try:
            with os.fdopen(write_fd, "wb") as f:
                # Send the first chunk IMMEDIATELY, without delay
                if raw_audio:
                    first_chunk = raw_audio[offset:offset+chunk_size]
                    if first_chunk:
                        f.write(first_chunk)
                        f.flush()
                        offset += len(first_chunk)
                        logger.info(f"First chunk {len(first_chunk):,} bytes sent immediately")
                
                # Continue transmission with delays
                while offset < len(raw_audio):
                    if server_data.stop_event.is_set():
                        break
                    chunk = raw_audio[offset:offset+chunk_size]
                    if not chunk:
                        break
                    try:
                        f.write(chunk)
                        f.flush()
                    except BrokenPipeError:
                        logger.info("Process closed fd/3, stopping feed")
                        break
                    except OSError as e:
                        logger.warning(f"Error writing to fd/3: {e}")
                        break
                    offset += len(chunk)
                    time.sleep(0.1)  # Real-time simulation
                
                logger.info(f"Finished feeding {offset:,} bytes")
        except Exception as e:
            logger.error(f"Error in feeder: {e}")

    # Start audio feed thread IMMEDIATELY
    feed_thread = threading.Thread(target=feeder, daemon=True)
    feed_thread.start()
    
    # Send AGI headers (to all binaries, not just Python scripts)
    # Send AGI headers AFTER starting the audio thread
    time.sleep(0.05)  # Small delay for initialization
    agi_env = (
        "agi_uniqueid: 1234567890\n"
        "agi_calleridname: 1234567890\n"
        f"agi_arg_1: {local_url}\n"
        f"agi_arg_2: {amd_duration_ms}\n"
        f"agi_arg_3: {amd_timeout_sec}\n"
        f"agi_arg_4: {vad_enabled}\n"
        f"agi_arg_5: {vad_aggressive}\n"
        "\n"
    )
    try:
        proc.stdin.write(agi_env.encode())
        proc.stdin.flush()
        logger.info("AGI headers sent")
    except BrokenPipeError:
        logger.warning("Process closed stdin before we could send AGI headers")
    except Exception as e:
        logger.error(f"Failed to send AGI headers: {e}")

    # 6. Wait for completion
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    
    server_data.stop_event.set()
    
    # 7. Analyze & Forward
    if server_data.received_audio:
        out_file = AUDIO_WAV
        save_wav(out_file, server_data.received_audio, rate)
        logger.info(f"Saved received audio to {out_file}")
        
        # Find offset
        # Simple heuristic: find the first 1000 bytes of received audio in original
        snippet = server_data.received_audio[:1000]
        idx = raw_audio.find(snippet)
        if idx >= 0:
            seconds = idx / (rate * 2) # 2 bytes per sample
            logger.info(f"MATCH FOUND! Voice started at {seconds:.3f} seconds in original file.")
        else:
            logger.warning("Could not find exact match in original file (maybe processed/transcoded?)")
            
        # Playback
        if args.play:
            logger.info("Playing back received audio...")
            subprocess.run(["aplay", out_file])
            
        # Waveform
        if args.image:
            display_waveform(server_data.received_audio, rate)

    else:
        logger.error("No audio received from amd_audio.")

    # Show logs
    stdout = proc.stdout.read().decode(errors='ignore')
    stderr = proc.stderr.read().decode(errors='ignore')
    if stdout: logger.info(f"STDOUT:\n{stdout}")
    if stderr: logger.info(f"STDERR:\n{stderr}")

if __name__ == "__main__":
    main()
