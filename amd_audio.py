#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install pyst2 requests numpy
import base64
import os
import sys
import json
from asterisk.agi import AGI  # type: ignore
from asterisk.agi import AGISIGHUPHangup  # type: ignore
import logging
import time
from multiprocessing import Pipe, Process
from urllib.request import urlopen, Request
import numpy as np
import math
import traceback

URL = "http://10.4.100.245:9000/a/"
REC_SECONDS = 2.0
TIMEOUT = 10
RATE = 16000
FRAME_SIZE = 1600  # 0.1 sec
SILENCE_THRESHOLD = 0.025
ZCR_THRESHOLD = 0.1
ENERGY_THRESHOLD = 0.02

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def decode_ulaw(frame):
    # decode μ-law to 16 bit linear PCM
    ulaw_table = np.array(
        [((256 | i) << 7) - 32768 for i in range(256)], dtype=np.int16
    )
    return ulaw_table[frame]


def is_silence1(frame, uniqueid, start_time, is_ulaw=False):
    # Check if the audio frame is empty
    if not frame:
        return True
    if is_ulaw:
        # If frame in ulaw format then use decode_ulaw
        ulaw_data = np.frombuffer(frame, dtype=np.uint8)
        audio_data = np.array([decode_ulaw(byte) for byte in ulaw_data], dtype=np.int16)
    else:
        # Convert the audio frame into an array of int16 data
        audio_data = np.frombuffer(frame, dtype=np.int16)
    if len(audio_data) == 0:
        return True
    # Normalize the data so that the values are in the range from -1 to 1
    normalized = audio_data / np.iinfo(np.int16).max
    # Calculate the energy of the audio frame
    # Energy is the sum of the squares of the normalized values divided by the number of values
    energy = np.mean(normalized.astype(np.float32) ** 2)
    # Calculate the zero-crossing rate (ZCR)
    # ZCR is the number of sign changes in the normalized values divided by twice the number of values
    zcr = np.sum(np.abs(np.diff(np.sign(normalized)))) / (2 * len(normalized))
    # Calculate the threshold value
    # The threshold is determined as the sum of ZCR multiplied by the ZCR_THRESHOLD constant
    # and the energy multiplied by the ENERGY_THRESHOLD constant
    threshold = ZCR_THRESHOLD * zcr + ENERGY_THRESHOLD
    logger.debug(
        f"{uniqueid} {time.monotonic() - start_time:.2f} sec.: Energy: {energy:.3f}, ZCR: {zcr:.3f}, Threshold: {threshold:.3f}"
    )
    # Return True if the energy is less than the threshold value (i.e., the audio frame is considered silence)
    return threshold > energy


def is_silence2(frame, uniqueid, start_time, is_ulaw=False):
    # Check if the audio frame is empty
    if not frame:
        return True
    if is_ulaw:
        # If frame in ulaw format then use decode_ulaw
        ulaw_data = np.frombuffer(frame, dtype=np.uint8)
        audio_data = np.array([decode_ulaw(byte) for byte in ulaw_data], dtype=np.int16)
    else:
        # Convert the audio frame into an array of int16 data
        audio_data = np.frombuffer(frame, dtype=np.int16)
    if len(audio_data) == 0:
        return True
    normalized = audio_data / np.iinfo(np.int16).max
    # Calculate the average amplitude of the audio frame
    average_amplitude = np.mean(np.abs(normalized.astype(np.float32)))
    # GPU version (requires GPU and cupy)
    # import cupy as cp
    # audio_data = cp.frombuffer(frame, dtype=cp.int16)
    # normalized = audio_data / cp.iinfo(cp.int16).max
    # average_amplitude = cp.mean(cp.abs(normalized)).get()
    # We can use static threshold for low CPU usage
    # return SILENCE_THRESHOLD > average_amplitude
    # Calculate the zero-crossing rate (ZCR)
    # ZCR is the number of sign changes in the normalized values divided by twice the number of values
    zcr = np.sum(np.abs(np.diff(np.sign(normalized)))) / (2 * len(normalized))
    # Calculate the threshold value
    # The threshold is determined as the sum of ZCR multiplied by the ZCR_THRESHOLD constant
    # and the energy multiplied by the ENERGY_THRESHOLD constant
    threshold = ZCR_THRESHOLD * zcr + ENERGY_THRESHOLD
    logger.debug(
        f"{uniqueid} {time.monotonic() - start_time:.2f} sec.: Average Amplitude: {average_amplitude:.3f}, Threshold: {threshold:.3f}"
    )
    return threshold > average_amplitude


def read_data(
    f, uniqueid, rec_seconds=REC_SECONDS, timeout=TIMEOUT, vad_enabled=True, vad_mode=1
):
    rec_size = int(RATE * rec_seconds)  # bytes
    if not vad_enabled:
        data = f.read(rec_size)
        return data
    if vad_mode == 1:
        is_silence = is_silence1
    else:
        is_silence = is_silence2

    def sender(conn, f):
        start_time = time.monotonic()
        frame, last_frame = None, None
        frames = []
        try:
            logger.info(f"{uniqueid} Read started!")
            while (time.monotonic() - start_time) < timeout:
                last_frame = frame if frame else None
                frame = f.read(FRAME_SIZE)
                if not is_silence(frame, uniqueid, start_time):
                    logger.info(
                        f"{uniqueid} Rec started from {time.monotonic() - start_time:.2f} sec."
                    )
                    frames.append(last_frame) if last_frame else None
                    frames.append(frame)
                    frame = f.read(rec_size - FRAME_SIZE * len(frames))
                    frames.append(frame)
                    break
            if frames:
                conn.send(b"".join(frames))
            else:
                conn.send(b"")
        except Exception as exc:
            logger.error(f"{uniqueid} {exc}\n{traceback.format_exc()}")
            conn.send(b"")
        finally:
            conn.close()

    conn1, conn2 = Pipe()
    p = Process(target=sender, args=(conn2, f))
    p.start()
    p.join(timeout=timeout + math.ceil(rec_seconds))
    if p.is_alive():
        p.terminate()
        p.join()
        logger.warning("Read timeout!")
        return b""
    if conn1.poll():
        data = conn1.recv()
        return data
    else:
        return b""


def main():
    # URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = URL
    # Recording time in milliseconds
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        rec_seconds = int(sys.argv[2]) / 1000
    else:
        rec_seconds = REC_SECONDS
    # Timeout execution in seconds
    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        timeout = int(sys.argv[3])
    else:
        timeout = TIMEOUT
    # Enabled VAD
    if len(sys.argv) > 4 and sys.argv[4] == "0":
        vad_enabled = False
    else:
        vad_enabled = True
    # VAD mode
    if len(sys.argv) > 5 and sys.argv[5].isdigit():
        vad_mode = int(sys.argv[5])
    else:
        # Golang script has three modes, we use mode 2
        vad_mode = 2
    exit_code = 0
    start_time = time.monotonic()
    agi = AGI()
    uniqueid = agi.env["agi_uniqueid"]
    lead_id = agi.env["agi_calleridname"]
    try:
        with os.fdopen(3, "rb") as f:
            audio = read_data(
                f,
                uniqueid,
                rec_seconds=rec_seconds,
                timeout=timeout,
                vad_enabled=vad_enabled,
                vad_mode=vad_mode,
            )
            logger.info(f"{uniqueid} {len(audio)} bytes")
            if audio:
                payload = {
                    "uniqueid": uniqueid,
                    "audio": base64.b64encode(audio).decode(),
                    "lead_id": int(lead_id[-10:]),
                    "host": os.uname().nodename,
                }
                data = json.dumps(payload).encode("utf-8")
                req = Request(
                    url=url, method="POST", headers={"Content-Type": "application/json"}
                )
                with urlopen(req, timeout=timeout, data=data) as r:
                    resp = r.read().decode("utf-8")
                logger.info(f"{uniqueid} {resp}")
                status = json.loads(resp)["answer"]
                agi.verbose(status)
                agi.set_variable("AMDSTATUS", status)
                agi.set_variable("AMDSTATS", "0000000000")
                if status == "HUMAN":
                    agi.set_variable("AMDCAUSE", "HUMAN-1000-1000")
                else:
                    agi.set_variable("AMDCAUSE", "MAXWORDS-4-4")
            else:
                logger.warning(f"{uniqueid} Empty data received!")
    except AGISIGHUPHangup:
        logger.warning(f"{uniqueid} Hangup detected!")
    except Exception as exc:
        logger.error(f"{uniqueid} {exc}\n{traceback.format_exc()}")
        exit_code = 1
    finally:
        logger.info(
            f"{uniqueid} Time elapsed: {time.monotonic() - start_time:.2f} sec. Exit code: {exit_code}"
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
