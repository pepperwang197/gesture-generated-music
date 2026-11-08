import os
import time
import urllib.request
import tinysoundfont

from pysinewave import SineWave
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
  HandLandmarker,
  HandLandmarkerOptions,
  RunningMode,
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Landmark indices (MediaPipe hand topology, 0–20)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

instruments = [
    "soundfonts/Full_Grand_Piano.sf2",
    "soundfonts/JazzFlute.sf2",
    "soundfonts/Wind_of_Fjords_MIDI.sf2",
    "soundfonts/The_Great_Strategy_MIDI.sf2",
    "soundfonts/Nokia_Tongbao_Bank.sf2"
]

synths = []

for instrument in instruments:
    synth = tinysoundfont.Synth()
    sfid = synth.sfload(instrument)
    synth.program_select(0, sfid, 0, 0)
    synth.start()
    synths.append(synth)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model (~12 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]

    for landmark in hand_landmarks:
        pt = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, pt, 5, (0, 220, 0), -1, cv2.LINE_AA)

def position_to_note(position):
    """
    Takes a number between 0 and 1 and maps it to the MIDI number of a note
    between C3 and C5, where a lower number is a higher note.
    """
    return 72 - position * 24

def normalize(landmarks):
    """
    """
    landmark_list = list(landmarks)
    wrist = landmark_list[0]
    translated = np.subtract(landmark_list,wrist)
    mcp = translated[5]
    wrist_t = translated[0]
    scale_factor = np.linalg.norm(mcp - wrist)
    if scale_factor == 0:
        scale_factor = 0.000001
    return translated/scale_factor

def detect_gesture(landmarks):
    """
    Given a list of landmarks, return a number between 0 and 4 (inclusive) 
    based on which fingertip is the farthest past the others.
    """
    n_landmarks = normalize(landmarks)
    if (n_landmarks[4] < n_landmarks[8] and n_landmarks[4] < n_landmarks[12]
        and n_landmarks[4]<n_landmarks[16] and n_landmarks[4]<n_landmarks[20]):
        gesture = 0
    elif (n_landmarks[8] < n_landmarks[4] and n_landmarks[8] < n_landmarks[12]
            and n_landmarks[8]<n_landmarks[16] and n_landmarks[8]<n_landmarks[20]):
        gesture = 1
    elif (n_landmarks[12] < n_landmarks[4] and n_landmarks[12] < n_landmarks[8]
                and n_landmarks[12]<n_landmarks[16] and n_landmarks[12]<n_landmarks[20]):
        gesture = 2
    elif (n_landmarks[16] < n_landmarks[4] and n_landmarks[16] < n_landmarks[8]
                and n_landmarks[16]<n_landmarks[12] and n_landmarks[16]<n_landmarks[20]):
        gesture = 3
    elif (n_landmarks[20] < n_landmarks[4] and n_landmarks[20] < n_landmarks[8]
                    and n_landmarks[20]<n_landmarks[12] and n_landmarks[20]<n_landmarks[16]):
        gesture = 4
    else:
        gesture = -1
    return gesture

def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    start_time = time.time()
    prev_tick = cv2.getTickCount()

    sinewave = SineWave(decibels_per_second = 10)

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_image, timestamp_ms)
            
            for hand_landmarks in results.hand_landmarks:
                draw_hand(frame, hand_landmarks)

            if len(results.hand_landmarks) >= 2:
                handL = (element.y for element in results.hand_landmarks[0])
                gestureL = detect_gesture(handL)
                handR = (element.y for element in results.hand_landmarks[1])
                gestureR = detect_gesture(handR)

                noteL = position_to_note(results.hand_landmarks[0][8].y)
                velocityL = results.hand_landmarks[0][8].x
                noteR = position_to_note(results.hand_landmarks[1][8].y)
                velocityR = results.hand_landmarks[1][8].x
                synths[gestureL].notes_off(0)
                synths[gestureR].notes_off(0)
                synths[gestureL].noteon(0, round(noteL), velocityL*127)
                synths[gestureR].noteon(0, round(noteR), velocityR*127)
                diffL = noteL%1
                diffR = noteR%1
                if diffL < 0.5:
                    synths[gestureL].pitchbend(0, int(8192 + diffL*2048*2))
                    print(int(8192 + diffL*1024))
                elif diffL >=0.5:
                    synths[gestureL].pitchbend(0, int(8192 - (1-diffL)*2048*2))
                    print(int(8192 - (1-diffL)*1024))
                if diffR < 0.5:
                    synths[gestureR].pitchbend(0, int(8192 + diffR*2048*2))
                    print(int(8192 + diffR*1024))
                elif diffR >=0.5:
                    synths[gestureR].pitchbend(0, int(8192 - (1-diffR)*2048*2))
                    print(int(8192 - (1-diffR)*1024))
                #time.sleep(0.01)
            else:
                #sinewave.stop()
                synth.notes_off(0)
            
            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()