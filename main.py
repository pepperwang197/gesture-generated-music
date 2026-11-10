"""
Gesture-controlled soundfont theremin. Tracks both hands with a webcam, and
for each hand maps fingertip height to pitch, fingertip horizontal position
to loudness, and the farthest extended finger to the instrument.
"""

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

# Soundfonts in gesture order: thumb, index, middle, ring, pinky
instruments = [
    "soundfonts/Full_Grand_Piano.sf2",
    "soundfonts/JazzFlute.sf2",
    "soundfonts/Wind_of_Fjords_MIDI.sf2",
    "soundfonts/The_Great_Strategy_MIDI.sf2",
    "soundfonts/Nokia_Tongbao_Bank.sf2"
]

# One synth per instrument, all started up front so switching is instant
synths = []

for instrument in instruments:
    synth = tinysoundfont.Synth()
    sfid = synth.sfload(instrument)
    synth.program_select(0, sfid, 0, 0)
    synth.start()
    synths.append(synth)

def download_model():
    """
    Downloads the MediaPipe hand landmarker model to MODEL_PATH if it is not
    already there.
    """
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model (~12 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def draw_hand(frame, hand_landmarks):
    """
    Draws a green dot on the frame at each landmark of a hand.
    """
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
    Given a list of landmark values, shifts them so the wrist is at zero and
    divides by the wrist to index knuckle distance, so the result does not
    depend on hand position or distance from the camera.
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
    based on which fingertip is the farthest past the others. Returns -1 if
    no single fingertip is farthest.
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
    """
    Runs the theremin. Reads webcam frames, finds the hands with MediaPipe, and
    when two hands are visible plays a note for each one. The gesture picks the
    instrument, the index fingertip height picks the note, and its horizontal
    position sets the volume. Press q in the video window to quit.
    """
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

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror the frame so on-screen movement matches the player's
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_image, timestamp_ms)

            for hand_landmarks in results.hand_landmarks:
                draw_hand(frame, hand_landmarks)

            if len(results.hand_landmarks) >= 2:
                # Instrument for each hand
                hand_l = (element.y for element in results.hand_landmarks[0])
                gesture_l = detect_gesture(hand_l)
                hand_r = (element.y for element in results.hand_landmarks[1])
                gesture_r = detect_gesture(hand_r)

                # Index fingertip: y gives the note, x gives the volume
                note_l = position_to_note(results.hand_landmarks[0][8].y)
                velocity_l = results.hand_landmarks[0][8].x
                note_r = position_to_note(results.hand_landmarks[1][8].y)
                velocity_r = results.hand_landmarks[1][8].x
                synths[gesture_l].notes_off(0)
                synths[gesture_r].notes_off(0)
                synths[gesture_l].noteon(0, round(note_l), velocity_l*127)
                synths[gesture_r].noteon(0, round(note_r), velocity_r*127)

                # Pitch bend makes up the fraction of a semitone lost by
                # round(). 8192 is no bend, and 4096 is one semitone.
                diff_l = note_l%1
                diff_r = note_r%1
                if diff_l < 0.5:
                    synths[gesture_l].pitchbend(0, int(8192 + diff_l*2048*2))
                    print(int(8192 + diff_l*1024))
                elif diff_l >=0.5:
                    synths[gesture_l].pitchbend(0, int(8192 - (1-diff_l)*2048*2))
                    print(int(8192 - (1-diff_l)*1024))
                if diff_r < 0.5:
                    synths[gesture_r].pitchbend(0, int(8192 + diff_r*2048*2))
                    print(int(8192 + diff_r*1024))
                elif diff_r >=0.5:
                    synths[gesture_r].pitchbend(0, int(8192 - (1-diff_r)*2048*2))
                    print(int(8192 - (1-diff_r)*1024))
                #time.sleep(0.01)
            else:
                synth.notes_off(0)
            
            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()