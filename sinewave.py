"""
Gesture-controlled sine wave theremin. Tracks both hands with a webcam. The
height of the first hand's index fingertip sets the pitch, and the horizontal
position of the second hand's index fingertip sets the volume.
"""

import os
import time
import urllib.request
import tinysoundfont

from pysinewave import SineWave

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


def download_model():
    """
    Downloads the MediaPipe hand landmarker model to MODEL_PATH if it is not
    already there.
    """
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model (~12 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def landmark_px(landmark, w, h):
    """
    Converts a landmark's normalized x and y (0 to 1) into pixel coordinates
    for a frame that is w pixels wide and h pixels tall.
    """
    return int(landmark.x * w), int(landmark.y * h)


def draw_hand(frame, hand_landmarks):
    """
    Draws a green dot on the frame at each landmark of a hand.
    """
    h, w = frame.shape[:2]

    for landmark in hand_landmarks:
        pt = landmark_px(landmark, w, h)
        cv2.circle(frame, pt, 5, (0, 220, 0), -1, cv2.LINE_AA)


def main():
    """
    Runs the theremin. Reads webcam frames and finds the hands with MediaPipe.
    When two hands are visible, the first hand's index fingertip height is
    converted to a note and then a frequency for the sine wave, and the second
    hand's index fingertip horizontal position sets the volume in decibels
    (-60 at the left edge to 0 at the right edge). The sound stops when fewer
    than two hands are visible. Press q in the video window to quit.
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
    prev_tick = cv2.getTickCount()

    # decibels_per_second is how fast the volume moves toward its target
    sinewave = SineWave(decibels_per_second = 10)

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
            if len(results.hand_landmarks) >= 2:
                for hand_landmarks in results.hand_landmarks:
                    draw_hand(frame, hand_landmarks)
                curr_tick = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
                prev_tick = curr_tick

                # Hand height to MIDI note (C3 to C5), then note to frequency
                note = 72 - results.hand_landmarks[0][8].y*24
                freq = 440*2**((note-69)/12)
                #print(freq)
                sinewave.set_frequency(freq)

                # Second hand's horizontal position to volume in decibels
                velocity =-60+results.hand_landmarks[1][8].x * 60
                sinewave.set_volume(velocity)
                sinewave.play()
                #synth.noteon(0, note, velocity)
                print("note:", note, "velocity:", velocity)
                #print("sinewave:",sinewave.decibels_per_second)
            else:
                sinewave.stop()
            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()