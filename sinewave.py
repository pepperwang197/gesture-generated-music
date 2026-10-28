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
  if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model (~12 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def landmark_px(landmark, w, h):
  return int(landmark.x * w), int(landmark.y * h)


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]

    for landmark in hand_landmarks:
        pt = landmark_px(landmark, w, h)
        cv2.circle(frame, pt, 5, (0, 220, 0), -1, cv2.LINE_AA)


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
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                draw_hand(frame, hand_landmarks)

            curr_tick = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
            prev_tick = curr_tick
            note = 72 - results.hand_landmarks[0][8].y*24
            freq = 440*2**((note-69)/12)
            #print(freq)
            sinewave.set_frequency(freq)
            velocity = results.hand_landmarks[0][8].x * 50+0
            sinewave.set_volume(velocity)
            sinewave.play()
            #synth.noteon(0, note, velocity)
            print("note:", note, "velocity:", velocity)
            #print("sinewave:",sinewave.decibels_per_second)
            
        cv2.imshow("Hand Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()