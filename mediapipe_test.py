import os
import time
import urllib.request

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

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

FINGER_LANDMARK_GROUPS = [
  [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
  [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
  [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
  [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
  [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
]

FINGER_COLORS = [
  (0, 140, 255),   # Thumb  — orange
  (0, 220, 0),     # Index  — green
  (220, 0, 220),   # Middle — magenta
  (0, 220, 220),   # Ring   — cyan
  (220, 220, 0),   # Pinky  — yellow
]


def download_model():
  if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model (~12 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def landmark_px(landmark, w, h):
  return int(landmark.x * w), int(landmark.y * h)


def draw_hand(frame, hand_landmarks):
  h, w = frame.shape[:2]
  lm = hand_landmarks

  for finger_idx, chain in enumerate(FINGER_LANDMARK_GROUPS):
    color = FINGER_COLORS[finger_idx]
    pts = [landmark_px(lm[i], w, h) for i in chain]
    for a, b in zip(pts, pts[1:]):
      cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
    for pt in pts[1:]:  # skip wrist repetition
      cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)

  # Palm cross-connections in white
  for i, j in [(INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP)]:
    cv2.line(frame, landmark_px(lm[i], w, h), landmark_px(lm[j], w, h), (255, 255, 255), 2, cv2.LINE_AA)

  # Wrist dot
  cv2.circle(frame, landmark_px(lm[WRIST], w, h), 7, (255, 255, 255), -1, cv2.LINE_AA)

  # Fingertip circles
  for tip_id, color in zip(FINGER_TIPS, FINGER_COLORS):
    cv2.circle(frame, landmark_px(lm[tip_id], w, h), 9, color, 2, cv2.LINE_AA)


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
      if len(results.hand_landmarks) > 0:
        print(results.hand_landmarks[0][8])
      #time.sleep(300)

      if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
          draw_hand(frame, hand_landmarks)

      curr_tick = cv2.getTickCount()
      fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
      prev_tick = curr_tick
      cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

      cv2.imshow("Hand Tracking", frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      if key == ord("s"):
        filename = f"frame_{int(time.time() * 1000)}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()