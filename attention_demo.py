"""Webcam demonstration of screen-facing behaviour. Press C to calibrate."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def face_offset(face):
    eyes = face[4:8].reshape(2, 2)
    axis = eyes[1] - eyes[0]
    distance = np.linalg.norm(axis)
    if distance < 12:
        return None
    horizontal = axis / distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    nose = (face[8:10] - eyes.mean(axis=0)) / distance
    return np.array([nose @ horizontal, nose @ vertical])


class Monitor:
    def __init__(self):
        self.center = None
        self.samples = []
        self.started = None
        self.away_since = None

    def update(self, offset, now):
        if offset is None:
            self.samples.clear()
            self.started = self.away_since = None
            return "Unknown"
        if self.center is None:
            if self.started is None:
                self.started = now
            self.samples.append(offset)
            if now - self.started < 2 or len(self.samples) < 8:
                return "Calibrating: face the screen"
            self.center = np.median(self.samples, axis=0)
            self.samples.clear()
        dx, dy = np.abs(offset - self.center)
        if dx <= 0.24 and dy <= 0.18:
            self.away_since = None
            return "Facing screen"
        if self.away_since is None:
            self.away_since = now
        return "Head turned" if now - self.away_since >= 1 else "Brief movement"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    model = Path(__file__).with_name("yunet.onnx")
    detector = cv2.FaceDetectorYN.create(str(model), "", (640, 480), 0.8)
    cam = cv2.VideoCapture(args.camera)
    monitor = Monitor()
    title = "Webcam attention demo"
    try:
        if not cam.isOpened():
            raise SystemExit("无法打开摄像头。请检查摄像头权限、占用情况或 --camera 编号。")
        cv2.namedWindow(title)
        while True:
            ok, frame = cam.read()
            if not ok:
                raise SystemExit("摄像头未返回画面。")
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, max(1, round(h * 640 / w))))
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)
            count = 0 if faces is None else len(faces)
            offset = None
            if count == 1:
                face = faces[0]
                offset = face_offset(face)
                x, y, w, h = face[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 180), 2)
                for point in face[4:14].reshape(5, 2).astype(int):
                    cv2.circle(frame, tuple(point), 3, (60, 200, 255), -1)
            status = monitor.update(offset, time.monotonic())
            if offset is None:
                status += {0: ": no face", 1: ": move closer"}.get(count, ": multiple faces")
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (640, 58), (30, 30, 30), -1)
            cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (220, 255, 215), 2)
            cv2.putText(frame, "C: recalibrate    Q: quit", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.49, (230, 230, 230), 1)
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("c"), ord("C")):
                monitor = Monitor()
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
