#!/usr/bin/env python3
import argparse
import time
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from scipy.spatial import distance  # optional; only used for speed readout

# your modules
from model import BallTrackerNet
from general import postprocess


def safe_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_camera(source: str):
    """
    Open a camera by index ('0', '1', ...) or by path ('/dev/video0').
    """
    # Allow integer indices or direct string paths
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)

    return cap


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: Optional[int]):
    """
    Attempt to set capture properties. Not all drivers will honor these.
    """
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)


def make_writer(path: str, width: int, height: int, fps: float):
    # Fallback to mp4v for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def draw_trace(frame, trail: List[Tuple[Optional[float], Optional[float]]], max_len: int = 12):
    """
    Draw a fading tail of recent ball positions onto the frame.
    """
    h, w = frame.shape[:2]
    # last max_len points
    last = trail[-max_len:] if len(trail) > max_len else trail
    thickness_base = 10
    for i, pt in enumerate(reversed(last)):
        x, y = pt
        if x is None or y is None:
            continue
        # fade thickness
        thickness = max(1, thickness_base - i)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)


def main():
    parser = argparse.ArgumentParser(description="Live ball tracking with BallTrackerNet.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to model .pth/.pt")
    parser.add_argument("--source", type=str, default="/dev/video0", help="Camera index or device path (e.g., 0 or /dev/video0)")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--save_path", type=str, default="", help="Optional path to save output video (e.g., out.mp4)")
    parser.add_argument("--capture_w", type=int, default=1920, help="Try to set camera capture width")
    parser.add_argument("--capture_h", type=int, default=1080, help="Try to set camera capture height")
    parser.add_argument("--capture_fps", type=int, default=30, help="Try to set camera fps")
    parser.add_argument("--model_w", type=int, default=640, help="Model input width")
    parser.add_argument("--model_h", type=int, default=360, help="Model input height")
    parser.add_argument("--trace", type=int, default=7, help="Number of frames to show trailing trace")
    parser.add_argument("--flip", type=int, default=-1, choices=[-1, 0, 1], help="Flip code for cv2.flip (-1 none, 0 vertical, 1 horizontal). Use -1 for no flip.")
    args = parser.parse_args()

    device = safe_device()
    print(f"[info] torch device: {device}")

    # Load model
    print("[info] loading model…")
    model = BallTrackerNet()
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("[info] model loaded.")

    # Open camera
    print(f"[info] opening source: {args.source}")
    cap = open_camera(args.source)
    if not cap.isOpened():
        print("[error] could not open camera/source. Check --source.")
        sys.exit(1)

    configure_capture(cap, args.capture_w, args.capture_h, args.capture_fps)
    # Probe actual size/fps
    ret, frame = cap.read()
    if not ret:
        print("[error] could not read from source.")
        sys.exit(1)

    if args.flip in (0, 1):
        frame = cv2.flip(frame, args.flip)

    out_h, out_w = frame.shape[:2][0], frame.shape[:2][1]
    # Video writer (optional)
    writer = None
    if args.save_path:
        # Use camera reported fps if available; fallback to arg
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        if not cam_fps or cam_fps <= 1:
            cam_fps = float(args.capture_fps)
        writer = make_writer(args.save_path, out_w, out_h, cam_fps)
        if not writer.isOpened():
            print("[warn] could not open writer; disabling save.")
            writer = None
        else:
            print(f"[info] saving to {args.save_path} at ~{cam_fps:.1f} fps")

    # Keep last two resized frames for the 3-frame stack (cur, prev, preprev)
    prev = None
    preprev = None

    # trail for drawing
    trail: List[Tuple[Optional[float], Optional[float]]] = []

    # simple FPS meter
    last_t = time.time()
    smoothed_fps = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[info] end of stream / cannot read.")
                break

            if args.flip in (0, 1):
                frame = cv2.flip(frame, args.flip)

            # Build 3-frame stack
            img = cv2.resize(frame, (args.model_w, args.model_h))
            if prev is None:
                # not enough history yet → initialize and continue
                prev = img.copy()
                preprev = img.copy()
                # optional: draw nothing first frame
                if args.show:
                    cv2.imshow("Ball Tracking (warming up)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if writer:
                    writer.write(frame)
                continue

            if preprev is None:
                preprev = img.copy()

            # Concatenate channels: (H,W,3) x 3 → (H,W,9), then CHW
            imgs = np.concatenate((img, prev, preprev), axis=2).astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)  # (9,H,W)
            inp = np.expand_dims(imgs, axis=0)  # (1,9,H,W)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(device))
                output = out.argmax(dim=1).detach().cpu().numpy()  # (1,H,W)
                x_pred, y_pred = postprocess(output)  # expected to return a single (x,y)

            # Save current as new history
            preprev = prev
            prev = img

            # Draw
            if x_pred and y_pred:
                trail.append((float(x_pred) * (out_w / args.model_w), float(y_pred) * (out_h / args.model_h)))
            else:
                trail.append((None, None))

            # keep trail bounded
            if len(trail) > 1000:
                trail = trail[-1000:]

            vis = frame.copy()
            draw_trace(vis, trail, max_len=args.trace)

            # FPS overlay
            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_t)
            last_t = now
            smoothed_fps = inst_fps if smoothed_fps is None else (0.9 * smoothed_fps + 0.1 * inst_fps)

            # speed (if last two valid points exist)
            speed_px = ""
            if len(trail) >= 2 and trail[-1][0] is not None and trail[-2][0] is not None:
                speed_px = f" v≈{distance.euclidean(trail[-1], trail[-2]):.1f}px/frame"

            cv2.putText(vis, f"FPS: {smoothed_fps:.1f}{speed_px}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow("Ball Tracking (press q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(vis)

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
