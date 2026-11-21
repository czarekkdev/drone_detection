"""See the detector + tracker + prediction crops + motion detector working, with switchable views.
  python demo.py ../test_sample_clean/test_6_hard_PXL_20261113_154422245_hard/video.mp4 --show
  python demo.py 0 --show            (live webcam / camera 0)
Keys in the window:
  1 camera view   2 difference image   3 motion mask   4 four-way: camera, difference, motion mask, background diff
  5 short diff   6 long diff   7 background diff   8 tracker map   9 previous frame   b brightness shape
  0 grid of camera + all 6 layers
  (layers 5-0 are the extra inputs for the 'B' model; they warm up for ~10 frames after you first switch to them)
  t trajectory on/off (trail behind + predicted path ahead)   m motion on/off   p prediction crops on/off   o helper boxes on/off (off at start)   space pause   q quit
Colours: orange/blue = drone 1/2 (final output), thin white = correct answer (gt, if the video has one),
  green squares = crops checked at a predicted spot, cyan squares = crops checked at a motion spot,
  small yellow boxes = moving spots, yellow + = where the tracker expects each drone.
Also writes demo_out/<name>.mp4 (what you see) and demo_out/<name>.txt (result file)."""
import argparse, os, time
import cv2, torch
from ultralytics import YOLO
from stack import DroneStack
import layers as LY

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt")
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--show", action="store_true")
p.add_argument("--view", type=int, default=1, help="start view 0-10 (10 = brightness shape, key b)")
p.add_argument("--motion", action="store_true", help="start with motion crops on (press m to toggle)"); p.add_argument("--no-pred", action="store_true")
p.add_argument("--out", default="demo_out")
p.add_argument("--no-save", action="store_true", help="don't write the output video (faster)")
a = p.parse_args()
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
src = int(a.video) if a.video.isdigit() else a.video
cap = cv2.VideoCapture(src)
W, H = int(cap.get(3)), int(cap.get(4)); fps_in = cap.get(5) or 30
if isinstance(src, int):
    name = f"camera{src}"; vdir = ""
else:
    vdir = os.path.dirname(os.path.abspath(a.video))
    name = os.path.basename(vdir) if os.path.basename(a.video) == "video.mp4" else os.path.splitext(os.path.basename(a.video))[0]
gt = {}
if vdir and os.path.exists(os.path.join(vdir, "gt", "gt.txt")):
    for l in open(os.path.join(vdir, "gt", "gt.txt")):
        v = l.split(","); gt.setdefault(int(v[0]), []).append(tuple(map(float, v[2:6])))
S = DroneStack(YOLO(a.weights), device=dev, imgsz=a.imgsz, use_pred=not a.no_pred, use_motion=a.motion, frame_size=(W, H))
os.makedirs(a.out, exist_ok=True)
writer = cv2.VideoWriter(os.path.join(a.out, name + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
COL = {1: (0, 165, 255), 2: (255, 120, 0)}
view, overlays, paused = a.view, False, False
lines, n, t0 = [], 0, time.time()
VIEWS = {1: "camera", 2: "difference image", 3: "motion mask", 4: "camera | difference / mask | background",
         5: "short diff", 6: "long diff", 7: "background diff", 8: "tracker map", 9: "previous frame", 10: "brightness shape", 0: "all layers"}
LAYER_KEY = {5: "short", 6: "long", 7: "bg", 8: "track", 9: "prev", 10: "shape"}
LM, L = None, None
traj, TRAIL = True, {}  # TRAIL: id -> last 30 centres


def draw(img, out):
    I = S.info
    for (x, y, w, h) in gt.get(n, []):
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)
    if overlays:
        for x, y, w, h, *_ in I.get("spots", []):
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 230, 230), 1)
        for r in I.get("pred_crops", []):
            cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)
        for r in I.get("motion_crops", []):
            cv2.rectangle(img, r[:2], r[2:], (255, 255, 0), 2)
        for _, pb, _ in I.get("preds", []):
            cx, cy = int((pb[0] + pb[2]) / 2), int((pb[1] + pb[3]) / 2)
            cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    if traj:
        for i, pts in TRAIL.items():
            col = COL.get(i, (0, 255, 0))
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], col, 2)
        for i, path in S.T.future_paths(15):
            col = COL.get(i, (0, 255, 0)); pts = [(int(x), int(y)) for x, y in path]
            for k in range(1, len(pts), 2):
                cv2.line(img, pts[k - 1], pts[k], col, 2)  # dashed = predicted
            cv2.circle(img, pts[-1], 6, col, 2)
    for f, i, (x1, y1, x2, y2), c in out:
        if f == n:
            col = COL.get(i, (0, 255, 0))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            cv2.putText(img, f"drone {i} {c:.2f}", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return img


frame = None
while True:
    if not paused or frame is None:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        S.need_motion_view = view in (2, 3, 4)
        preds_before = [pb for _, pb, _ in S.T.predictions()]
        out = S.step(frame)
        if view in LAYER_KEY or view in (0, 4) or LM is not None:  # once started, keep it running so history stays valid
            LM = LM or LY.LayerMaker()
            L = LM.step(frame, preds_before)
        seen = set()
        for f, i, (x1, y1, x2, y2), c in out:
            if f == n:
                seen.add(i); TRAIL.setdefault(i, []).append((int((x1 + x2) / 2), int((y1 + y2) / 2))); TRAIL[i] = TRAIL[i][-30:]
        for i in list(TRAIL):
            if i not in seen:
                TRAIL[i] = TRAIL[i][1:]  # fade the trail out while the drone is gone
                if not TRAIL[i]:
                    del TRAIL[i]
        for f, i, (x1, y1, x2, y2), c in out:
            lines.append(f"{f},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    cam = draw(frame.copy(), out)
    if view == 2:
        img = draw(S.MD.diff_view((W, H)), out)
    elif view == 3:
        img = draw(S.MD.mask_view((W, H)), out)
    elif view == 4:
        hw, hh = W // 2, H // 2
        tiles = [cam, draw(S.MD.diff_view((W, H)), out), draw(S.MD.mask_view((W, H)), out),
                 draw(LY.view(L, "bg", (W, H)), out) if L is not None else cam]
        tiles = [cv2.resize(t, (hw, hh)) for t in tiles]
        img = cv2.resize(cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])]), (W, H))
    elif view in LAYER_KEY and L is not None:
        img = draw(LY.view(L, LAYER_KEY[view], (W, H)), out)
    elif view == 0 and L is not None:
        img = LY.grid(L, cam, (W, H))
    else:
        img = cam
    fps = n / (time.time() - t0)
    cv2.putText(img, f"frame {n}  {fps:.0f} fps  view {view}: {VIEWS[view]}  motion {'on' if S.use_motion else 'off'}  pred {'on' if S.use_pred else 'off'}  traj {'on' if traj else 'off'}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if not paused and not a.no_save:
        writer.write(img)
    if a.show:
        cv2.imshow("drone demo", cv2.resize(img, (W // 2, H // 2)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif ord("0") <= k <= ord("9"):
            view = k - ord("0")
        elif k == ord("b"):
            view = 10
        elif k == ord("t"):
            traj = not traj
        elif k == ord("m"):
            S.use_motion = not S.use_motion
        elif k == ord("p"):
            S.use_pred = not S.use_pred
        elif k == ord("o"):
            overlays = not overlays
        elif k == ord(" "):
            paused = not paused
writer.release(); cap.release(); cv2.destroyAllWindows()
lines = sorted(set(lines), key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
open(os.path.join(a.out, name + ".txt"), "w").write("\n".join(lines) + "\n")
print(f"{n} frames, {n / max(time.time() - t0, 1e-6):.1f} fps. Saved {a.out}/{name}.mp4 and {a.out}/{name}.txt")
