# Drone detection: plain colour vs multi-layer input

Run in this folder (`code/` inside your "drone detecting" folder):

    pip install ultralytics opencv-python
    python prep_data.py --src ../hackathon_dataset --dst drone_data
    python train.py --variant rgb --epochs 30
    python train.py --variant ml  --epochs 30
    python run_video.py --weights runs/rgb/weights/best.pt --videos ../test_sample_clean --out preds/rgb
    python run_video.py --weights runs/ml/weights/best.pt  --videos ../test_sample_clean --out preds/ml
    python score.py --pred preds/rgb --gt ../test_sample_clean
    python score.py --pred preds/ml  --gt ../test_sample_clean

- `rgb`: normal YOLO11n on colour images.
- `ml`: same model, but its first layer sees 5 layers: R, G, B, grayscale and inverted grayscale.
  The extra layers are computed inside the model, so augmentation is identical for both.
- Both are trained only on hackathon_dataset; the 3 practice videos are used only for scoring.

## Test it yourself (pretrained SkyGuard + tracker)

    source .venv/bin/activate
    pip install scipy motmetrics          # once
    python demo.py ../test_sample_clean/test_6_hard_PXL_20261113_154422245_hard/video.mp4 --show

Opens a live window (q quits) and saves `demo_out/<video>.mp4` with boxes and ids
(orange = drone 1, blue = drone 2, thin white = the correct answer from gt) plus a result file.
Works on any video: `python demo.py some_other_video.mp4 --show`. Add `--no-track` to see the raw detector.

Score all 3 practice videos with the tracker:

    python run_video.py --weights ../pretrained/drone-detection-yolov11/best.pt --videos ../test_sample_clean --out preds/hf_skyguard --imgsz 1280 --conf 0.01 --keep drone   # (already done)
    python run_tracker.py --dets preds/hf_skyguard --out preds/skyguard_trk
    python eval_track.py --pred preds/skyguard_trk --gt ../test_sample_clean
