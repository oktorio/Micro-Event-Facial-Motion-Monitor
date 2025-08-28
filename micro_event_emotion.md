# Micro-Event Facial Motion Monitor (Demo)

**Windows-friendly GUI app** that uses your webcam to estimate **brief facial motion bursts** in three regions: **brows**, **eyes/eyelids**, and **mouth**, as a *very rough* proxy for micro-expression-like events.

> ⚠️ **Important**: This is **NOT** a lie detector. Do **not** use it to decide truthfulness. Use with informed consent, good lighting, and (ideally) a high‑fps camera for better sensitivity.

## Features
- Live webcam preview with **face mesh** overlay (MediaPipe)
- Per-frame motion metrics for **brows/eyes/mouth**
- **Burst detector** compares a short recent window (~200ms) vs the immediately prior window
- **Adjustable thresholds** for each region
- **Baseline calibration (10s)** to normalize per-person motion
- **CSV export** (timestamp, motions, and baseline-adjusted columns)

## Install (Windows)
1. Install Python 3.10+ (64-bit).
2. Create/activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python micro_event_gui.py
   ```

## Notes & Tips
- Default camera index is **0**; change `DEFAULT_CAMERA_INDEX` in the script if needed.
- For more sensitivity, use a **higher-FPS** camera and ensure good, even lighting.
- The “baseline” helps reduce false positives by accounting for a person’s natural micromovements.
- You can tweak `SHORT_MS`, `HIST_LEN`, and thresholds in-app or in code.

## Ethics & Compliance
- Always obtain **explicit consent**.
- Log purpose, retention, and access if used in professional settings.
- Do **not** automate decisions that affect rights or access based solely on this tool.

## New in this version
- **Emotion detection (optional)** via [FER](https://github.com/justinshenk/fer). If FER (and its backend) isn't installed, the app runs without emotion labels.
- **Eye-gaze direction** using MediaPipe iris + eye landmarks (labels: Left/Right/Up/Down/Center and diagonals).
- **CSV export** now includes `emotion_top`, individual `emo_*` scores, and `gaze_horiz`, `gaze_vert`, `gaze_label`.

### Extra install (for emotions)
`FER` may require additional dependencies (e.g., TensorFlow). If install is heavy or fails, you can skip it—the app will still work (emotion disabled). Try:
```bash
pip install fer
# If needed on Windows:
pip install tensorflow==2.15.0
```
