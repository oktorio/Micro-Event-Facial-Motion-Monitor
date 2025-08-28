# Micro-Event Facial Motion Monitor (Windows GUI)

A Windows-friendly **Tkinter GUI** that uses your webcam to estimate **brief facial motion bursts** (brows, eyes/eyelids, mouth), **coarse emotions** (via FER + TensorFlow), and **gaze direction** (Left/Right/Up/Down/Center using MediaPipe iris). Includes **CSV export**. A no-flicker UI loop keeps all Tk updates on the main thread.

> ⚠️ **Important**: This is **NOT** a lie detector. Do **not** use it to decide truthfulness or eligibility. Use only with informed consent.

---

## ✨ Features

- Real-time webcam preview with face-mesh overlay (MediaPipe FaceMesh)
- **Motion bursts**: compares a short recent window (~200 ms) vs prior window for brows/eyes/mouth
- **Emotion** (optional): anger, disgust, fear, happy, sad, surprise, neutral (FER + TensorFlow)
- **Gaze** estimation with MediaPipe iris (Left/Right/Up/Down/Center + diagonals)
- **No-flicker UI**: thread-safe queue + `after()` redraw
- **Baseline** (10 s) to normalize per-person micromovements
- **CSV export** with timestamp, motions, baseline-adjusted metrics, emotion & gaze columns
- **Low-GPU mode** + **preview size selector** to reduce CPU/GPU load

---

## 🚀 Quickstart (Windows, Python 3.10)

```bat
:: 1) Create & activate isolated environment
python -m venv .venv
.venv\Scripts\activate

:: 2) Tooling
python -m pip install --upgrade pip setuptools wheel

:: 3) Install deps (pinned for stability)
pip install -r requirements.txt

:: 4) Run the app
python micro_event_gui_emotion_gaze_no_flicker.py
```

### Requirements (pinned)
See `requirements.txt`. Highlights:
- `tensorflow==2.15.0` (CPU build) for FER
- `moviepy==1.0.3` (compat layer for `moviepy.editor`)
- `mediapipe`, `opencv-python`, `numpy`, `pandas`, `pillow`

> If you already use PyTorch elsewhere, keep it in a **separate venv** to avoid dependency conflicts.

---

## 🖥️ Usage

1. **Start Camera** → look at the camera with good, even lighting.
2. Optionally click **Record 10s Baseline** (neutral face) so burst detection adapts to the person.
3. Tune **Brow/Eye/Mouth thresholds** if you see too many / too few flags.
4. Use **Low GPU mode** and/or reduce **Preview size** if UI feels heavy.
5. **Export CSV** to save per-frame metrics for analysis.

**CSV columns (main):**
- `timestamp`
- `brow_motion`, `eye_motion`, `mouth_motion`
- `brow_minus_baseline`, `eye_minus_baseline`, `mouth_minus_baseline` (if baseline set)
- `emotion_top`, plus individual `emo_anger`, `emo_disgust`, `emo_fear`, `emo_happy`, `emo_sad`, `emo_surprise`, `emo_neutral` (0–1)
- `gaze_horiz`, `gaze_vert` (−1..1), `gaze_label` (e.g., `Left`, `Right-Up`)

---

## 🛠️ Troubleshooting

**Emotion shows “(disabled)”**
- Ensure FER + TF are installed in the *same* venv:
  ```bat
  python -c "import tensorflow as tf; print(tf.__version__)"
  python -c "from fer import FER; FER(mtcnn=False); print('FER OK')"
  ```
- If FER import fails on `moviepy.editor`, install **MoviePy 1.0.3** (newer 2.x removed `editor`):
  ```bat
  pip install --force-reinstall "moviepy==1.0.3"
  ```

**Blinking window / flicker**
- This build uses a **no-flicker** UI. If you still see brightness pulsing, that’s usually **camera auto-exposure under 50/60 Hz lights**. Improve lighting or lock exposure in your camera driver.

**Camera in use / black screen**
- Close other apps (Zoom/Teams/OBS) that might hold the camera; try device index 1 instead of 0.

**TensorFlow DLL errors**
- Use CPU TF 2.15.0 on Python 3.10. Upgrade basics:
  ```bat
  pip install --upgrade numpy protobuf h5py
  ```

**Environment checker**
- Run the included `tools/env_check_emotion.py` to print versions and test one-frame inference.

---

## ⚖️ Ethics & Compliance

- Obtain **explicit consent** before capturing/processing video.
- Document purpose, retention, and access controls.
- Do **not** automate decisions that affect rights or access based solely on these signals.
- Micro-expression science is contested; treat outputs as **auxiliary context** only.

---

## 📌 Roadmap

- [ ] Optional MP4 recording synced with CSV
- [ ] Small charts panel (live emotion/gaze plots)
- [ ] Per-session JSON config export/import
- [ ] Exposure/white-balance lock toggles

---

## 📜 License

This project is released under the **MIT License** (see `LICENSE`).

---

## 🙏 Acknowledgements

- MediaPipe FaceMesh/Iris by Google  
- FER (Facial Emotion Recognition) by Justin Shenk et al.  
- OpenCV, TensorFlow, MoviePy
