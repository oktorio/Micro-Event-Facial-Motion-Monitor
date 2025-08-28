#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Facial Motion + Emotion + Gaze (Demo) — No-Flicker Tkinter Update
- All Tkinter widget updates happen on the main thread via .after()
- Worker thread only does capture + compute, pushes results through a Queue
- Optional FPS throttle & smaller preview reduce strain
"""
import cv2, numpy as np, mediapipe as mp, pandas as pd, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading, time, datetime, queue

# Optional emotion detector
FER_AVAILABLE = True
try:
    from fer import FER
except Exception:
    FER_AVAILABLE = False

DEFAULT_CAMERA_INDEX = 0
TARGET_FPS = 30.0
SHORT_MS = 200
HIST_LEN = 600
FONT = ("Segoe UI", 10)

mp_face = mp.solutions.face_mesh

# Landmark groups
BROWS   = list(range(70,90)) + list(range(300,320))
EYES    = list(range(33,133)) + list(range(362,463))
MOUTH   = list(range(78,308))

RIGHT_EYE_OUTER = 33; RIGHT_EYE_INNER = 133; RIGHT_EYE_TOP = 159; RIGHT_EYE_BOTTOM = 145; RIGHT_IRIS_IDX=[474,475,476,477]
LEFT_EYE_OUTER  = 263; LEFT_EYE_INNER  = 362; LEFT_EYE_TOP  = 386; LEFT_EYE_BOTTOM  = 374; LEFT_IRIS_IDX =[469,470,471,472]

def mean_motion(flow, idx):
    if not idx: return 0.0
    v = flow[idx]
    return float(np.sqrt((v**2).sum(axis=1)).mean())

def clamp01(x): return max(0.0, min(1.0, float(x)))

def gaze_from_landmarks(pts):
    try:
        rx0, rx1 = pts[RIGHT_EYE_OUTER,0], pts[RIGHT_EYE_INNER,0]
        ry0, ry1 = pts[RIGHT_EYE_TOP,1],   pts[RIGHT_EYE_BOTTOM,1]
        rleft, rright = min(rx0, rx1), max(rx0, rx1)
        rtop, rbot    = min(ry0, ry1), max(ry0, ry1)
        r_w, r_h = max(1.0, rright-rleft), max(1.0, rbot-rtop)
        r_iris = pts[RIGHT_IRIS_IDX].mean(axis=0)
        r_nx = clamp01((r_iris[0]-rleft)/r_w); r_ny = clamp01((r_iris[1]-rtop)/r_h)

        lx0, lx1 = pts[LEFT_EYE_OUTER,0], pts[LEFT_EYE_INNER,0]
        ly0, ly1 = pts[LEFT_EYE_TOP,1],   pts[LEFT_EYE_BOTTOM,1]
        lleft, lright = min(lx0, lx1), max(lx0, lx1)
        ltop, lbot    = min(ly0, ly1), max(ly0, ly1)
        l_w, l_h = max(1.0, lright-lleft), max(1.0, lbot-ltop)
        l_iris = pts[LEFT_IRIS_IDX].mean(axis=0)
        l_nx = clamp01((l_iris[0]-lleft)/l_w); l_ny = clamp01((l_iris[1]-ltop)/l_h)

        nx = float((r_nx + l_nx)/2.0); ny = float((r_ny + l_ny)/2.0)
        horiz = (nx-0.5)*2.0; vert = (ny-0.5)*2.0
        h_lab = "Left" if horiz<-0.2 else ("Right" if horiz>0.2 else "Center")
        v_lab = "Up"   if vert<-0.2  else ("Down"  if vert>0.2  else "Center")
        label = "Center" if (h_lab=="Center" and v_lab=="Center") else (h_lab if v_lab=="Center" else (v_lab if h_lab=="Center" else f"{h_lab}-{v_lab}"))
        return horiz, vert, label
    except Exception:
        return 0.0, 0.0, "Unknown"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Motion + Emotion + Gaze (No-Flicker) — NOT a lie detector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.queue = queue.Queue(maxsize=2)  # frames/metrics pipeline
        self.running = False
        self.worker = None
        self.cap = None
        self.short_n = max(3, int(TARGET_FPS*(SHORT_MS/1000.0)))
        self.baseline = None

        self.flow_hist=[]; self.timestamps=[]; self.emotions=[]; self.emotion_labels=[]; self.gaze_vals=[]
        self.th_brow=tk.DoubleVar(value=0.35); self.th_eye=tk.DoubleVar(value=0.30); self.th_mouth=tk.DoubleVar(value=0.40)
        self.low_gpu=tk.BooleanVar(value=False)
        self.res_choice=tk.StringVar(value="800x450")

        # Optional emotion
        self.fer = None
        if FER_AVAILABLE:
            try: self.fer = FER(mtcnn=False)
            except Exception: self.fer=None

        self.build_ui()
        self.schedule_ui_update()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10); top.pack(fill="both", expand=True)
        self.video_label = ttk.Label(top); self.video_label.grid(row=0, column=0, rowspan=12, sticky="nsew", padx=(0,10))

        ctrl = ttk.Frame(top); ctrl.grid(row=0, column=1, sticky="nsew")
        btns = ttk.Frame(ctrl); btns.pack(fill="x", pady=(0,6))
        self.start_btn=ttk.Button(btns,text="Start Camera",command=self.start)
        self.stop_btn =ttk.Button(btns,text="Stop",command=self.stop,state="disabled")
        self.base_btn =ttk.Button(btns,text="Record 10s Baseline",command=self.record_baseline,state="disabled")
        self.exp_btn  =ttk.Button(btns,text="Export CSV",command=self.export_csv,state="disabled")
        self.start_btn.pack(side="left",padx=(0,6)); self.stop_btn.pack(side="left",padx=(0,6))
        self.base_btn.pack(side="left",padx=(0,6)); self.exp_btn.pack(side="left")

        opts = ttk.LabelFrame(ctrl,text="Performance",padding=6); opts.pack(fill="x",pady=(6,6))
        ttk.Checkbutton(opts,text="Low GPU mode (throttle FPS)",variable=self.low_gpu).pack(anchor="w")
        res_row=ttk.Frame(opts); res_row.pack(fill="x", pady=2)
        ttk.Label(res_row,text="Preview size:",width=12).pack(side="left")
        ttk.Combobox(res_row,textvariable=self.res_choice,values=["640x360","800x450","960x540","1280x720"],state="readonly",width=10).pack(side="left")

        thr = ttk.LabelFrame(ctrl, text="Burst thresholds (Δ recent vs prior)", padding=6); thr.pack(fill="x", pady=(6,6))
        self.add_slider(thr,"Brow", self.th_brow,0.0,2.0); self.add_slider(thr,"Eye", self.th_eye,0.0,2.0); self.add_slider(thr,"Mouth",self.th_mouth,0.0,2.0)

        self.status = ttk.LabelFrame(ctrl, text="Live metrics", padding=6); self.status.pack(fill="x", pady=(6,6))
        self.lbl_fps=ttk.Label(self.status,text="FPS: —",font=FONT)
        self.lbl_brow=ttk.Label(self.status,text="Brow motion: —",font=FONT)
        self.lbl_eye =ttk.Label(self.status,text="Eye/Eyelid motion: —",font=FONT)
        self.lbl_mouth=ttk.Label(self.status,text="Mouth motion: —",font=FONT)
        self.lbl_flags=ttk.Label(self.status,text="Flags: —",font=FONT)
        self.lbl_base=ttk.Label(self.status,text="Baseline: not set",font=FONT)
        self.lbl_emot=ttk.Label(self.status,text=f"Emotion: {'(disabled)' if self.fer is None else '—'}",font=FONT)
        self.lbl_gaze=ttk.Label(self.status,text="Gaze: —",font=FONT)
        for w in (self.lbl_fps,self.lbl_brow,self.lbl_eye,self.lbl_mouth,self.lbl_flags,self.lbl_base,self.lbl_emot,self.lbl_gaze):
            w.pack(anchor="w")

        ttk.Label(self.root,text=("This demo estimates brief motion bursts, coarse emotions (optional), and gaze direction.\n"
                                  "All UI updates run on the main thread to avoid flicker. Do NOT use for lie detection."),
                  wraplength=820,font=("Segoe UI",9)).pack(padx=10,pady=(0,8))

        top.columnconfigure(0,weight=1); top.columnconfigure(1,weight=0); top.rowconfigure(0,weight=1)

    def add_slider(self,parent,name,var,frm,to):
        row=ttk.Frame(parent); row.pack(fill="x",pady=2)
        ttk.Label(row,text=f"{name}:",width=12).pack(side="left")
        val=tk.StringVar(value=f"{var.get():.2f}")
        s=ttk.Scale(row,variable=var,from_=frm,to=to,command=lambda v: val.set(f"{float(v):.2f}"))
        s.pack(side="left",fill="x",expand=True,padx=6)
        ttk.Label(row,textvariable=val).pack(side="left")

    def start(self):
        if self.running: return
        self.cap=cv2.VideoCapture(DEFAULT_CAMERA_INDEX)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera Error","Cannot open camera.")
            return
        # Try to reduce flicker from mains lighting by locking FPS (may be ignored by some cams)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running=True
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.base_btn.config(state="normal"); self.exp_btn.config(state="normal")
        self.worker=threading.Thread(target=self.worker_loop,daemon=True); self.worker.start()

    def stop(self):
        self.running=False
        self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled"); self.base_btn.config(state="disabled")

    def on_close(self):
        self.running=False
        if self.cap:
            try: self.cap.release()
            except Exception: pass
        self.root.after(200,self.root.destroy)

    def record_baseline(self):
        if not self.running:
            messagebox.showinfo("Info","Start the camera first."); return
        if not self.flow_hist:
            messagebox.showinfo("Info","Wait a moment for data."); return
        # Collect last ~10s from history
        n=int(30*10)
        arr=np.array(self.flow_hist[-n:]) if len(self.flow_hist)>=n else np.array(self.flow_hist)
        if arr.size==0:
            messagebox.showwarning("Warning","Not enough data for baseline."); return
        self.baseline={"brow":float(arr[:,0].mean()),"eye":float(arr[:,1].mean()),"mouth":float(arr[:,2].mean())}
        self.lbl_base.config(text=f"Baseline set (brow={self.baseline['brow']:.3f}, eye={self.baseline['eye']:.3f}, mouth={self.baseline['mouth']:.3f})")

    def export_csv(self):
        if not self.timestamps or not self.flow_hist:
            messagebox.showinfo("Info","No data to export yet."); return
        from tkinter import filedialog
        fpath=filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")], initialfile="session_no_flicker.csv")
        if not fpath: return
        df=pd.DataFrame(self.flow_hist,columns=["brow_motion","eye_motion","mouth_motion"])
        df.insert(0,"timestamp",self.timestamps)
        if self.baseline:
            df["brow_minus_baseline"]=df["brow_motion"]-self.baseline["brow"]
            df["eye_minus_baseline"]=df["eye_motion"]-self.baseline["eye"]
            df["mouth_minus_baseline"]=df["mouth_motion"]-self.baseline["mouth"]
        # emotions
        if self.emotion_labels: df["emotion_top"]=self.emotion_labels
        if self.emotions:
            keys=set(); [keys.update(d.keys()) for d in self.emotions if isinstance(d,dict)]
            for k in sorted(keys):
                df[f"emo_{k}"]=[(d.get(k) if isinstance(d,dict) else None) for d in self.emotions]
        # gaze
        if self.gaze_vals:
            gh=[]; gv=[]; gl=[]
            for g in self.gaze_vals:
                if isinstance(g,(list,tuple)) and len(g)==3: gh.append(g[0]); gv.append(g[1]); gl.append(g[2])
                else: gh.append(None); gv.append(None); gl.append(None)
            df["gaze_horiz"]=gh; df["gaze_vert"]=gv; df["gaze_label"]=gl
        df.to_csv(fpath,index=False); messagebox.showinfo("Saved", f"Exported {len(df)} rows to:\n{fpath}")

    # ---------- Worker thread (no Tk calls here!) ----------
    def worker_loop(self):
        prev_pts=None
        last_time=time.time()
        fps_acc=TARGET_FPS
        with mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
            while self.running:
                ok, frame = self.cap.read()
                if not ok: time.sleep(0.01); continue

                # Optional throttle
                if self.low_gpu.get():
                    time.sleep(0.01)

                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=mesh.process(rgb)

                brow_m=eye_m=mouth_m=0.0; flags=[]; emo_label=None; emo_scores={}; gaze_triplet=None

                if res.multi_face_landmarks:
                    h,w=frame.shape[:2]
                    lms=res.multi_face_landmarks[0].landmark
                    pts=np.array([[lm.x*w, lm.y*h] for lm in lms], dtype=np.float32)

                    # motion
                    if prev_pts is not None and len(prev_pts)==len(pts):
                        flow=pts-prev_pts
                        brow_m=mean_motion(flow,BROWS); eye_m=mean_motion(flow,EYES); mouth_m=mean_motion(flow,MOUTH)
                        self.flow_hist.append([brow_m,eye_m,mouth_m])
                        self.timestamps.append(datetime.datetime.now().isoformat())
                        if len(self.flow_hist)>HIST_LEN:
                            self.flow_hist=self.flow_hist[-HIST_LEN:]; self.timestamps=self.timestamps[-HIST_LEN:]
                            if self.emotions: self.emotions=self.emotions[-HIST_LEN:]
                            if self.emotion_labels: self.emotion_labels=self.emotion_labels[-HIST_LEN:]
                            if self.gaze_vals: self.gaze_vals=self.gaze_vals[-HIST_LEN:]

                        if len(self.flow_hist)>=self.short_n*2:
                            recent=np.array(self.flow_hist[-self.short_n:])
                            prior =np.array(self.flow_hist[-self.short_n*2:-self.short_n])
                            delta=(recent.mean(axis=0)-prior.mean(axis=0))
                            if self.baseline:
                                delta[0]-=0.2*self.baseline["brow"]; delta[1]-=0.2*self.baseline["eye"]; delta[2]-=0.2*self.baseline["mouth"]
                            if delta[0]>float(self.th_brow.get()): flags.append("brow burst")
                            if delta[1]>float(self.th_eye.get()):  flags.append("eye burst")
                            if delta[2]>float(self.th_mouth.get()):flags.append("mouth burst")

                    prev_pts=pts

                    # gaze
                    gaze_triplet=gaze_from_landmarks(pts)

                    # draw overlay (do drawing in worker to reduce main-thread work)
                    for idx in [BROWS,EYES,MOUTH]:
                        for i in idx:
                            x,y=int(pts[i,0]),int(pts[i,1])
                            cv2.circle(frame,(x,y),1,(0,255,0),-1)

                    if flags:
                        cv2.putText(frame,"Brief motion: "+", ".join(flags),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                    # emotion (optional)
                    if self.fer is not None:
                        x0=int(np.clip(np.min(pts[:,0])-20,0,w-1)); y0=int(np.clip(np.min(pts[:,1])-20,0,h-1))
                        x1=int(np.clip(np.max(pts[:,0])+20,0,w-1)); y1=int(np.clip(np.max(pts[:,1])+20,0,h-1))
                        face_crop=rgb[y0:y1, x0:x1]
                        if face_crop.size>0 and face_crop.shape[0]>10 and face_crop.shape[1]>10:
                            try:
                                det=self.fer.detect_emotions(face_crop)
                                if det:
                                    em=det[0].get('emotions',{})
                                    if em:
                                        emo_scores={k:float(v) for k,v in em.items()}
                                        emo_label=max(emo_scores,key=emo_scores.get)
                                        cv2.putText(frame,f"Emotion: {emo_label}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                            except Exception:
                                pass

                # book-keeping arrays (must stay aligned with timestamps)
                if self.timestamps and (len(self.timestamps)>len(self.emotion_labels)):
                    self.emotion_labels.append(emo_label)
                    self.emotions.append(emo_scores if isinstance(emo_scores,dict) else {})
                if self.timestamps and (len(self.timestamps)>len(self.gaze_vals)):
                    self.gaze_vals.append(gaze_triplet if gaze_triplet else (None,None,None))

                # FPS
                now=time.time(); fps=1.0/max(1e-6,(now-last_time)); last_time=now
                cv2.putText(frame,f"FPS: {fps:.1f}",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

                # push to queue (drop oldest if queue is full)
                try:
                    if not self.queue.empty():
                        self.queue.get_nowait()
                    self.queue.put_nowait((frame, fps, brow_m, eye_m, mouth_m, flags, emo_label, gaze_triplet))
                except queue.Full:
                    pass

    # ---------- Main-thread UI updater ----------
    def schedule_ui_update(self):
        try:
            frame, fps, brow_m, eye_m, mouth_m, flags, emo_label, gaze_triplet = self.queue.get_nowait()
            # Resize to selected preview
            w,h = map(int, self.res_choice.get().split("x"))
            disp=cv2.resize(frame,(w,h))
            disp=cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
            im=Image.fromarray(disp); imgtk=ImageTk.PhotoImage(image=im)
            self.video_label.imgtk=imgtk; self.video_label.configure(image=imgtk)

            # Update labels
            self.lbl_fps.config(text=f"FPS: {fps:.1f}")
            self.lbl_brow.config(text=f"Brow motion: {brow_m:.3f}")
            self.lbl_eye.config(text=f"Eye/Eyelid motion: {eye_m:.3f}")
            self.lbl_mouth.config(text=f"Mouth motion: {mouth_m:.3f}")
            self.lbl_flags.config(text=f"Flags: {', '.join(flags) if flags else '—'}")
            self.lbl_emot.config(text=f"Emotion: {(emo_label or ('(disabled)' if self.fer is None else '—'))}")
            if gaze_triplet:
                self.lbl_gaze.config(text=f"Gaze: {gaze_triplet[2]}")
        except queue.Empty:
            pass
        # Schedule next UI update
        self.root.after(10, self.schedule_ui_update)

def main():
    root=tk.Tk()
    style=ttk.Style()
    try: style.theme_use('vista')
    except Exception: pass
    app=App(root)
    root.mainloop()

if __name__=="__main__":
    main()
