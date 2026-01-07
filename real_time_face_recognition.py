import cv2
import face_recognition
import json
import numpy as np
import os
import csv
import datetime
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

try:
    import cv2
except ImportError:
    print("❌ Error: 'opencv-python' module is not installed.")
    sys.exit(1)

try:
    import face_recognition
except ImportError:
    print("❌ Error: 'face_recognition' module is not installed.")
    sys.exit(1)

class CameraThread(threading.Thread):
    def __init__(self, src=0, name="CameraThread"):
        super().__init__(name=name)
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def run(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            
            # Read the next frame
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.join()
        if self.cap.isOpened():
            self.cap.release()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("1100x700")
        
        # --- State Variables ---
        self.running = True
        self.camera_thread = None
        self.current_frame = None
        self.latest_scan_result = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognized_faces_session = set()
        
        # FPS Calculation
        self.fps_start_time = time.time()
        self.fps_frame_counter = 0
        self.fps = 0
        
        # Load configuration
        self.load_encodings()
        self.setup_logging()
        
        # --- GUI Layout ---
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel: Video Feed
        self.video_panel = ttk.LabelFrame(main_frame, text="Live Camera Feed")
        self.video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_label = ttk.Label(self.video_panel, text="Starting Camera...")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right Panel: Controls and Logs
        right_panel = ttk.Frame(main_frame, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Control Box
        control_group = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        control_group.pack(fill=tk.X, pady=5)
        
        # Auto-Scan Toggle
        self.auto_scan_var = tk.BooleanVar(value=True)
        self.chk_auto = ttk.Checkbutton(control_group, text="Auto-Scan (Enable)", variable=self.auto_scan_var)
        self.chk_auto.pack(fill=tk.X, pady=5)
        
        # Manual Scan Button
        self.btn_scan = ttk.Button(control_group, text="📸 Scan Now", command=self.manual_scan)
        self.btn_scan.pack(fill=tk.X, pady=5)
        
        # Quit Button
        self.btn_quit = ttk.Button(control_group, text="Quit Application", command=self.on_closing)
        self.btn_quit.pack(fill=tk.X, pady=20)
        
        # Log Box
        log_group = ttk.LabelFrame(right_panel, text="Recognition Log", padding="5")
        log_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.tree = ttk.Treeview(log_group, columns=("Name", "Time"), show="headings", height=20)
        self.tree.heading("Name", text="Name")
        self.tree.column("Name", width=120)
        self.tree.heading("Time", text="Time")
        self.tree.column("Time", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initialization ---
        self.start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_encodings(self):
        try:
            with open("encodings.json", "r") as f:
                data = json.load(f)
            
            if "encodings" in data:
                self.known_face_encodings = [np.array(e) for e in data["encodings"]]
                self.known_face_names = data["names"]
                print(f"✅ Loaded {len(self.known_face_names)} identities.")
            else:
                 messagebox.showwarning("Warning", "No 'encodings' found in JSON.")
                 self.known_face_names = []
                 self.known_face_encodings = []

        except FileNotFoundError:
            messagebox.showerror("Error", "encodings.json not found!\nPlease run encode_Face_script.py first.")
            self.known_face_names = []
            self.known_face_encodings = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load encodings: {e}")

    def setup_logging(self):
        # Ensure timestamp directory exists
        timestamp_dir = os.path.join(os.getcwd(), "timestamp")
        os.makedirs(timestamp_dir, exist_ok=True)
        self.log_file = os.path.join(timestamp_dir, "timestamplog.csv")

    def log_recognition(self, name):
        """Log a recognized face if not recently logged"""
        if name in self.recognized_faces_session or name == "Unknown":
            return

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Write to file
        try:
            with open(self.log_file, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([name, date_str, time_str])
        except Exception as e:
            print(f"Logging error: {e}")

        # Update GUI
        self.tree.insert("", 0, values=(name, time_str)) # Insert at top
        self.recognized_faces_session.add(name)
        self.status_var.set(f"Recognized: {name}")

    def start_camera(self):
        try:
            self.camera_thread = CameraThread(src=0)
            self.camera_thread.start()
            
            # Start loop
            self.update_video_feed()
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))

    def manual_scan(self):
        """Force a scan regardless of auto settings"""
        if self.current_frame is not None:
             self.status_var.set("Scanning...")
             # Run in a separate thread to not freeze UI
             threading.Thread(target=self.process_face_recognition, args=(self.current_frame.copy(),), daemon=True).start()

    def process_face_recognition(self, frame_bgr):
        """Run face recognition logic (CPU intensive)"""
        try:
            # Resize for speed
            small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            detected_names = []
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                
                if True in matches:
                    matched_indexes = [i for (i, match) in enumerate(matches) if match]
                    counts = {}
                    for i in matched_indexes:
                        nm = self.known_face_names[i]
                        counts[nm] = counts.get(nm, 0) + 1
                    name = max(counts, key=counts.get)
                
                detected_names.append(name)
                self.log_recognition(name)
            
            # Store result to be drawn on the main UI thread
            # Rescale locations back to original size
            scaled_locations = []
            for (top, right, bottom, left) in face_locations:
                scaled_locations.append((top*4, right*4, bottom*4, left*4))
            
            self.latest_scan_result = (scaled_locations, detected_names)
            
            if not detected_names:
                self.status_var.set("Scan complete. No known faces found.")
            
        except Exception as e:
            print(f"Recognition Error: {e}")

    def update_video_feed(self):
        if not self.running:
            return
        
        if self.camera_thread:
            grabbed, frame = self.camera_thread.read()
            
            if grabbed:
                self.current_frame = frame
                
                # FPS Calculation
                self.fps_frame_counter += 1
                if time.time() - self.fps_start_time > 1:
                    self.fps = self.fps_frame_counter / (time.time() - self.fps_start_time)
                    self.fps_frame_counter = 0
                    self.fps_start_time = time.time()
                
                # Auto-scan logic
                if not hasattr(self, 'frame_count'): self.frame_count = 0
                self.frame_count += 1
                
                if self.auto_scan_var.get() and self.frame_count % 30 == 0: 
                     threading.Thread(target=self.process_face_recognition, args=(frame.copy(),), daemon=True).start()

                # Overlay
                display_frame = frame.copy()
                
                if self.latest_scan_result:
                    locations, names = self.latest_scan_result
                    for (top, right, bottom, left), name in zip(locations, names):
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(display_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                # Draw FPS
                cv2.putText(display_frame, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Tkinter Update
                cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk 
                self.video_label.configure(image=imgtk)
        
        # 30 FPS target (approx 33ms)
        self.root.after(30, self.update_video_feed)

    def on_closing(self):
        self.running = False
        if self.camera_thread:
            self.camera_thread.stop()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
