import cv2
import face_recognition
import pickle
import os
import csv
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import sys

def initialize_gui():
    """Initialize the GUI window and components"""
    root = tk.Tk()
    root.title("Face Recognition Log")
    
    # Create a Treeview to display logs
    tree = ttk.Treeview(root, columns=("Name", "Date", "Time"), show="headings")
    tree.heading("Name", text="Name")
    tree.heading("Date", text="Date")
    tree.heading("Time", text="Time")
    tree.pack(fill="both", expand=True)
    
    return root, tree

def load_encodings():
    """Load the face encodings from the pickle file"""
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
        if "encodings" not in data or "names" not in data:
            raise ValueError("Missing required keys in encodings file")
        if len(data["encodings"]) == 0:
            raise ValueError("No encodings found")
        return data
    except FileNotFoundError:
        raise FileNotFoundError("encodings.pickle not found. Run encode_Face_script.py first.")
    except Exception as e:
        raise Exception(f"Error loading encodings: {str(e)}")

def initialize_camera():
    """Initialize the camera"""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Unable to open camera")
    return video_capture

def log_recognition(name, tree, log_file):
    """Log a recognized face"""
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Write to log file
    with open(log_file, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([name, date, time])

    # Update GUI log
    tree.insert("", "end", values=(name, date, time))
    print(f"✅ Recognized {name} at {date} {time}")

def main():
    # Ensure timestamp directory exists
    timestamp_dir = os.path.join(os.getcwd(), "timestamp")
    os.makedirs(timestamp_dir, exist_ok=True)
    log_file = os.path.join(timestamp_dir, "timestamplog.csv")

    try:
        # Initialize GUI
        root, tree = initialize_gui()

        # Load encodings
        data = load_encodings()
        
        # Initialize camera
        video_capture = initialize_camera()
        
        print("✅ Press 'q' to exit")

        # Process every nth frame for efficiency
        frame_process_interval = 10
        frame_count = 0
        recognized_faces = set()

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Error: Failed to capture video frame")
                break

            # Convert the image from BGR color to RGB color
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Only process every nth frame
            if frame_count % frame_process_interval == 0:
                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")

                if face_locations:
                    try:
                        # Generate encodings for detected faces
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        # Process each detected face
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.6)
                            name = "Unknown"
                            rectangle_color = (0, 0, 255)  # Red for unknown

                            if True in matches:
                                matched_indexes = [i for (i, match) in enumerate(matches) if match]
                                name_counts = {}
                                for index in matched_indexes:
                                    matched_name = data["names"][index]
                                    name_counts[matched_name] = name_counts.get(matched_name, 0) + 1
                                
                                name = max(name_counts, key=name_counts.get)
                                rectangle_color = (0, 255, 0)  # Green for recognized

                                # Log only if not already logged this session
                                if name not in recognized_faces:
                                    recognized_faces.add(name)
                                    log_recognition(name, tree, log_file)

                            # Draw rectangle and name
                            cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), rectangle_color, cv2.FILLED)
                            cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    except Exception as e:
                        print(f"❌ Error processing faces: {e}")
                        continue

            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Update GUI
            try:
                root.update()
            except tk.TclError:
                break  # Handle case where GUI window is closed

            frame_count += 1

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("✅ Exiting program")
                break

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"❌ Error: {str(e)}")
        return 1

    finally:
        # Clean up
        try:
            video_capture.release()
            cv2.destroyAllWindows()
            root.quit()
        except:
            pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
