import tkinter as tk
from tkinter import Label
import subprocess
import cv2
from PIL import Image, ImageTk

def start_camera():
    global cap
    global camera_label
    # Check if the camera is already opened
    if 'cap' in globals() and cap.isOpened():
        # If the camera is already opened, release it
        cap.release()
        camera_label.destroy()
        cv2.destroyAllWindows()
    # Open the camera
    cap = cv2.VideoCapture(0)

    def update_frame():
        # Read a frame from the camera
        ret, frame = cap.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            # Resize the frame to fit the label
            frame = cv2.resize(frame, (900, 600))
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL Image
            img = Image.fromarray(frame)
            # Convert the PIL Image to an ImageTk object
            imgtk = ImageTk.PhotoImage(image=img)
            # Update the label with the new frame
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
        # Call this function again after 1ms
        camera_label.after(1, update_frame)

    # Create a label to display the camera feed
    camera_label = Label(root)
    camera_label.pack()
    update_frame()

def stop_camera():
    global cap
    # Close the camera feed
    if 'cap' in globals() and cap.isOpened():
        cap.release()
        # Destroy the label displaying the camera feed
        camera_label.destroy()
        # Close all OpenCV windows
    cv2.destroyAllWindows()

def run_real_time_face_recognition():
    # Run the real_time_face_recognition.py script
    subprocess.run(["python", "real_time_face_recognition.py"])

def run_check_camera():
    # Run the check_camera.py script
    subprocess.run(["python", "check_Camera.py"])

def on_hover(event, button):
    # Change the button's background color on hover
    button.config(bg="#d1d1d1")

def on_leave(event, button):
    # Revert the button's background color when the mouse leaves
    button.config(bg="SystemButtonFace")

# Create the main application window
root = tk.Tk()
root.title("Facial Recognition Attendance")
root.geometry("1200x900")  # Set the window size (width x height)
root.minsize(1200, 900)  # Set the background color

# Create a frame to hold the buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)  # Add some padding around the frame

# Add a button to start the camera
start_button = tk.Button(button_frame, text="Start Camera", command=start_camera)
start_button.pack(side=tk.LEFT, padx=5)  # Align horizontally with padding
start_button.config(font=("Comic Sans MS", 12), width=20, height=2)

# Add a button to stop the camera
stop_button = tk.Button(button_frame, text="Stop Camera", command=stop_camera)
stop_button.pack(side=tk.LEFT, padx=5)  # Align horizontally with padding
stop_button.config(font=("Comic Sans MS", 12), width=20, height=2)

# Add a button to run the real_time_face_recognition script
run_button = tk.Button(button_frame, text="Run Face Recognition", command=run_real_time_face_recognition)
run_button.pack(side=tk.LEFT, padx=5)  # Align horizontally with padding
run_button.config(font=("Comic Sans MS", 12), width=20, height=2)

# Add a button to run the check_camera script
check_camera_button = tk.Button(button_frame, text="Check Camera", command=run_check_camera)
check_camera_button.pack(side=tk.LEFT, padx=5)  # Align horizontally with padding
check_camera_button.config(font=("Comic Sans MS", 12), width=20, height=2)

# Bind hover events to the run_button
run_button.bind("<Enter>", lambda event: on_hover(event, run_button))
run_button.bind("<Leave>", lambda event: on_leave(event, run_button))

# Bind hover events to the check_camera_button
check_camera_button.bind("<Enter>", lambda event: on_hover(event, check_camera_button))
check_camera_button.bind("<Leave>", lambda event: on_leave(event, check_camera_button))

# Add a background color
root.configure(bg="#f0f0f0")

# Run the application
root.mainloop()