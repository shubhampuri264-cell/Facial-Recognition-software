# Facial Recognition Software

## Setup- First Time Only

If you haven't set up the environment yet:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
You can run the software directly from the terminal.

### Quick Start
Run the launch script:
- **Command Prompt**: `run.bat`
Or via Python:
> `python real_time_face_recognition.py`

### Controls
- **Auto-Scan (Enable)**: When checked, the camera continuously searches for faces.
- **Scan Now**: Click this to force an immediate check.
- **Quit**: Closes the application.

---

## How to Add New Faces

To teach the software new people, use the built-in capture tool.

### Capture Faces
Run the capture script:
> `python capture.py`

- Enter the person's name when prompted.
- The camera window will open.
- Press **'c'** to capture a photo (take at least 5 photos from different angles).
- Press **'q'** when finished.

### Encode Faces
After capturing photos, you must update the database:
> `python encode_Face_script.py`

### Restart the App
Restart the main application (`run.bat`) to load the new faces.

