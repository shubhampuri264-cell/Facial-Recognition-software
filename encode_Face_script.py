import sys
import os
import json
import argparse
import numpy as np
try:
    import face_recognition
except ImportError:
    print("❌ Error: 'face_recognition' module is not installed.")
    print("   Please run: pip install face_recognition")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("❌ Error: 'opencv-python' module is not installed.")
    print("   Please run: pip install opencv-python")
    sys.exit(1)

def preprocess_image(image_path):
    """
    Load an image and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve face detection in poor lighting.
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge matching L-channel with a and b channels
        limg = cv2.merge((cl, a, b))

        # Convert back to RGB (face_recognition uses RGB)
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final_img
    except Exception as e:
        print(f"   ⚠️  Warning during preprocessing {image_path}: {e}")
        # Fallback to standard load if advanced preprocessing fails
        return face_recognition.load_image_file(image_path)

def encode_faces(dataset_dir="dataset", output_file="encodings.json", detection_method="hog", jitters=1):
    """
    Encode faces from images in the dataset directory and save them to a JSON file.
    """
    print(f"🚀 Starting face encoding...")
    print(f"   📂 Dataset: {dataset_dir}")
    print(f"   💾 Output: {output_file}")
    print(f"   🔍 Method: {detection_method}")
    print(f"   🔄 Jitters: {jitters}\n")

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        return False

    known_encodings = []
    known_names = []

    stats = {
        "processed": 0,
        "success": 0,
        "skipped_no_face": 0,
        "skipped_multi_face": 0,
        "errors": 0
    }

    # Walk through the dataset
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"👤 Processing user: {person_name}")

        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            stats["processed"] += 1
            image_path = os.path.join(person_dir, image_name)

            try:
                # Preprocess image (lighting correction)
                image = preprocess_image(image_path)
                
                if image is None:
                    print(f"   ❌ Could not load {image_name}")
                    stats["errors"] += 1
                    continue

                # Detect face locations first so we can check count
                face_locations = face_recognition.face_locations(image, model=detection_method)

                if len(face_locations) == 0:
                    print(f"   ⚠️  Skipped: No face found in {image_name}")
                    stats["skipped_no_face"] += 1
                    continue
                
                if len(face_locations) > 1:
                    print(f"   ⚠️  Skipped: Multiple faces ({len(face_locations)}) found in {image_name}. Keep dataset pure.")
                    stats["skipped_multi_face"] += 1
                    continue

                # Compute encodings with jittering for robustness
                # We pass the known location to speed it up and ensure we encode the right face
                encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=jitters)

                if len(encodings) > 0:
                    # Convert to list for JSON serialization (numpy arrays are not JSON serializable)
                    known_encodings.append(encodings[0].tolist())
                    known_names.append(person_name)
                    stats["success"] += 1
                    print(f"   ✅ Encoded {image_name}")
                else:
                    # Should be rare if location was found
                    print(f"   ⚠️  Failed to encode face in {image_name}")
                    stats["errors"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"   ❌ Error processing {image_name}: {str(e)}")

    if stats["success"] == 0:
        print("\n❌ Error: No faces were successfully encoded.")
        return False

    # Save to file
    data = {"encodings": known_encodings, "names": known_names}
    try:
        with open(output_file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"\n❌ Error saving JSON file: {e}")
        return False

    # Final Summary
    print("\n" + "="*40)
    print("📊 ENCODING SUMMARY")
    print("="*40)
    print(f"  Total Images Scanned:  {stats['processed']}")
    print(f"  ✅ Successfully Encoded: {stats['success']}")
    print(f"  ⚠️  Skipped (No Face):    {stats['skipped_no_face']}")
    print(f"  ⚠️  Skipped (Multi Face): {stats['skipped_multi_face']}")
    print(f"  ❌ Errors:               {stats['errors']}")
    print("-" * 40)
    print(f"  Unique Persons:        {len(set(known_names))}")
    print(f"  Output File:           {output_file}")
    print("="*40 + "\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode faces from dataset for facial recognition.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to the dataset directory")
    parser.add_argument("--encodings", type=str, default="encodings.json", help="Path to save the encodings JSON file")
    parser.add_argument("--detection-method", type=str, default="hog", choices=["hog", "cnn"], help="Face detection model to use: 'hog' (fast) or 'cnn' (accurate)")
    parser.add_argument("--jitters", type=int, default=1, help="Number of times to resample the face. Higher (e.g. 10) is better for glasses/masks but slower.")
    
    args = parser.parse_args()

    success = encode_faces(
        dataset_dir=args.dataset,
        output_file=args.encodings,
        detection_method=args.detection_method,
        jitters=args.jitters
    )

    if not success:
        sys.exit(1)
