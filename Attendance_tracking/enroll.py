import os
import json
from deepface import DeepFace
import time

print("[INFO] Starting enrollment script...")

#Configuration

STUDENT_IMAGES_PATH = r"C:\Users\HP\OneDrive\Desktop\Hackloop\Image_Processing\enrollment_images" #change directory accodingly 

MODEL_NAME = "ArcFace" 

#output file
DATABASE_NAME = "face_database.json"



face_database = {}

start_time = time.time()

for root, dirs, files in os.walk(STUDENT_IMAGES_PATH):
    
    if root == STUDENT_IMAGES_PATH:
        continue
        
    usn = os.path.basename(root)
    
    if usn not in face_database:
        face_database[usn] = [] 
    
    print(f"\n[INFO] Processing student: {usn}")


    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(root, file)
        
        try:
            # It loads the image, finds the face, and converts it
            # into a 512-dimension vector (the faceprint).
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                enforce_detection=True # Fails if no face is found
            )
            
            faceprint = embedding[0]["embedding"]
            
            face_database[usn].append(faceprint)
            print(f"  > Successfully processed {file} (1 face added)")

        except ValueError as e:

            print(f"  [WARNING] No face detected in {image_path}. Skipping.")
        except Exception as e:
            print(f"  [ERROR] An error occurred with {image_path}: {e}")

# --- Save to Disk ---
try:
    with open(DATABASE_NAME, 'w') as f:
        json.dump(face_database, f, indent=4)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*30)
    print(f"[SUCCESS] Enrollment Complete!")
    print(f"Total students processed: {len(face_database)}")
    print(f"Data saved to: {DATABASE_NAME}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print("="*30)

except Exception as e:
    print(f"\n[ERROR] Could not save database file: {e}")