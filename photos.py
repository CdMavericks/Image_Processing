import mysql.connector
import requests
import os
from pathlib import Path

OUTPUT_DIR = "enrollment_images" 


# MySQL DB
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
# List of columns containing image URLs in your enrollments table
URL_COLUMNS = [
    'url_frontal',
    'url_left',
    'url_right',
    'url_up',
    'url_down'
]

def download_images_from_db():
    """Connects to MySQL, fetches enrollment data, and downloads images."""
    
    print(f"Connecting to database: {DB_NAME} at {DB_HOST}:{DB_PORT}...")
    
    try:
        # Establish connection to the online MySQL server
        db = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            # SSL mode REQUIRED for your Aiven database
            ssl_disabled=False,
            #ssl_verify_identity=True 
            # If the above SSL lines fail due to certificate issues, try:
            # ssl_verify_identity=False 
        )
        
        cursor = db.cursor(dictionary=True)
        print("✅ Database connection successful.")
        
        # 1. Fetch all enrollment data
        columns_to_fetch = ['USN'] + URL_COLUMNS
        select_query = f"SELECT {', '.join(columns_to_fetch)} FROM enrollments"
        cursor.execute(select_query)
        
        enrollments = cursor.fetchall()
        
        if not enrollments:
            print("🛑 No enrollment data found in the table.")
            return

        # 2. Setup output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        print(f"💾 Downloading images for {len(enrollments)} students...")

        # 3. Process each student enrollment
        for row in enrollments:
            usn = row['USN']
            
            # Create a dedicated directory for the USN
            usn_dir = Path(OUTPUT_DIR) / usn
            usn_dir.mkdir(exist_ok=True)
            
            print(f"\n--- Processing USN: {usn} ---")
            
            # 4. Download each image URL
            for pose_name in URL_COLUMNS:
                url = row[pose_name]
                
                if not url:
                    print(f"⚠️ Missing URL for {pose_name}. Skipping.")
                    continue

                # Determine the filename (e.g., frontal.jpg, left.jpg)
                # We use .split('_')[-1] to get the pose name from 'url_frontal'
                filename = f"{pose_name.split('_')[-1]}.jpg" 
                file_path = usn_dir / filename
                
                try:
                    # Download the image data
                    img_data = requests.get(url).content
                    
                    # Write the image data to the local file
                    with open(file_path, 'wb') as handler:
                        handler.write(img_data)
                    
                    print(f"  Downloaded: {filename}")
                
                except requests.exceptions.RequestException as e:
                    print(f"❌ Failed to download {pose_name} from {url}: {e}")
                except Exception as e:
                    print(f"❌ An error occurred while saving {filename}: {e}")

        print("\n✅ All enrollment images processed and downloaded.")

    except mysql.connector.Error as err:
        print(f"\n❌ Database Error: {err}")
        print("Please check your .env credentials and firewall settings.")
        
    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()
            print("Connection closed.")

if __name__ == "__main__":
    download_images_from_db()