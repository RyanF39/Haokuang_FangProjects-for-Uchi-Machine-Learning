#%%
import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Theme: Street Traffic
classes = ["Car", "Bus", "Truck", "Motorcycle", "Bicycle"]

# STRICT CONSTRAINT: We specifically request 300 images PER CLASS.
# 300 * 5 classes = 1,500 total images.
# This satisfies:
#   a. Minimum ~1,000 total images 
#   b. At least 100 images per class [cite: 21]
samples_per_class = 300 
split_type = "train"

# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
final_output_dir = os.path.join(repo_root, "bigdata", "traffic_data")

print(f" Target Directory: {final_output_dir}")
print(f" Goal: EXACTLY {samples_per_class} images for EACH class.")

# ---------------------------------------------------------
# 1. CLEANUP 
# ---------------------------------------------------------
if os.path.exists(final_output_dir):
    print(" Cleaning up old data folder...")
    shutil.rmtree(final_output_dir)

# Create the class folders immediately
for c in classes:
    os.makedirs(os.path.join(final_output_dir, c), exist_ok=True)

# ---------------------------------------------------------
# 2. DOWNLOAD LOOP (The "Balanced" Strategy)
# ---------------------------------------------------------
# We download one class at a time to guarantee the count.

for target_class in classes:
    print(f"\n  Processing Class: {target_class}...")
    
    # Download specific subset for this class
    # We use a unique dataset_name for every class to prevent mixing
    dataset_name = f"download_{target_class}_final"
    
    try:
        # Check if dataset exists in FiftyOne and delete it to ensure fresh download
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split_type,
            label_types=["classifications"],
            classes=[target_class], # Only ask for THIS class
            max_samples=samples_per_class,
            shuffle=True,
            seed=42,
            dataset_name=dataset_name
        )
    except Exception as e:
        print(f" Error downloading {target_class}: {e}")
        continue

    # Move files to the correct folder
    count = 0
    dest_folder = os.path.join(final_output_dir, target_class)
    
    for sample in dataset:
        src_path = sample.filepath
        if os.path.exists(src_path):
            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_folder, filename)
            
            # Copy file
            shutil.copy2(src_path, dest_path)
            count += 1
            
    print(f" Successfully saved {count} images to /{target_class}")

# ---------------------------------------------------------
# 3. FINAL VERIFICATION
# ---------------------------------------------------------
print("\n" + "="*40)
print("FINAL DATASET AUDIT")
print("="*40)

total_images = 0
success = True

for c in classes:
    folder_path = os.path.join(final_output_dir, c)
    if os.path.exists(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"📂 {c}: {count} images")
        total_images += count
        
        if count < 100:
            print(f"   FAIL: Less than 100 images!")
            success = False
    else:
        print(f" {c}: 0 images (Missing Folder!)")
        success = False

print("-" * 40)
print(f"TOTAL IMAGES: {total_images}")

if success and total_images >= 1000:
    print("\n SUCCESS: Dataset meets all syllabus requirements.")
    print(f"   Location: {final_output_dir}")
else:
    print("\n FAILURE: Dataset is still missing requirements.")
# %%
