import zipfile
import os

zip_path = os.path.expandvars(
    r"%USERPROFILE%\.cache\kagglehub\datasets\gti-upm\leapgestrecog\1.archive"
)

output_dir = "dataset"

classes = [
    "01_palm",
    "03_fist",
    "05_thumb",
    "06_index",
    "07_ok"
]

max_images_per_class = 1000

print("Opening dataset...")

with zipfile.ZipFile(zip_path, "r") as z:
    counts = {c: 0 for c in classes}

    for file_info in z.infolist():

        if not file_info.filename.lower().endswith(".png"):
            continue

        parts = file_info.filename.replace("\\", "/").split("/")

        # Example:
        # leapGestRecog/00/01_palm/frame_00_01_0002.png

        if len(parts) < 4:
            continue

        class_name = parts[2]

        if class_name not in classes:
            continue

        if counts[class_name] >= max_images_per_class:
            continue

        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        file_name = os.path.basename(file_info.filename)
        output_path = os.path.join(class_folder, file_name)

        with z.open(file_info) as source:
            with open(output_path, "wb") as target:
                target.write(source.read())

        counts[class_name] += 1

        if counts[class_name] % 100 == 0:
            print(f"{class_name}: {counts[class_name]} images extracted")

print("\nExtraction completed!")

for class_name in classes:
    print(f"{class_name}: {counts[class_name]} images")