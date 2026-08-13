import zipfile
import json
import shutil
import os

SOURCE_MODEL = "pneumonia_model.keras"
OUTPUT_MODEL = "pneumonia_model_fixed.keras"

print("Reading original model...")

# Create a temporary folder
TEMP_DIR = "keras_temp"

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR)

# Extract the .keras file
with zipfile.ZipFile(SOURCE_MODEL, "r") as z:
    z.extractall(TEMP_DIR)

print("Reading config.json...")

config_path = os.path.join(TEMP_DIR, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


def remove_quantization_config(obj):
    """Remove quantization_config from all layer configurations."""

    if isinstance(obj, dict):
        obj.pop("quantization_config", None)

        for value in obj.values():
            remove_quantization_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


remove_quantization_config(config)

print("Removed unsupported quantization_config entries.")

# Save repaired config
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("Creating repaired model...")

# Create new .keras file
with zipfile.ZipFile(OUTPUT_MODEL, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            file_path = os.path.join(root, file)

            archive_name = os.path.relpath(
                file_path,
                TEMP_DIR
            )

            z.write(file_path, archive_name)

# Remove temporary folder
shutil.rmtree(TEMP_DIR)

print()
print("SUCCESS!")
print(f"Created: {OUTPUT_MODEL}")