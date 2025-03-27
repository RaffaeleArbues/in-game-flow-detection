import os
import zipfile

# These functions are useful to fix the format of .json files dropped by neurosity SDK

def extract_and_fix_json_files(directory_path):
    """
    Extracts only the "power_by_band.json" file from the .zip files in a directory,
    fixes its formatting, and returns the path of the corrected JSON file.

    Parameters:
        directory_path (str): Path to the directory containing the .zip files.

    Returns:
        list: A list of absolute paths to the corrected "power_by_band.json" files.
    """

    json_file_paths = []

    for file in os.listdir(directory_path):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(directory_path, file)
            extracted_path = os.path.splitext(zip_file_path)[0]

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if "power_by_band.json" in zip_ref.namelist():
                    zip_ref.extract("power_by_band.json", extracted_path)
                    json_file_path = os.path.join(extracted_path, "power_by_band.json")
                    
                    if fix_json_format(json_file_path):
                        json_file_paths.append(json_file_path)

    return json_file_paths


def fix_json_format(file_path):
    """
    Fixes the format of a JSON file by replacing '][' with '],['.

    Parameters:
        file_path (str): Path to the JSON file to be fixed.

    Returns:
        bool: True if the file was successfully fixed, False otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        if "}][{" in content:
            corrected_content = content.replace("][", ",")
            with open(file_path, 'w') as f:
                f.write(corrected_content)
            print(f"Fixed the file: {file_path}")
            return True

    except Exception as e:
        print(f"Error occurring fixing {file_path}: {e}")

    return False