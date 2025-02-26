import os

def ensure_folder_exists(folder_path):
    """Ensure the folder exists. If not, create it."""
    os.makedirs(folder_path, exist_ok=True)

def ensure_file_overwrite(file_path):
    """Check if file exists and automatically overwrite it."""
    if os.path.isfile(file_path):
        print(f"Overwriting existing file: {file_path}")
    return True  # Always overwrite

# Example usage:
# ensure_folder_exists("data/raw")
# ensure_file_overwrite("data/raw/raw.csv")
