import os
from PIL import Image

def validate_images_in_directory(directory):
    corrupted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError) as e:
                    corrupted_files.append(file_path)
    return corrupted_files

def remove_files(file_list):
    for file_path in file_list:
        try:
            os.remove(file_path)
            print(f"Removed corrupted file: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

if __name__ == "__main__":
    train_dir = 'data/train'
    test_dir = 'data/test'

    print("Validating training images...")
    corrupted_train = validate_images_in_directory(train_dir)
    if corrupted_train:
        print(f"Corrupted images found in training directory ({len(corrupted_train)}):")
        for f in corrupted_train:
            print(f)
        print("Removing corrupted training images...")
        remove_files(corrupted_train)
    else:
        print("No corrupted images found in training directory.")

    print("\nValidating test images...")
    corrupted_test = validate_images_in_directory(test_dir)
    if corrupted_test:
        print(f"Corrupted images found in test directory ({len(corrupted_test)}):")
        for f in corrupted_test:
            print(f)
        print("Removing corrupted test images...")
        remove_files(corrupted_test)
    else:
        print("No corrupted images found in test directory.")

    # After cleaning, re-run the training script
    print("\nRe-running the training script...")
    import subprocess
    subprocess.run(["python", "4. ML Codes/nsl.py"])
