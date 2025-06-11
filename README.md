# Ishara-Setu

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Ishara-Setu](##ishara-setu)
- [Project Structure](##Project-Structure)
- [Requirements](##requirements)
- [Usage](###usage)
- [Notes](##notes)
- [Running the Project](##running-the-project)
- [License](##license)

---

## 📘 Ishara-Setu

Ishara-Setu is a Nepali Sign Language recognition project that includes data collection, image preprocessing, machine learning model training, and a user interface for real-time recognition.

---

## 🗂️ Project-Structure

- `1. Data Collection/`: Scripts for collecting and shuffling data.
- `2. Image Preprocessing/`: Image preprocessing steps and sample images.
- `3. Images To Pickle/`: Scripts to convert images to pickle format.
- `4. ML Codes/`: Machine learning model training, validation, and reports.
- `5. Analysis/`: Analysis images and reports.
- `6. UI/`: User interface scripts and resources.

---

## ⚙️ Requirements

Install the required Python packages listed in `requirement.txt`.

---

## 🚀 Usage

- Use the scripts in `1. Data Collection/` to collect and prepare data.
- Preprocess images using scripts in `2. Image Preprocessing/`.
- Train and validate models using scripts in `4. ML Codes/`.
- Use the UI in `6. UI/` for real-time sign language recognition.

---

## 📝 Notes

- Large model files, datasets, and generated files are excluded from the repository to keep it lightweight.
- Please refer to the project files for detailed implementation.

---

## ▶️ Running the Project

Since large model files, datasets, and other generated files are excluded from this repository, to run the project successfully on your machine, please follow these steps:

1. **Download Required Files and Folders**:  
   Download the following files and folders from the Google Drive link:

   [![Google Drive](https://img.shields.io/badge/Google%20Drive-Folder-blue?logo=google-drive&style=for-the-badge)](https://drive.google.com/drive/folders/1ciFBrT-N5Nv-2_vOE9Pn9AEoF-0mjy4r?usp=sharing)

   - `data/`
   - `newdata/`
   - `NSL/`
   - `nsldata/`
   - All model files with `.h5` extension (e.g., `best_model.h5`, `finalNsl.h5`, etc.)
   - Pickle files (`*.pkl`)
   - Database file `gesture_db1.db` located in `6. UI/` folder

2. **Place Files Correctly**:  
   After downloading, place these files and folders in the root directory of the project, maintaining the same folder structure.

3. **Generate Missing Files (Optional)**:  
   If you prefer not to download some files, you can generate them by running the respective scripts:

   - Use scripts in `3. Images To Pickle/` to generate pickle files.
   - Use scripts in `4. ML Codes/` to train models and generate `.h5` files.
   - Create or set up the database file as needed for the UI.

4. **Install Dependencies**:  
   Install the required Python packages listed in `requirement.txt`.

Following these steps will help you set up the project environment and run the project without issues.

---

## 📄 License

Refer to the [LICENSE](LICENSE) file in the project for license details.
