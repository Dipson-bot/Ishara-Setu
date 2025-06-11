from tkinter import *
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pygame
from gtts import gTTS

# Load model
model = load_model("finalNsl.h5")

# App Config
root = Tk()
root.title("Nepali Sign Language Recognition and Translation")
root.iconbitmap("6. UI/icon.ico")
root.config(bg="#f7f7f7")  # Light background

# Data Variables
language = "ne"
words = {"text": "", "words1": "", "words2": ""}
text = StringVar()
words1 = StringVar()
words2 = StringVar()

# Label mapping
index_to_label = {
    0: "क", 1: "ख", 2: "ग", 3: "घ", 4: "ङ",
    5: "च", 6: "छ", 7: "ज", 8: "झ", 9: "ञ",
    10: "ट", 11: "ठ", 12: "ड", 13: "ढ", 14: "ण",
    15: "त", 16: "थ", 17: "द", 18: "ध", 19: "न",
    20: "प", 21: "फ", 22: "ब", 23: "भ", 24: "म",
    25: "य", 26: "र", 27: "ल", 28: "व",
    29: "श", 30: "ष", 31: "स", 32: "ह",
    33: "क्ष", 34: "त्र", 35: "ज्ञ", 36: "खाली"
}

# Fonts and styles
bold_font = ("Segoe UI", 14, "bold")
header_font = ("Segoe UI", 18, "bold")
label_color = "#333333"
highlight_color = "#ff6600"

# Title Label
Label(root, text="Nepali Sign Language Recognition & Translation", font=header_font,
      fg="#0b5394", bg="#f7f7f7", pady=10).pack()

Label(root, text="Predicted Sign", font=bold_font, fg=highlight_color, bg="#f7f7f7").pack()
Label(root, textvariable=text, font=bold_font, fg=label_color, bg="#f7f7f7").pack()

Label(root, text="Word", font=bold_font, fg=highlight_color, bg="#f7f7f7").pack()
Label(root, textvariable=words1, font=bold_font, fg=label_color, bg="#f7f7f7").pack()

Label(root, text="Sentence", font=bold_font, fg=highlight_color, bg="#f7f7f7").pack()
Label(root, textvariable=words2, font=bold_font, fg=label_color, bg="#f7f7f7").pack()

# Video feed
video_label = Label(root, bg="#f7f7f7")
video_label.pack(pady=5)

# Upload Image Prediction Function
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        try:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))
            pred_probab, pred_class = keras_predict(model, gray)
            if pred_probab > 0.9:
                label_pred = index_to_label.get(pred_class, "")
                words["text"] = label_pred
                messagebox.showinfo("Prediction", f"Predicted Sign: {label_pred}")
            else:
                messagebox.showwarning("Low Confidence", "Prediction confidence is low.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process the image.\n\n{e}")

# Upload Button
upload_btn = Button(root, text="Upload Image for Prediction", font=bold_font,
                    command=upload_and_predict, bg="#007acc", fg="white", padx=10, pady=5)
upload_btn.pack(pady=10)

# Keypress Handler
def check_keypress(event):
    if event.keysym == "s":
        words["words1"] += words["text"]
    elif event.keysym == "q":
        words["words2"] += words["words1"] + " "
        words["words1"] = ""
    elif event.keysym == "a":
        tts = gTTS(words["words2"], lang="ne")
        tts.save("hello.mp3")
        pygame.mixer.init()
        sound = pygame.mixer.Sound("hello.mp3")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))

video_label.bind("<Key>", check_keypress)
video_label.focus_set()

# Update dynamic text on UI
def update_text():
    text.set(words["text"])
    words1.set(words["words1"])
    words2.set(words["words2"])
    root.after(500, update_text)

# Image Preprocessing
def keras_process_image(img):
    img = cv2.resize(img, (128, 128))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, 128, 128, 1))
    return img

# Prediction using Keras model
def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = np.argmax(pred_probab)
    return max(pred_probab), pred_class

# Webcam Feed
cap = cv2.VideoCapture(0)

def show_frame():
    ret, frame = cap.read()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(image, 1)
        cv2.rectangle(frame, (270, 9), (620, 355), (0, 102, 204), 2)
        roi = frame[50:350, 270:570]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        blur = cv2.bilateralFilter(blur, 3, 75, 75)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        ret, roi = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.resize(roi, (128, 128))
        pred_probab, pred_class = keras_predict(model, roi)
        if pred_probab == 1.0:
            words["text"] = index_to_label.get(pred_class, "")
        img_pil = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img_pil)
        video_label.config(image=photo)
        video_label.image = photo
    root.after(10, show_frame)

# Start
show_frame()
update_text()
root.mainloop()