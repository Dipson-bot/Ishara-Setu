import cv2
import numpy as np
import tensorflow as tf

# Parameters
roi_top, roi_bottom, roi_right, roi_left = 100, 300, 350, 550
accumulated_weight = 0.5
bg = None
img_size = 64
class_labels = [chr(0x0915 + i) for i in range(36)]  # Placeholder Nepali consonants Unicode range (adjust as needed)

def calc_accum_avg(frame, accumulated_weight):
    global bg
    if bg is None:
        bg = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, bg, accumulated_weight)

def segment_hand(frame, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

def preprocess_for_model(thresholded):
    resized = cv2.resize(thresholded, (img_size, img_size))
    normalized = resized.astype("float32") / 255.0
    reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
    return reshaped

def main():
    global bg
    model = tf.keras.models.load_model("nepali_sign_language_cnn.h5")
    cap = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        if num_frames < 60:
            calc_accum_avg(gray, accumulated_weight)
            cv2.putText(clone, "Calibrating background... Please wait", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            hand = segment_hand(gray)
            if hand is not None:
                thresholded, segmented = hand
                cv2.drawContours(clone[roi_top:roi_bottom, roi_right:roi_left], [segmented + (roi_right, roi_top)], -1, (0,255,0), 2)
                processed = preprocess_for_model(thresholded)
                prediction = model.predict(processed)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                label = class_labels[class_id]
                cv2.putText(clone, f"Predicted: {label} ({confidence*100:.2f}%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0,255,0), 2)
        cv2.imshow("Real-Time Nepali Sign Language Recognition", clone)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        num_frames += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
