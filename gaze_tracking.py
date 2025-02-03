import cv2
import dlib
import numpy as np
from scipy.signal import savgol_filter
from collections import deque

class GazeTracker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.gaze_data = deque(maxlen=1000)

    def get_eye_aspect_ratio(self, eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def get_gaze_ratio(self, eye_region, threshold=70):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        _, binary_eye = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY)
        height, width = binary_eye.shape
        left_part = binary_eye[:, :width // 2]
        right_part = binary_eye[:, width // 2:]
        left_white = cv2.countNonZero(left_part)
        right_white = cv2.countNonZero(right_part)

        gaze_ratio = (left_white + 1) / (right_white + 1)
        return gaze_ratio

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye = np.array([[landmarks.part(36).x, landmarks.part(36).y],
                                 [landmarks.part(37).x, landmarks.part(37).y],
                                 [landmarks.part(38).x, landmarks.part(38).y],
                                 [landmarks.part(39).x, landmarks.part(39).y],
                                 [landmarks.part(40).x, landmarks.part(40).y],
                                 [landmarks.part(41).x, landmarks.part(41).y]])

            right_eye = np.array([[landmarks.part(42).x, landmarks.part(42).y],
                                  [landmarks.part(43).x, landmarks.part(43).y],
                                  [landmarks.part(44).x, landmarks.part(44).y],
                                  [landmarks.part(45).x, landmarks.part(45).y],
                                  [landmarks.part(46).x, landmarks.part(46).y],
                                  [landmarks.part(47).x, landmarks.part(47).y]])

            left_ear = self.get_eye_aspect_ratio(left_eye)
            right_ear = self.get_eye_aspect_ratio(right_eye)
            average_ear = (left_ear + right_ear) / 2.0

            left_gaze = self.get_gaze_ratio(frame[landmarks.part(37).y:landmarks.part(41).y,
                                                landmarks.part(36).x:landmarks.part(39).x])
            right_gaze = self.get_gaze_ratio(frame[landmarks.part(43).y:landmarks.part(47).y,
                                                 landmarks.part(42).x:landmarks.part(45).x])

            gaze_ratio = (left_gaze + right_gaze) / 2.0
            self.gaze_data.append(gaze_ratio)

        return frame, average_ear, gaze_ratio

    def get_smoothed_gaze_data(self, window_length=11, polyorder=2):
        if len(self.gaze_data) < window_length:
            return list(self.gaze_data)

        smoothed_data = savgol_filter(list(self.gaze_data), window_length, polyorder)
        return smoothed_data

    def detect_fixations(self, threshold=0.2, min_duration=5):
        smoothed_data = self.get_smoothed_gaze_data()
        fixations = []
        start_idx = None

        for i in range(1, len(smoothed_data)):
            if abs(smoothed_data[i] - smoothed_data[i - 1]) < threshold:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None and (i - start_idx) >= min_duration:
                    fixations.append((start_idx, i))
                    start_idx = None

        return fixations

    def save_gaze_data(self, filename='gaze_data.csv'):
        df = pd.DataFrame(list(self.gaze_data), columns=['gaze_ratio'])
        df.to_csv(filename, index=False)

def draw_eye_contours(frame, landmarks):
    for i in range(36, 48):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame

def run_gaze_tracking():
    gaze_tracker = GazeTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, ear, gaze_ratio = gaze_tracker.process_frame(frame)
        smoothed_data = gaze_tracker.get_smoothed_gaze_data()

        cv2.putText(processed_frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_frame, f"Gaze Ratio: {gaze_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Gaze Tracker', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    gaze_tracker.save_gaze_data()

if __name__ == "__main__":
    run_gaze_tracking()
