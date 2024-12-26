import cv2
import numpy as np
import time
from collections import deque
import threading

class BalloonDetector:
    def __init__(self):
        # Pencere isimleri
        self.window_detection = "Balon Tespiti"
        self.window_controls = "Kontroller"
        
        # FPS hesaplama için değişkenler
        self.fps_start_time = 0
        self.fps = 0
        self.frame_count = 0
        
        # Son tespit edilen balonların merkez noktalarını takip etmek için
        self.red_points = deque(maxlen=32)
        self.blue_points = deque(maxlen=32)
        
        # Varsayılan HSV değerleri
        self.hsv_values = {
            'red_low_h': 0, 'red_low_s': 120, 'red_low_v': 70,
            'red_high_h': 10, 'red_high_s': 255, 'red_high_v': 255,
            'blue_low_h': 100, 'blue_low_s': 120, 'blue_low_v': 70,
            'blue_high_h': 130, 'blue_high_s': 255, 'blue_high_v': 255
        }
        
        self.create_trackbars()

    def create_trackbars(self):
        cv2.namedWindow(self.window_controls)
        for key in self.hsv_values:
            cv2.createTrackbar(key, self.window_controls, self.hsv_values[key], 255, lambda x: None)

    def get_trackbar_values(self):
        values = {}
        for key in self.hsv_values:
            values[key] = cv2.getTrackbarPos(key, self.window_controls)
        return values

    def preprocess_frame(self, frame):
        # Görüntü boyutunu küçült
        frame = cv2.resize(frame, (640, 480))
        # Gürültü azaltma
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame

    def detect_balloons(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        values = self.get_trackbar_values()

        # Kırmızı renk maskeleri
        lower_red1 = np.array([values['red_low_h'], values['red_low_s'], values['red_low_v']])
        upper_red1 = np.array([values['red_high_h'], values['red_high_s'], values['red_high_v']])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, values['red_low_s'], values['red_low_v']])
        upper_red2 = np.array([180, values['red_high_s'], values['red_high_v']])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Mavi renk maskesi
        lower_blue = np.array([values['blue_low_h'], values['blue_low_s'], values['blue_low_v']])
        upper_blue = np.array([values['blue_high_h'], values['blue_high_s'], values['blue_high_v']])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        return mask_red, mask_blue

    def analyze_shape(self, contour):
        # Dairesellik analizi
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return 0.5 < circularity < 1.2  # Balon şekline uygun aralık

    def draw_bounding_boxes(self, frame, mask_red, mask_blue):
        frame_with_boxes = frame.copy()

        # Kırmızı balonları işaretle
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000 and self.analyze_shape(contour):
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                self.red_points.appendleft(center)
                
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(frame_with_boxes, center, 5, (0, 0, 255), -1)
                cv2.putText(frame_with_boxes, f"Kirmizi Balon ({area:.0f})", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Mavi balonları işaretle
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000 and self.analyze_shape(contour):
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                self.blue_points.appendleft(center)
                
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame_with_boxes, center, 5, (255, 0, 0), -1)
                cv2.putText(frame_with_boxes, f"Mavi Balon ({area:.0f})", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Hareket izlerini çiz
        self.draw_motion_trails(frame_with_boxes)
        
        # FPS göster
        self.calculate_fps()
        cv2.putText(frame_with_boxes, f"FPS: {self.fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame_with_boxes

    def draw_motion_trails(self, frame):
        # Kırmızı balonların izlerini çiz
        for i in range(1, len(self.red_points)):
            if self.red_points[i - 1] is None or self.red_points[i] is None:
                continue
            cv2.line(frame, self.red_points[i - 1], self.red_points[i], (0, 0, 255), 1)

        # Mavi balonların izlerini çiz
        for i in range(1, len(self.blue_points)):
            if self.blue_points[i - 1] is None or self.blue_points[i] is None:
                continue
            cv2.line(frame, self.blue_points[i - 1], self.blue_points[i], (255, 0, 0), 1)

    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count == 1:
            self.fps_start_time = time.time()
        else:
            seconds = time.time() - self.fps_start_time
            self.fps = self.frame_count / seconds
        if self.frame_count == 30:
            self.frame_count = 0
            self.fps_start_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera açılamadı!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Görüntü alınamadı!")
                break

            # Görüntü ön işleme
            frame = self.preprocess_frame(frame)

            # Balon tespiti
            mask_red, mask_blue = self.detect_balloons(frame)

            # Sonuçları çiz
            frame_with_boxes = self.draw_bounding_boxes(frame, mask_red, mask_blue)

            # Görüntüleri göster
            cv2.imshow(self.window_detection, frame_with_boxes)
            #cv2.imshow("Kirmizi Mask", mask_red)
            #cv2.imshow("Mavi Mask", mask_blue)

            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = BalloonDetector()
    detector.run()