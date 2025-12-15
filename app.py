import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import math

# --- AYARLAR ---
# Göz Hassasiyet Ayarları (Kişiden kişiye değişebilir)
RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153] # MediaPipe Göz Landmarkları
LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]

# Mors Alfabesi Sözlüğü
MORSE_CODE_DICT = { 
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', 
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', 
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', 
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', 
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', 
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3', 
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9'
}

# Ses Motorunu Başlat
engine = pyttsx3.init()

class MorseEyeApp:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        
        # Durum Değişkenleri
        self.current_morse = ""
        self.decoded_message = ""
        self.last_look_direction = "CENTER" # CENTER, LEFT, RIGHT
        self.blink_counter = 0
        self.look_start_time = 0
        self.blink_start_time = 0
        self.consecutive_blinks = 0
        self.last_blink_time = 0
        self.looking_down_start = None
        
    def speak(self, text):
        """Metni sesli okur"""
        if text:
            engine.say(text)
            engine.runAndWait()

    def landmark_detection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        return mesh_coord

    def blink_ratio(self, img, landmarks, right_indices, left_indices):
        """Gözün açıklık oranını hesaplar (Göz kırpma tespiti için)"""
        # Sağ Göz
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[1]]
        rv_top = landmarks[right_indices[2]]
        rv_bottom = landmarks[right_indices[6]]
        
        # Yatay ve Dikey mesafe
        rh_distance = math.hypot(rh_right[0]-rh_left[0], rh_right[1]-rh_left[1])
        rv_distance = math.hypot(rv_top[0]-rv_bottom[0], rv_top[1]-rv_bottom[1])
        
        if rv_distance == 0: return 0
        ratio = rh_distance / rv_distance
        return ratio

    def gaze_detection(self, img, landmarks, iris_indices, eye_indices):
        """Gözün nereye baktığını tespit eder (İrisin göz içindeki konumu)"""
        try:
            eye_left = np.array(landmarks[eye_indices[0]])
            eye_right = np.array(landmarks[eye_indices[1]])
            iris_center = np.array(landmarks[iris_indices[0]]) # İris merkezi (MediaPipe 468 veya 473)
            
            # Gözün toplam genişliği
            eye_width = np.linalg.norm(eye_right - eye_left)
            
            # İrisin sol köşeye uzaklığı
            iris_dist = np.linalg.norm(iris_center - eye_left)
            
            # Oran: 0.0 (Tam Sol) - 1.0 (Tam Sağ)
            ratio = iris_dist / eye_width
            return ratio
        except:
            return 0.5

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success: break

            image = cv2.flip(image, 1) # Ayna etkisi
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            img_h, img_w, _ = image.shape
            status_text = "MERKEZ"
            
            if results.multi_face_landmarks:
                mesh_coords = self.landmark_detection(image, results, False)
                
                # --- GÖZ KIRPMA TESPİTİ ---
                ratio = self.blink_ratio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
                if ratio > 5.5: # Göz kapalı (Eşik değeri ayarlanabilir)
                    if self.blink_start_time == 0:
                        self.blink_start_time = time.time()
                    
                    cv2.putText(image, "KIRPMA", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                else:
                    if self.blink_start_time != 0:
                        duration = time.time() - self.blink_start_time
                        if duration < 0.5: # Kısa kırpma
                            if (time.time() - self.last_blink_time) < 1.0:
                                self.consecutive_blinks += 1
                            else:
                                self.consecutive_blinks = 1
                            self.last_blink_time = time.time()
                        self.blink_start_time = 0

                # --- BAKIŞ YÖNÜ TESPİTİ (468: Sol İris, 473: Sağ İris) ---
                gaze_ratio_left = self.gaze_detection(image, mesh_coords, [468], LEFT_EYE)
                gaze_ratio_right = self.gaze_detection(image, mesh_coords, [473], RIGHT_EYE)
                avg_gaze = (gaze_ratio_left + gaze_ratio_right) / 2

                # Eşik Değerler (Kalibrasyon gerekebilir)
                if avg_gaze < 0.42: # Sola Bakış
                    direction = "LEFT"
                elif avg_gaze > 0.58: # Sağa Bakış
                    direction = "RIGHT"
                else: # Merkez
                    direction = "CENTER"
                    
                # Yukarı/Aşağı Tespiti (Basit Y koordinatı kontrolü)
                nose_tip = mesh_coords[1]
                forehead = mesh_coords[10]
                face_vertical_len = math.hypot(nose_tip[0]-forehead[0], nose_tip[1]-forehead[1])
                
                # Aşağı Bakış (Sesli Okuma İçin)
                if mesh_coords[468][1] > mesh_coords[6][1] + (face_vertical_len*0.3): # Göz burun hizasına yakınsa
                    if self.looking_down_start is None:
                        self.looking_down_start = time.time()
                    elif time.time() - self.looking_down_start > 2.0:
                        self.speak(self.decoded_message)
                        self.looking_down_start = None
                        cv2.putText(image, "SESLI OKUMA...", (200, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                else:
                    self.looking_down_start = None

                # --- MORS MANTIĞI ---
                if direction != self.last_look_direction:
                    if direction == "LEFT": # Nokta
                        self.current_morse += "."
                        self.speak("dıt") # Geri bildirim sesi
                    elif direction == "RIGHT": # Çizgi
                        self.current_morse += "-"
                        self.speak("daa")
                    self.last_look_direction = direction

                # --- KIRPMA KOMUTLARI ---
                # 2 Kırpma: Harfi Onayla
                if self.consecutive_blinks == 2 and (time.time() - self.last_blink_time) > 1.0:
                    letter = MORSE_CODE_DICT.get(self.current_morse, "?")
                    self.decoded_message += letter
                    self.current_morse = "" # Sıfırla
                    self.consecutive_blinks = 0
                    self.speak("Onay")
                
                # 3 Kırpma: Boşluk
                if self.consecutive_blinks == 3 and (time.time() - self.last_blink_time) > 1.0:
                    self.decoded_message += " "
                    self.consecutive_blinks = 0
                    self.speak("Boşluk")

                # 5 Kırpma: Temizle/Başla
                if self.consecutive_blinks >= 5:
                    self.decoded_message = ""
                    self.current_morse = ""
                    self.consecutive_blinks = 0
                    self.speak("Sistem Sıfırlandı")

                status_text = f"YON: {direction}"

            # --- EKRAN ARAYÜZÜ ---
            # Arka plan paneli
            cv2.rectangle(image, (0, 0), (img_w, 150), (30, 30, 30), -1)
            
            # Metinler
            cv2.putText(image, f"MORS: {self.current_morse}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f"MESAJ: {self.decoded_message}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"DURUM: {status_text}", (img_w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(image, f"KIRPMA: {self.consecutive_blinks}", (img_w - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Yardımcı Çizgiler (İsteğe bağlı, gözlerin üzerine nokta koyar)
            if results.multi_face_landmarks:
                cv2.circle(image, mesh_coords[473], 3, (0, 255, 0), -1) # Sağ İris
                cv2.circle(image, mesh_coords[468], 3, (0, 255, 0), -1) # Sol İris

            cv2.imshow('MORSE-EYE Project', image)
            if cv2.waitKey(5) & 0xFF == 27: # ESC ile çık
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MorseEyeApp()
    app.run()