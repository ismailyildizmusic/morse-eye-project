import av
import time
import threading
from collections import deque
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import mediapipe as mp

# -----------------------------
# AYARLAR
# -----------------------------
SMOOTHING_WINDOW = 3
WINK_DIFF_THRESH = 0.02
AUTO_CONFIRM_DELAY = 1.5  # Saniye cinsinden bekleme süresi

MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9"
}

# -----------------------------
# GLOBAL DURUM
# -----------------------------
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "morse": "",
        "text": "",
        "last_char": "",
        "status": "Sistem Hazır",
        "active": True,
        "threshold": 0.22,
        "last_input_time": 0,    # Son giriş zamanı
        "speech_queue": "",      # Seslendirilecek metin kuyruğu
        "auto_read": True        # Otomatik okuma modu
    }

STATE = st.session_state.app_state
LOCK = threading.Lock()

# -----------------------------
# SESLENDİRME (JS)
# -----------------------------
def speak_js(text):
    if not text: return
    safe_text = text.replace('"', '\\"')
    js = f"""
    <script>
        window.speechSynthesis.cancel();
        let msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.lang = "tr-TR";
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js, height=0)

def decode_morse(code):
    return MORSE_TO_CHAR.get(code, "")

# -----------------------------
# GÖRÜNTÜ İŞLEME MOTORU
# -----------------------------
class MorseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.left_history = deque(maxlen=SMOOTHING_WINDOW)
        self.right_history = deque(maxlen=SMOOTHING_WINDOW)
        
        self.blink_start_time = 0
        self.wink_cooldown = 0
        self.consecutive_blinks = 0
        self.last_blink_end = 0
        
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def get_ear(self, landmarks, indices, w, h):
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        hor = np.linalg.norm(coords[0] - coords[3])
        if hor == 0: return 0
        return (v1 + v2) / (2.0 * hor)

    def process_logic(self, left_ear, right_ear):
        now = time.time()
        
        with LOCK:
            thresh = STATE["threshold"]
            current_morse = STATE["morse"]
            last_input = STATE["last_input_time"]

        # --- OTOMATİK ONAY SİSTEMİ ---
        # Eğer mors kodu varsa ve belirli süre işlem yapılmadıysa onayla
        if current_morse and (now - last_input > AUTO_CONFIRM_DELAY):
            # Otomatik Confirm Tetikle
            return "AUTO_CONFIRM"

        if now < self.wink_cooldown: return None

        left_closed = left_ear < thresh
        right_closed = right_ear < thresh
        
        action = None
        
        # 1. ÇİFT KIRPMA (Komutlar)
        if left_closed and right_closed:
            if self.blink_start_time == 0:
                self.blink_start_time = now
        else:
            if self.blink_start_time > 0:
                duration = now - self.blink_start_time
                self.blink_start_time = 0
                
                if 0.1 < duration < 0.8:
                    if (now - self.last_blink_end) < 0.8:
                        self.consecutive_blinks += 1
                    else:
                        self.consecutive_blinks = 1
                    self.last_blink_end = now
                    
                    # 3 Kere kırpma = Boşluk (Kelime bitti)
                    if self.consecutive_blinks == 3: action = "SPACE"
                    elif self.consecutive_blinks == 4: action = "BACKSPACE"

        # 2. TEK KIRPMA (Mors Girişi)
        if self.blink_start_time == 0 and self.consecutive_blinks == 0:
            # SOL GÖZ (Çizgi)
            if left_closed and not right_closed:
                if right_ear > (thresh * 1.1): 
                    action = "DASH"
                    self.wink_cooldown = now + 0.5 

            # SAĞ GÖZ (Nokta)
            elif right_closed and not left_closed:
                if left_ear > (thresh * 1.1):
                    action = "DOT"
                    self.wink_cooldown = now + 0.5

        return action

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w, c = img.shape
            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                
                # EAR Hesapla
                raw_left_ear = self.get_ear(lm, self.LEFT_EYE, w, h)
                raw_right_ear = self.get_ear(lm, self.RIGHT_EYE, w, h)
                
                self.left_history.append(raw_left_ear)
                self.right_history.append(raw_right_ear)
                l_ear = np.mean(self.left_history)
                r_ear = np.mean(self.right_history)
                
                action = self.process_logic(l_ear, r_ear)
                
                # State Güncelleme
                with LOCK:
                    now = time.time()
                    
                    if action == "DOT":
                        STATE["morse"] += "."
                        STATE["status"] = "Nokta (.)"
                        STATE["last_input_time"] = now # Süreyi sıfırla
                    elif action == "DASH":
                        STATE["morse"] += "-"
                        STATE["status"] = "Cizgi (-)"
                        STATE["last_input_time"] = now # Süreyi sıfırla
                    
                    elif action == "AUTO_CONFIRM" or action == "CONFIRM":
                        char = decode_morse(STATE["morse"])
                        if char:
                            STATE["text"] += char
                            STATE["last_char"] = char
                            STATE["morse"] = "" # Tamponu temizle
                            STATE["status"] = f"Onaylandi: {char}"
                            # Sesli okuma kuyruğuna ekle (Harf harf okumasını istersek buraya)
                            if STATE["auto_read"]:
                                STATE["speech_queue"] = char
                        else:
                             STATE["morse"] = "" # Geçersizse sil
                    
                    elif action == "SPACE":
                        STATE["text"] += " "
                        STATE["status"] = "Bosluk"
                        # Boşlukta tüm kelimeyi okusun mu? (Opsiyonel)
                        
                    elif action == "BACKSPACE":
                        STATE["text"] = STATE["text"][:-1]
                        STATE["morse"] = ""
                        STATE["status"] = "Silindi"

                # --- GÖRSELLEŞTİRME ---
                thresh = STATE["threshold"]
                
                # 1. Göz Durumları (Daireler)
                color_l = (0, 0, 255) if l_ear < thresh else (0, 255, 0)
                color_r = (0, 0, 255) if r_ear < thresh else (0, 255, 0)
                
                cv2.circle(img, (50, 80), 15, color_l, -1)
                cv2.putText(img, "SOL", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                cv2.circle(img, (w-50, 80), 15, color_r, -1)
                cv2.putText(img, "SAG", (w-70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # 2. Ortada Kocaman Mors Kodu
                if STATE['morse']:
                    # Metni ortala
                    text_size = cv2.getTextSize(STATE['morse'], cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(img, STATE['morse'], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                    
                    # 3. Otomatik Onay İçin Zaman Çubuğu (Progress Bar)
                    time_passed = time.time() - STATE["last_input_time"]
                    progress = min(time_passed / AUTO_CONFIRM_DELAY, 1.0)
                    
                    bar_w = 300
                    bar_h = 20
                    bar_x = (w - bar_w) // 2
                    bar_y = text_y + 40
                    
                    # Arkaplan
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
                    # Dolum (Mavi)
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (255, 200, 0), -1)
                    cv2.putText(img, "Bekleyince Kaydeder...", (bar_x, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # 4. Alt Rehber (Footer)
                footer_h = 80
                cv2.rectangle(img, (0, h - footer_h), (w, h), (20, 20, 20), -1)
                
                # Rehber metinleri 
                guide_text = "SAG: Nokta (.)  |  SOL: Cizgi (-)  |  BEKLE: Kaydet  |  3x KIRP: Bosluk"
                cv2.putText(img, guide_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            else:
                cv2.putText(img, "YUZ ARANIYOR...", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            return frame

# -----------------------------
# ARAYÜZ
# -----------------------------
st.set_page_config(page_title="MORSE-EYE PRO", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .big-text { font-size: 24px; font-weight: bold; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("👁️ MORSE-EYE: Akıllı Asistan Modu")

col1, col2 = st.columns([2, 1])

with col1:
    # WebRTC
    ctx = webrtc_streamer(
        key="morse-assist",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=MorseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("⚙️ Ayarlar")
    
    # Eşik Ayarı
    thresh_val = st.slider("Hassasiyet Eşiği", 0.10, 0.35, STATE["threshold"], 0.01, help="Gözleriniz kapalıyken kırmızı yanmıyorsa artırın.")
    STATE["threshold"] = thresh_val
    
    # Sesli Okuma Modu
    auto_read = st.toggle("🔊 Otomatik Sesli Okuma", value=True)
    STATE["auto_read"] = auto_read
    
    st.divider()
    st.markdown("### 📝 YAZILAN METİN")
    st.text_area("", value=STATE["text"], height=150, disabled=True)
    
    # Butonlar
    c1, c2 = st.columns(2)
    if c1.button("🗑️ SİL"):
        STATE["text"] = ""
        STATE["morse"] = ""
        st.rerun()
    if c2.button("🔊 HEPSİNİ OKU"):
        if STATE["text"]: speak_js(STATE["text"])

# Sesli Okuma Tetikleyicisi (JS Enjeksiyonu)
if STATE["speech_queue"]:
    speak_js(STATE["speech_queue"])
    STATE["speech_queue"] = "" # Kuyruğu temizle
    
# Otomatik Yenileme
if ctx.state.playing:
    time.sleep(0.5)
    st.rerun()
