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
SMOOTHING_WINDOW = 3  # Daha hızlı tepki için düşürdüm
WINK_DIFF_THRESH = 0.02 # Eşik çok düşürüldü (Daha kolay algılasın diye)

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
        "active": True, # VARSAYILAN OLARAK AÇIK
        "threshold": 0.22
    }

STATE = st.session_state.app_state
LOCK = threading.Lock()

def speak_js(text):
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
        
        # Göz İndeksleri
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
        if now < self.wink_cooldown: return None

        with LOCK:
            thresh = STATE["threshold"]
            # active kontrolünü kaldırdım, her zaman algılasın

        left_closed = left_ear < thresh
        right_closed = right_ear < thresh
        
        action = None
        
        # 1. ÇİFT KIRPMA (BLINK) - Komutlar
        if left_closed and right_closed:
            if self.blink_start_time == 0:
                self.blink_start_time = now
        else:
            if self.blink_start_time > 0:
                duration = now - self.blink_start_time
                self.blink_start_time = 0
                
                if 0.1 < duration < 0.8: # Normal bir kırpma süresi
                    if (now - self.last_blink_end) < 0.8:
                        self.consecutive_blinks += 1
                    else:
                        self.consecutive_blinks = 1
                    self.last_blink_end = now
                    
                    if self.consecutive_blinks == 2: action = "CONFIRM"
                    elif self.consecutive_blinks == 3: action = "SPACE"
                    elif self.consecutive_blinks == 4: action = "BACKSPACE"

        # 2. TEK KIRPMA (WINK) - Mors
        # Blink sayacı 0 iken ve gözler tamamen kapalı değilken
        if self.blink_start_time == 0 and self.consecutive_blinks == 0:
            
            # SOL GÖZ (Çizgi) -> Sol kapalı, Sağ açık
            # WINK_DIFF_THRESH çok düşük olduğu için hafif fark yetecek
            if left_closed and not right_closed:
                # Ekstra kontrol: Sağ göz eşiğin %10 üzerinde mi?
                if right_ear > (thresh * 1.1): 
                    action = "DASH"
                    self.wink_cooldown = now + 0.5 

            # SAĞ GÖZ (Nokta) -> Sağ kapalı, Sol açık
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
                
                # Hafif yumuşatma
                self.left_history.append(raw_left_ear)
                self.right_history.append(raw_right_ear)
                l_ear = np.mean(self.left_history)
                r_ear = np.mean(self.right_history)
                
                action = self.process_logic(l_ear, r_ear)
                
                # Eylemleri Kaydet
                with LOCK:
                    if action == "DOT":
                        STATE["morse"] += "."
                        STATE["status"] = "Nokta (.)"
                    elif action == "DASH":
                        STATE["morse"] += "-"
                        STATE["status"] = "Cizgi (-)"
                    elif action == "CONFIRM":
                        char = decode_morse(STATE["morse"])
                        if char:
                            STATE["text"] += char
                            STATE["last_char"] = char
                            STATE["morse"] = ""
                            STATE["status"] = f"Harf: {char}"
                    elif action == "SPACE":
                        STATE["text"] += " "
                        STATE["status"] = "Bosluk"
                    elif action == "BACKSPACE":
                        STATE["text"] = STATE["text"][:-1]
                        STATE["morse"] = ""
                        STATE["status"] = "Silindi"

                # --- GÖRSELLEŞTİRME (EN ÖNEMLİ KISIM) ---
                thresh = STATE["threshold"]
                
                # Göz Durumuna Göre Renkler
                # Sol Göz
                if l_ear < thresh:
                    cv2.circle(img, (50, 100), 20, (0, 0, 255), -1) # Kırmızı Daire
                    cv2.putText(img, "KAPALI", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.circle(img, (50, 100), 10, (0, 255, 0), -1) # Yeşil Daire

                # Sağ Göz
                if r_ear < thresh:
                    cv2.circle(img, (w-50, 100), 20, (0, 0, 255), -1)
                    cv2.putText(img, "KAPALI", (w-180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.circle(img, (w-50, 100), 10, (0, 255, 0), -1)
                
                # EAR Değerlerini Yaz (Debug için)
                cv2.putText(img, f"L:{l_ear:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, f"R:{r_ear:.2f}", (w-120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Ortaya Büyükçe Durum ve Mors Yaz
                if STATE['morse']:
                    cv2.putText(img, STATE['morse'], (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                
                if action: # Bir eylem olduysa ekranda parlasın
                    cv2.putText(img, STATE['status'], (w//2 - 100, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

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
    .morse-display { 
        font-family: monospace; 
        font-size: 60px; 
        font-weight: bold; 
        color: #e74c3c; 
        text-align: center; 
        background: #fdf2f2; 
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("👁️ MORSE-EYE: Canlı Test Modu")

col1, col2 = st.columns([1.5, 1])

with col1:
    # WebRTC
    ctx = webrtc_streamer(
        key="morse-test",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=MorseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🎛️ Ayarlar")
    
    st.info("Eğer kırmızı 'KAPALI' yazısı çıkmıyorsa eşiği artırın.")
    
    # Eşik Ayarı
    thresh_val = st.slider("Göz Kapanma Eşiği (Hassasiyet)", 0.10, 0.35, STATE["threshold"], 0.01)
    STATE["threshold"] = thresh_val
    
    st.divider()
    
    st.markdown("### 📡 ANLIK MORS KODU")
    st.markdown(f"<div class='morse-display'>{STATE['morse'] if STATE['morse'] else '---'}</div>", unsafe_allow_html=True)
    
    st.markdown("### 📝 YAZILAN METİN")
    st.info(STATE["text"] if STATE["text"] else "Henüz bir şey yazılmadı...")
    
    # Butonlar
    c1, c2 = st.columns(2)
    if c1.button("🗑️ TEMİZLE"):
        STATE["text"] = ""
        STATE["morse"] = ""
        st.rerun()
    if c2.button("🔊 OKU"):
        if STATE["text"]: speak_js(STATE["text"])

# Otomatik Yenileme
if ctx.state.playing:
    time.sleep(0.5)
    st.rerun()
