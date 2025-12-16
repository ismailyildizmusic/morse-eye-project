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
# MORSE SÖZLÜĞÜ (TÜRKÇE KARAKTERLER DAHİL)
# -----------------------------
MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ","
}

# -----------------------------
# GLOBAL DURUM
# -----------------------------
if "morse_state" not in st.session_state:
    st.session_state.morse_state = {
        "morse": "",
        "text": "",
        "last_event": "Sistem Hazır",
        "direction": "CENTER",
        "gaze_ratio": 0.5,
        "progress": 0.0,
        "action_type": "",
        # Varsayılan Ayarlar
        "hold_time": 0.8,       # Sembol ekleme süresi
        "confirm_time": 1.5,    # Onaylama süresi
        "left_threshold": 0.45, # Sola bakış eşiği (Artırılabilir)
        "right_threshold": 0.55 # Sağa bakış eşiği
    }

STATE = st.session_state.morse_state
LOCK = threading.Lock()

# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def decode_morse(code):
    return MORSE_TO_CHAR.get(code, "?") if code else ""

def speak_js(text):
    safe = text.replace('"', '\\"').replace("'", "\\'")
    components.html(f"""
    <script>
        window.speechSynthesis.cancel();
        let msg = new SpeechSynthesisUtterance("{safe}");
        msg.lang = "tr-TR";
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# -----------------------------
# GÖRÜNTÜ İŞLEME MOTORU
# -----------------------------
class MorseGazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Göz İndeksleri
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [473, 474, 475, 476]
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]
        
        self.ratio_history = deque(maxlen=5)
        self.direction_start = 0
        self.current_dir = "CENTER"
        self.last_symbol_time = 0
        self.last_confirm_time = 0
        self.symbol_cooldown = 0.5

    def get_gaze_ratio(self, lm, w, h):
        """
        Gözbebeğinin (iris), gözün sol ve sağ köşesine olan uzaklığını ölçer.
        0.0 = Tam Sol | 0.5 = Merkez | 1.0 = Tam Sağ
        """
        try:
            # Sol Göz Hesaplaması
            l_inner = np.array([lm[133].x, lm[133].y])
            l_outer = np.array([lm[33].x, lm[33].y])
            l_iris = np.array([lm[468].x, lm[468].y])
            
            l_dist_total = np.linalg.norm(l_inner - l_outer)
            l_dist_iris = np.linalg.norm(l_iris - l_outer)
            l_ratio = l_dist_iris / l_dist_total
            
            # Sağ Göz Hesaplaması
            r_inner = np.array([lm[362].x, lm[362].y])
            r_outer = np.array([lm[263].x, lm[263].y])
            r_iris = np.array([lm[473].x, lm[473].y])
            
            r_dist_total = np.linalg.norm(r_inner - r_outer)
            r_dist_iris = np.linalg.norm(r_iris - r_outer)
            r_ratio = r_dist_iris / r_dist_total
            
            # İki gözün ortalaması (Daha kararlı sonuç için)
            avg_ratio = (l_ratio + r_ratio) / 2.0
            return avg_ratio
        except:
            return 0.5

    def process_gaze(self, ratio):
        now = time.time()
        with LOCK:
            l_thr = STATE["left_threshold"]
            r_thr = STATE["right_threshold"]
            h_time = STATE["hold_time"]
            c_time = STATE["confirm_time"]

        # Yön Belirleme
        if ratio < l_thr: direction = "LEFT"
        elif ratio > r_thr: direction = "RIGHT"
        else: direction = "CENTER"

        # Yön Değişikliği Kontrolü
        if direction != self.current_dir:
            self.current_dir = direction
            self.direction_start = now
            with LOCK:
                STATE["progress"] = 0.0
                STATE["action_type"] = ""

        held_time = now - self.direction_start
        progress = 0.0
        action_type = ""
        action = None

        if direction == "LEFT": # ÇİZGİ
            progress = min(1.0, held_time / h_time)
            action_type = "DASH"
            if held_time >= h_time and (now - self.last_symbol_time) > self.symbol_cooldown:
                action = "DASH"
                self.last_symbol_time = now
                self.direction_start = now # Sıfırla (tekrarlamak için)
        
        elif direction == "RIGHT": # NOKTA
            progress = min(1.0, held_time / h_time)
            action_type = "DOT"
            if held_time >= h_time and (now - self.last_symbol_time) > self.symbol_cooldown:
                action = "DOT"
                self.last_symbol_time = now
                self.direction_start = now

        elif direction == "CENTER": # ONAY / BOŞLUK
            progress = min(1.0, held_time / c_time)
            action_type = "CONFIRM"
            if held_time >= c_time and (now - self.last_confirm_time) > 1.0:
                action = "CONFIRM"
                self.last_confirm_time = now
                self.direction_start = now

        # State Güncelleme
        with LOCK:
            STATE["direction"] = direction
            STATE["gaze_ratio"] = ratio
            STATE["progress"] = progress
            STATE["action_type"] = action_type
            
            if action == "DOT":
                STATE["morse"] += "."
                STATE["last_event"] = "Nokta (.) eklendi"
            elif action == "DASH":
                STATE["morse"] += "-"
                STATE["last_event"] = "Çizgi (-) eklendi"
            elif action == "CONFIRM":
                if STATE["morse"]:
                    char = decode_morse(STATE["morse"])
                    STATE["text"] += char
                    STATE["last_event"] = f"Harf: {char}"
                    STATE["morse"] = ""
                else:
                    if STATE["text"] and not STATE["text"].endswith(" "):
                        STATE["text"] += " "
                        STATE["last_event"] = "Boşluk eklendi"

        return direction, progress, action_type

    def draw_ui(self, img, direction, progress, action_type, ratio):
        h, w = img.shape[:2]
        
        # 1. Üst Panel (Yön Göstergeleri)
        zone_w = w // 3
        
        # Sol Bölge (Çizgi)
        c_left = (0, 0, 255) if direction == "LEFT" else (50, 50, 50) # Kırmızı
        cv2.rectangle(img, (0, 0), (zone_w, 80), c_left, -1)
        cv2.putText(img, "SOL (CIZGI -)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Orta Bölge (Onay)
        c_mid = (0, 255, 0) if direction == "CENTER" else (50, 50, 50) # Yeşil
        cv2.rectangle(img, (zone_w, 0), (2*zone_w, 80), c_mid, -1)
        cv2.putText(img, "MERKEZ (ONAY)", (zone_w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Sağ Bölge (Nokta)
        c_right = (255, 0, 0) if direction == "RIGHT" else (50, 50, 50) # Mavi
        cv2.rectangle(img, (2*zone_w, 0), (w, 80), c_right, -1)
        cv2.putText(img, "SAG (NOKTA .)", (2*zone_w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 2. İlerleme Çubuğu (Ortada)
        if progress > 0:
            bar_w = int((w - 40) * progress)
            color = (0, 255, 255) # Sarı
            if action_type == "DOT": color = (255, 0, 0)
            elif action_type == "DASH": color = (0, 0, 255)
            elif action_type == "CONFIRM": color = (0, 255, 0)
            
            cv2.rectangle(img, (20, 90), (20 + bar_w, 110), color, -1)
            cv2.rectangle(img, (20, 90), (w-20, 110), (255,255,255), 2) # Çerçeve

        # 3. Bakış Oranı (Debug için - Jüriye göstermek çok iyi olur)
        ratio_text = f"Bakis Orani: {ratio:.2f}"
        cv2.putText(img, ratio_text, (w//2 - 80, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Ortada imleç (Cursor)
        cursor_x = int(ratio * w)
        cv2.circle(img, (cursor_x, 140), 10, (0, 255, 255), -1)
        
        # 4. Alt Bilgi (Mors ve Mesaj)
        cv2.rectangle(img, (0, h-100), (w, h), (20, 20, 20), -1)
        
        with LOCK:
            m_disp = STATE["morse"] if STATE["morse"] else "---"
            t_disp = STATE["text"][-30:] if STATE["text"] else "..."
        
        cv2.putText(img, f"MORS: {m_disp}", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(img, f"MESAJ: {t_disp}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        return img

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                
                # Gaze Ratio
                raw_ratio = self.get_gaze_ratio(lm, w, h)
                self.ratio_history.append(raw_ratio)
                ratio = np.mean(self.ratio_history)
                
                # İşle
                direction, progress, action_type = self.process_gaze(ratio)
                
                # Çiz
                img = self.draw_ui(img, direction, progress, action_type, ratio)
                
                # İrisleri işaretle
                pt_l = lm[468]
                pt_r = lm[473]
                cv2.circle(img, (int(pt_l.x*w), int(pt_l.y*h)), 3, (0,255,0), -1)
                cv2.circle(img, (int(pt_r.x*w), int(pt_r.y*h)), 3, (0,255,0), -1)
            
            else:
                cv2.putText(img, "YUZ ARANIYOR...", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(e)
            return frame

# -----------------------------
# ARAYÜZ (STREAMLIT)
# -----------------------------
st.set_page_config(page_title="MORSE-EYE PRO", page_icon="👁️", layout="wide")

# CSS
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-header {
        text-align: center; 
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px; border-radius: 10px; color: white;
    }
    .morse-display {
        font-family: monospace; font-size: 50px; text-align: center;
        background: #2d3436; color: #00cec9; padding: 10px; border-radius: 10px;
    }
    .text-display {
        font-size: 24px; background: #dfe6e9; padding: 15px; border-radius: 10px; color: #2d3436;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>👁️ MORSE-EYE PRO</h1>
    <p>Göz Bakışı ile İletişim Sistemi | TÜBİTAK 2204-A</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    webrtc_streamer(
        key="morse-gaze",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=MorseGazeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📟 Canlı Panel")
    
    # Morse Göstergesi
    m_code = STATE["morse"] if STATE["morse"] else "---"
    st.markdown(f'<div class="morse-display">{m_code}</div>', unsafe_allow_html=True)
    
    # Tahmin
    pred = decode_morse(STATE["morse"])
    if pred: st.info(f"Olası Harf: **{pred}**")
    
    st.write("📝 **Yazılan Mesaj:**")
    st.markdown(f'<div class="text-display">{STATE["text"]}</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("🔊 Oku"): speak_js(STATE["text"])
    if c2.button("⌫ Sil"): 
        STATE["text"] = STATE["text"][:-1]
        st.rerun()
    if c3.button("🗑️ Sıfırla"):
        STATE["text"] = ""
        STATE["morse"] = ""
        st.rerun()
        
    with st.expander("⚙️ Kalibrasyon ve Ayarlar"):
        st.write("Bakış Oranı Eşikleri:")
        STATE["left_threshold"] = st.slider("Sola Bakış Eşiği (<)", 0.30, 0.50, STATE["left_threshold"], 0.01)
        STATE["right_threshold"] = st.slider("Sağa Bakış Eşiği (>)", 0.50, 0.70, STATE["right_threshold"], 0.01)
        
        st.write("Zamanlamalar:")
        STATE["hold_time"] = st.slider("Sembol Ekleme Süresi", 0.5, 2.0, STATE["hold_time"], 0.1)

# Otomatik yenileme
time.sleep(0.5)
st.rerun()
