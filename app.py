# MORSE-EYE PRO — Göz Bakışı ile Mors Kodu
# TÜBİTAK 2204-A Projesi
# Sağa bak = Nokta (.) | Sola bak = Çizgi (-) | Ortaya bak = Onayla

import time
import threading
from collections import deque
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import mediapipe as mp
import av

# -----------------------------
# MORSE SÖZLÜĞÜ
# -----------------------------
MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", 
    "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7", 
    "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?"
}

# Ters sözlük (harf -> morse)
CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}

# -----------------------------
# GLOBAL DURUM
# -----------------------------
if "morse_state" not in st.session_state:
    st.session_state.morse_state = {
        "morse": "",
        "text": "",
        "last_event": "",
        "direction": "CENTER",
        "gaze_ratio": 0.5,
        "progress": 0.0,  # 0-1 arası ilerleme çubuğu
        "action_type": "",  # DOT, DASH, CONFIRM, SPACE
        # Ayarlar
        "hold_time": 0.8,  # Sembol eklemek için bekleme süresi
        "confirm_time": 1.5,  # Harf onaylamak için ortada bekleme
        "left_threshold": 0.42,
        "right_threshold": 0.58,
    }

STATE = st.session_state.morse_state
LOCK = threading.Lock()

# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def decode_morse(code):
    return MORSE_TO_CHAR.get(code, "?") if code else ""

def speak_js(text, lang="tr-TR"):
    """Tarayıcıda sesli okuma"""
    safe = text.replace('"', '\\"').replace("'", "\\'")
    components.html(f"""
    <script>
        window.speechSynthesis.cancel();
        let msg = new SpeechSynthesisUtterance("{safe}");
        msg.lang = "{lang}";
        msg.rate = 0.9;
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# -----------------------------
# VIDEO İŞLEMCİ
# -----------------------------
class MorseGazeProcessor(VideoProcessorBase):
    def __init__(self):
        # MediaPipe yüz mesh
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Göz landmark indeksleri
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [473, 474, 475, 476]
        
        # Smoothing
        self.ratio_history = deque(maxlen=5)
        
        # Zamanlama
        self.direction_start = 0
        self.current_dir = "CENTER"
        self.last_symbol_time = 0
        self.last_confirm_time = 0
        self.symbol_cooldown = 0.3  # Semboller arası minimum süre
        
    def get_gaze_ratio(self, landmarks, w, h):
        """İris pozisyonundan bakış yönü hesapla (0=sol, 1=sağ)"""
        try:
            # Sol göz
            left_left = landmarks[self.LEFT_EYE_LEFT]
            left_right = landmarks[self.LEFT_EYE_RIGHT]
            left_iris_pts = [landmarks[i] for i in self.LEFT_IRIS]
            left_iris_x = np.mean([p.x for p in left_iris_pts])
            
            left_eye_width = abs(left_right.x - left_left.x)
            if left_eye_width > 0.001:
                left_ratio = (left_iris_x - left_left.x) / left_eye_width
            else:
                left_ratio = 0.5
            
            # Sağ göz
            right_left = landmarks[self.RIGHT_EYE_LEFT]
            right_right = landmarks[self.RIGHT_EYE_RIGHT]
            right_iris_pts = [landmarks[i] for i in self.RIGHT_IRIS]
            right_iris_x = np.mean([p.x for p in right_iris_pts])
            
            right_eye_width = abs(right_right.x - right_left.x)
            if right_eye_width > 0.001:
                right_ratio = (right_iris_x - right_left.x) / right_eye_width
            else:
                right_ratio = 0.5
            
            # Ortalama
            ratio = (left_ratio + right_ratio) / 2.0
            return max(0.0, min(1.0, ratio))
            
        except:
            return 0.5
    
    def process_gaze(self, ratio):
        """Bakış yönüne göre sembol/onay işle"""
        now = time.time()
        
        with LOCK:
            left_thr = STATE["left_threshold"]
            right_thr = STATE["right_threshold"]
            hold_time = STATE["hold_time"]
            confirm_time = STATE["confirm_time"]
        
        # Yön belirle
        if ratio < left_thr:
            direction = "LEFT"
        elif ratio > right_thr:
            direction = "RIGHT"
        else:
            direction = "CENTER"
        
        # Yön değişti mi?
        if direction != self.current_dir:
            self.current_dir = direction
            self.direction_start = now
            with LOCK:
                STATE["progress"] = 0.0
                STATE["action_type"] = ""
        
        # Ne kadar süredir bu yönde?
        held_time = now - self.direction_start
        
        action = None
        progress = 0.0
        action_type = ""
        
        if direction == "LEFT":
            # Çizgi (-) ekleme
            progress = min(1.0, held_time / hold_time)
            action_type = "DASH"
            if held_time >= hold_time and (now - self.last_symbol_time) > self.symbol_cooldown:
                action = "DASH"
                self.last_symbol_time = now
                self.direction_start = now  # Sıfırla (sürekli ekleme için)
                
        elif direction == "RIGHT":
            # Nokta (.) ekleme
            progress = min(1.0, held_time / hold_time)
            action_type = "DOT"
            if held_time >= hold_time and (now - self.last_symbol_time) > self.symbol_cooldown:
                action = "DOT"
                self.last_symbol_time = now
                self.direction_start = now
                
        elif direction == "CENTER":
            # Harf onaylama
            progress = min(1.0, held_time / confirm_time)
            action_type = "CONFIRM"
            if held_time >= confirm_time and (now - self.last_confirm_time) > 1.0:
                action = "CONFIRM"
                self.last_confirm_time = now
                self.direction_start = now
        
        # State güncelle
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
                    if char and char != "?":
                        STATE["text"] += char
                        STATE["last_event"] = f"Harf onaylandı: {STATE['morse']} → {char}"
                    else:
                        STATE["last_event"] = f"Geçersiz kod: {STATE['morse']}"
                    STATE["morse"] = ""
                else:
                    # Morse boşsa boşluk ekle
                    if STATE["text"] and not STATE["text"].endswith(" "):
                        STATE["text"] += " "
                        STATE["last_event"] = "Boşluk eklendi"
        
        return direction, progress, action_type, action

    def draw_ui(self, img, direction, progress, action_type, ratio):
        """Ekran üzerine görsel arayüz çiz"""
        h, w = img.shape[:2]
        
        with LOCK:
            morse = STATE["morse"]
            text = STATE["text"]
            last_event = STATE["last_event"]
        
        # Arka plan paneli (üst)
        cv2.rectangle(img, (0, 0), (w, 140), (40, 40, 40), -1)
        
        # Yön göstergesi (3 bölge)
        zone_width = w // 3
        
        # Sol bölge (Çizgi)
        left_color = (0, 100, 255) if direction == "LEFT" else (80, 80, 80)
        cv2.rectangle(img, (0, 0), (zone_width, 60), left_color, -1)
        cv2.putText(img, "< CIZGI (-)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Orta bölge (Onay)
        center_color = (0, 200, 0) if direction == "CENTER" else (80, 80, 80)
        cv2.rectangle(img, (zone_width, 0), (2*zone_width, 60), center_color, -1)
        cv2.putText(img, "ONAYLA", (zone_width + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Sağ bölge (Nokta)
        right_color = (255, 100, 0) if direction == "RIGHT" else (80, 80, 80)
        cv2.rectangle(img, (2*zone_width, 0), (w, 60), right_color, -1)
        cv2.putText(img, "NOKTA (.) >", (2*zone_width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # İlerleme çubuğu
        bar_y = 70
        bar_height = 25
        cv2.rectangle(img, (10, bar_y), (w-10, bar_y + bar_height), (60, 60, 60), -1)
        
        if progress > 0:
            bar_width = int((w - 20) * progress)
            if action_type == "DOT":
                bar_color = (255, 100, 0)  # Turuncu
            elif action_type == "DASH":
                bar_color = (0, 100, 255)  # Kırmızı-mavi
            else:
                bar_color = (0, 200, 0)  # Yeşil
            cv2.rectangle(img, (10, bar_y), (10 + bar_width, bar_y + bar_height), bar_color, -1)
        
        # Yüzde göster
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(img, pct_text, (w//2 - 30, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Morse kodu (büyük, ortada)
        morse_display = morse if morse else "---"
        morse_size = 1.5 if len(morse_display) < 6 else 1.0
        text_size = cv2.getTextSize(morse_display, cv2.FONT_HERSHEY_SIMPLEX, morse_size, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(img, morse_display, (text_x, 125), cv2.FONT_HERSHEY_SIMPLEX, morse_size, (0, 255, 255), 3)
        
        # Anlık harf çözümü
        if morse:
            predicted = decode_morse(morse)
            if predicted and predicted != "?":
                cv2.putText(img, f"= {predicted}", (text_x + text_size[0] + 10, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Alt panel - Mesaj
        cv2.rectangle(img, (0, h-70), (w, h), (30, 30, 30), -1)
        
        # Mesaj metni
        display_text = text[-40:] if len(text) > 40 else text
        if not display_text:
            display_text = "(Mesaj burada gorunecek)"
        cv2.putText(img, display_text, (15, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Son olay
        if last_event:
            cv2.putText(img, last_event, (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Bakış noktası göstergesi (ortada küçük bir çizgi)
        gaze_x = int(w * ratio)
        cv2.line(img, (gaze_x, 145), (gaze_x, 165), (0, 255, 255), 3)
        cv2.circle(img, (gaze_x, 155), 8, (0, 255, 255), -1)
        
        return img

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Ayna görüntüsü
            h, w = img.shape[:2]
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Bakış oranı hesapla
                raw_ratio = self.get_gaze_ratio(landmarks, w, h)
                self.ratio_history.append(raw_ratio)
                ratio = np.mean(self.ratio_history)
                
                # İşle
                direction, progress, action_type, action = self.process_gaze(ratio)
                
                # Çiz
                img = self.draw_ui(img, direction, progress, action_type, ratio)
                
                # İris noktalarını göster (debug)
                for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                    pt = landmarks[idx]
                    x, y = int(pt.x * w), int(pt.y * h)
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            else:
                # Yüz bulunamadı
                cv2.rectangle(img, (w//2-150, h//2-30), (w//2+150, h//2+30), (0, 0, 200), -1)
                cv2.putText(img, "YUZ BULUNAMADI", (w//2-130, h//2+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # Hata durumunda orijinal frame'i döndür
            return frame


# -----------------------------
# STREAMLIT ARAYÜZÜ
# -----------------------------
st.set_page_config(page_title="MORSE-EYE PRO", page_icon="👁️", layout="wide")

# CSS
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    
    .morse-box {
        font-family: 'Courier New', monospace;
        font-size: 48px;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        background: #1a1a2e;
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #00ff88;
        margin: 10px 0;
        min-height: 80px;
    }
    
    .text-box {
        font-size: 28px;
        color: #ffffff;
        background: #2d3436;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #636e72;
        min-height: 100px;
        margin: 10px 0;
    }
    
    .guide-box {
        background: #0a3d62;
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    
    .guide-item {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #1e5f74;
    }
    
    .guide-icon {
        font-size: 24px;
        margin-right: 15px;
        width: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("""
<div class="main-title">
    <h1>👁️ MORSE-EYE PRO</h1>
    <p>Göz Hareketleri ile İletişim Sistemi | TÜBİTAK 2204-A</p>
</div>
""", unsafe_allow_html=True)

# Ana düzen
col_video, col_panel = st.columns([1.5, 1])

with col_video:
    # WebRTC kamera
    webrtc_streamer(
        key="morse-gaze",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        video_processor_factory=MorseGazeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Kullanım rehberi
    st.markdown("""
    <div class="guide-box">
        <h4>📖 KULLANIM REHBERİ</h4>
        <div class="guide-item">
            <span class="guide-icon">👉</span>
            <span><b>SAĞA BAK</b> (0.8 sn) → Nokta (.) ekler</span>
        </div>
        <div class="guide-item">
            <span class="guide-icon">👈</span>
            <span><b>SOLA BAK</b> (0.8 sn) → Çizgi (-) ekler</span>
        </div>
        <div class="guide-item">
            <span class="guide-icon">🎯</span>
            <span><b>ORTAYA BAK</b> (1.5 sn) → Harfi onaylar</span>
        </div>
        <div class="guide-item">
            <span class="guide-icon">⏸️</span>
            <span><b>ORTAYA BAK</b> (morse boşken) → Boşluk ekler</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_panel:
    st.subheader("📟 Mors Kodu")
    morse_display = STATE["morse"] if STATE["morse"] else "---"
    st.markdown(f'<div class="morse-box">{morse_display}</div>', unsafe_allow_html=True)
    
    # Anlık çözüm
    if STATE["morse"]:
        predicted = decode_morse(STATE["morse"])
        st.info(f"🔤 Tahmin edilen harf: **{predicted}**")
    
    st.subheader("💬 Mesaj")
    text_display = STATE["text"] if STATE["text"] else "(Henüz mesaj yok)"
    st.markdown(f'<div class="text-box">{text_display}</div>', unsafe_allow_html=True)
    
    # Kontrol butonları
    st.subheader("🎮 Kontroller")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🗑️ Temizle", use_container_width=True):
            STATE["text"] = ""
            STATE["morse"] = ""
            STATE["last_event"] = "Temizlendi"
            st.rerun()
    
    with btn_col2:
        if st.button("↩️ Geri Sil", use_container_width=True):
            if STATE["morse"]:
                STATE["morse"] = STATE["morse"][:-1]
            elif STATE["text"]:
                STATE["text"] = STATE["text"][:-1]
            st.rerun()
    
    with btn_col3:
        if st.button("🔊 Oku", use_container_width=True):
            if STATE["text"]:
                speak_js(STATE["text"])
    
    # Ayarlar
    with st.expander("⚙️ Ayarlar"):
        STATE["hold_time"] = st.slider(
            "Sembol ekleme süresi (saniye)", 
            0.3, 2.0, STATE["hold_time"], 0.1,
            help="Sağa/sola bu kadar süre bakınca sembol eklenir"
        )
        
        STATE["confirm_time"] = st.slider(
            "Onaylama süresi (saniye)", 
            0.5, 3.0, STATE["confirm_time"], 0.1,
            help="Ortaya bu kadar süre bakınca harf onaylanır"
        )
        
        st.write("**Bakış Eşikleri:**")
        col_l, col_r = st.columns(2)
        with col_l:
            STATE["left_threshold"] = st.slider("Sol eşik", 0.30, 0.50, STATE["left_threshold"], 0.01)
        with col_r:
            STATE["right_threshold"] = st.slider("Sağ eşik", 0.50, 0.70, STATE["right_threshold"], 0.01)
    
    # Morse tablosu
    with st.expander("📖 Morse Alfabesi"):
        morse_table = """
        | Harf | Kod | Harf | Kod | Harf | Kod |
        |------|-----|------|-----|------|-----|
        | A | .- | J | .--- | S | ... |
        | B | -... | K | -.- | T | - |
        | C | -.-. | L | .-.. | U | ..- |
        | D | -.. | M | -- | V | ...- |
        | E | . | N | -. | W | .-- |
        | F | ..-. | O | --- | X | -..- |
        | G | --. | P | .--. | Y | -.-- |
        | H | .... | Q | --.- | Z | --.. |
        | I | .. | R | .-. | | |
        """
        st.markdown(morse_table)

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>👁️ MORSE-EYE PRO | TÜBİTAK 2204-A Projesi</p>
    <p>Hareket kısıtlılığı olan bireyler için göz takibi ile iletişim sistemi</p>
</div>
""", unsafe_allow_html=True)
