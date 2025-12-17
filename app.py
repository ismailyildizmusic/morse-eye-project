# MORSE-HAND v2 — El Hareketleri ile Mors Kodu
# TÜBİTAK 2204-A Projesi
# Daha yavaş ve görünür mesaj paneli

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

# -----------------------------
# KOMUT TANIMLARI
# -----------------------------
COMMANDS = {
    0: {"name": "YUMRUK", "action": "CONFIRM", "desc": "Harfi Onayla", "color": (0, 255, 0), "emoji": "✊"},
    1: {"name": "1 PARMAK", "action": "DOT", "desc": "Nokta (.)", "color": (255, 165, 0), "emoji": "☝️"},
    2: {"name": "2 PARMAK", "action": "DASH", "desc": "Çizgi (-)", "color": (255, 0, 150), "emoji": "✌️"},
    3: {"name": "3 PARMAK", "action": "SPACE", "desc": "Boşluk", "color": (100, 150, 255), "emoji": "🤟"},
    4: {"name": "4 PARMAK", "action": "BACKSPACE", "desc": "Geri Sil", "color": (255, 100, 100), "emoji": "🖖"},
    5: {"name": "5 PARMAK", "action": "CLEAR", "desc": "Temizle", "color": (200, 200, 200), "emoji": "🖐️"},
}

# -----------------------------
# GLOBAL DURUM
# -----------------------------
if "hand_state" not in st.session_state:
    st.session_state.hand_state = {
        "morse": "",
        "text": "",
        "last_event": "",
        "finger_count": -1,
        "gesture_name": "Bekleniyor...",
        "progress": 0.0,
        "active": True,
        "last_action_time": 0,
        # YAVAŞLATILMIŞ SÜRELER
        "hold_time": 1.2,      # Komut için bekleme (artırıldı)
        "cooldown": 0.8,       # Komutlar arası bekleme (artırıldı)
        "confirm_time": 1.5,   # Onaylama için ekstra süre
    }

STATE = st.session_state.hand_state
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
        msg.rate = 0.8;
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# -----------------------------
# VIDEO İŞLEMCİ
# -----------------------------
class MorseHandProcessor(VideoProcessorBase):
    def __init__(self):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Parmak ucu ve eklem indeksleri
        self.FINGER_TIPS = [4, 8, 12, 16, 20]
        self.FINGER_PIPS = [3, 6, 10, 14, 18]
        
        # Zamanlama
        self.gesture_start = 0
        self.current_fingers = -1
        self.last_action_time = 0
        self.finger_history = deque(maxlen=7)  # Daha fazla stabilite
        self.action_executed = False  # Aynı jestte tekrar çalışmasın
        
    def count_fingers(self, hand_landmarks, handedness):
        """Açık parmak sayısını hesapla"""
        landmarks = hand_landmarks.landmark
        fingers_up = []
        
        is_right = handedness.classification[0].label == "Right"
        
        # Başparmak
        thumb_tip = landmarks[self.FINGER_TIPS[0]]
        thumb_ip = landmarks[self.FINGER_PIPS[0]]
        
        if is_right:
            fingers_up.append(thumb_tip.x < thumb_ip.x)
        else:
            fingers_up.append(thumb_tip.x > thumb_ip.x)
        
        # Diğer 4 parmak
        for i in range(1, 5):
            tip = landmarks[self.FINGER_TIPS[i]]
            pip = landmarks[self.FINGER_PIPS[i]]
            fingers_up.append(tip.y < pip.y)
        
        return sum(fingers_up)
    
    def process_gesture(self, finger_count):
        """Parmak sayısına göre komut işle - YAVAŞLATILMIŞ"""
        now = time.time()
        
        with LOCK:
            hold_time = STATE["hold_time"]
            cooldown = STATE["cooldown"]
            confirm_time = STATE["confirm_time"]
        
        # Stabilite için geçmiş değerlere bak
        self.finger_history.append(finger_count)
        
        if len(self.finger_history) >= 5:
            # En sık görülen değer
            counts = {}
            for f in self.finger_history:
                counts[f] = counts.get(f, 0) + 1
            stable_count = max(counts, key=counts.get)
            
            # En az 4 kez görülmüş olmalı (daha kararlı)
            if counts[stable_count] < 4:
                stable_count = -1
        else:
            stable_count = -1
        
        # Parmak sayısı değişti mi?
        if stable_count != self.current_fingers:
            self.current_fingers = stable_count
            self.gesture_start = now
            self.action_executed = False  # Yeni jest, aksiyon sıfırla
            with LOCK:
                STATE["progress"] = 0.0
            return None, 0.0
        
        # Geçersiz jest
        if stable_count == -1:
            return None, 0.0
        
        # Ne kadar süredir bu jest?
        held_time = now - self.gesture_start
        
        # Onaylama (yumruk) için daha uzun süre
        required_time = confirm_time if stable_count == 0 else hold_time
        
        progress = min(1.0, held_time / required_time)
        
        # Cooldown kontrolü
        if (now - self.last_action_time) < cooldown:
            with LOCK:
                STATE["progress"] = progress
                STATE["finger_count"] = stable_count
                if stable_count in COMMANDS:
                    STATE["gesture_name"] = COMMANDS[stable_count]["name"]
            return None, progress
        
        # Süre doldu ve aksiyon henüz çalışmadı mı?
        action = None
        if held_time >= required_time and not self.action_executed:
            if stable_count in COMMANDS:
                action = COMMANDS[stable_count]["action"]
                self.last_action_time = now
                self.action_executed = True  # Bu jestte bir kez çalıştı
                
                # Aksiyonu uygula
                with LOCK:
                    STATE["last_action_time"] = now
                    
                    if action == "DOT":
                        STATE["morse"] += "."
                        STATE["last_event"] = "✓ Nokta (.) eklendi"
                    elif action == "DASH":
                        STATE["morse"] += "-"
                        STATE["last_event"] = "✓ Çizgi (-) eklendi"
                    elif action == "CONFIRM":
                        if STATE["morse"]:
                            char = decode_morse(STATE["morse"])
                            if char and char != "?":
                                STATE["text"] += char
                                STATE["last_event"] = f"✓ {STATE['morse']} → {char}"
                            else:
                                STATE["last_event"] = f"✗ Geçersiz: {STATE['morse']}"
                            STATE["morse"] = ""
                        else:
                            STATE["last_event"] = "ℹ Morse boş"
                    elif action == "SPACE":
                        if STATE["text"] and not STATE["text"].endswith(" "):
                            STATE["text"] += " "
                            STATE["last_event"] = "✓ Boşluk eklendi"
                    elif action == "BACKSPACE":
                        if STATE["morse"]:
                            STATE["morse"] = STATE["morse"][:-1]
                            STATE["last_event"] = "✓ Sembol silindi"
                        elif STATE["text"]:
                            STATE["text"] = STATE["text"][:-1]
                            STATE["last_event"] = "✓ Harf silindi"
                    elif action == "CLEAR":
                        STATE["morse"] = ""
                        STATE["text"] = ""
                        STATE["last_event"] = "✓ Temizlendi"
        
        with LOCK:
            STATE["finger_count"] = stable_count
            STATE["progress"] = progress
            if stable_count in COMMANDS:
                STATE["gesture_name"] = COMMANDS[stable_count]["name"]
        
        return action, progress

    def draw_ui(self, img, finger_count, progress, hand_landmarks=None):
        """Basitleştirilmiş ekran arayüzü"""
        h, w = img.shape[:2]
        
        with LOCK:
            morse = STATE["morse"]
        
        # === ÜST PANEL - Sadece aktif komut ===
        cv2.rectangle(img, (0, 0), (w, 80), (20, 20, 20), -1)
        
        if finger_count in COMMANDS:
            cmd = COMMANDS[finger_count]
            
            # İlerleme çubuğu
            bar_width = int(w * progress)
            cv2.rectangle(img, (0, 0), (bar_width, 80), cmd["color"], -1)
            
            # Yarı saydam overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (bar_width, 80), cmd["color"], -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            
            # Komut bilgisi
            text = f"{cmd['emoji']} {cmd['desc']} - %{int(progress*100)}"
            cv2.putText(img, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.putText(img, "Elini goster...", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        
        # === ORTA - Büyük Morse Kodu ===
        morse_display = morse if morse else ""
        if morse_display:
            # Arka plan kutusu
            box_w = min(len(morse_display) * 60 + 40, w - 40)
            box_x = (w - box_w) // 2
            cv2.rectangle(img, (box_x, h//2 - 50), (box_x + box_w, h//2 + 50), (0, 0, 0), -1)
            cv2.rectangle(img, (box_x, h//2 - 50), (box_x + box_w, h//2 + 50), (0, 255, 255), 3)
            
            # Morse kodu
            font_size = 2.5 if len(morse_display) < 6 else 1.8
            text_size = cv2.getTextSize(morse_display, cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(img, morse_display, (text_x, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 4)
            
            # Tahmin edilen harf
            predicted = decode_morse(morse)
            if predicted and predicted != "?":
                cv2.putText(img, f"= {predicted}", (box_x + box_w + 10, h//2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # === El çizimi ===
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                img, hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Parmak sayısı göster (el üzerinde)
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h) - 60
            if 0 < cy < h and 0 < cx < w:
                cv2.circle(img, (cx, cy), 35, (0, 0, 0), -1)
                cv2.circle(img, (cx, cy), 35, (0, 255, 255), 3)
                cv2.putText(img, str(finger_count), (cx - 12, cy + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        return img

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            finger_count = -1
            progress = 0.0
            hand_landmarks = None
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                finger_count = self.count_fingers(hand_landmarks, handedness)
                action, progress = self.process_gesture(finger_count)
            else:
                self.current_fingers = -1
                self.finger_history.clear()
                self.action_executed = False
                with LOCK:
                    STATE["finger_count"] = -1
                    STATE["gesture_name"] = "El yok"
                    STATE["progress"] = 0.0
            
            img = self.draw_ui(img, finger_count, progress, hand_landmarks)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            return frame


# -----------------------------
# STREAMLIT ARAYÜZÜ
# -----------------------------
st.set_page_config(page_title="MORSE-HAND", page_icon="✋", layout="wide")

# CSS
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    .big-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 15px;
        color: white;
        margin-bottom: 15px;
    }
    
    .morse-display {
        font-family: 'Courier New', monospace;
        font-size: 72px;
        font-weight: bold;
        color: #00ffcc;
        text-align: center;
        background: #0a0a0a;
        padding: 30px;
        border-radius: 20px;
        border: 4px solid #00ffcc;
        margin: 15px 0;
        min-height: 120px;
        text-shadow: 0 0 30px #00ffcc;
        letter-spacing: 15px;
    }
    
    .message-display {
        font-size: 36px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 30px;
        border-radius: 20px;
        border: 3px solid #4a90d9;
        min-height: 100px;
        margin: 15px 0;
        word-wrap: break-word;
    }
    
    .predicted-char {
        font-size: 48px;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        padding: 10px;
    }
    
    .command-btn {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 3px;
    }
    
    .status-box {
        background: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 30px;
        background: #333;
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00ccff);
        transition: width 0.1s;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("""
<div class="big-title">
    <h1 style="margin:0;">✋ MORSE-HAND</h1>
    <p style="margin:5px 0 0 0;">El Hareketleri ile Mors Kodu | TÜBİTAK 2204-A</p>
</div>
""", unsafe_allow_html=True)

# Ana düzen - Video solda, panel sağda
col_video, col_panel = st.columns([1.2, 1])

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="morse-hand-v2",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        video_processor_factory=MorseHandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_panel:
    # === MORS KODU (BÜYÜK) ===
    st.markdown("### 📟 MORS KODU")
    morse_val = STATE["morse"] if STATE["morse"] else "---"
    st.markdown(f'<div class="morse-display">{morse_val}</div>', unsafe_allow_html=True)
    
    # Tahmin edilen harf
    if STATE["morse"]:
        predicted = decode_morse(STATE["morse"])
        if predicted and predicted != "?":
            st.markdown(f'<div class="predicted-char">Harf: {predicted}</div>', unsafe_allow_html=True)
        else:
            st.warning("❓ Geçersiz mors kodu")
    
    # === MESAJ (BÜYÜK) ===
    st.markdown("### 💬 MESAJ")
    msg_val = STATE["text"] if STATE["text"] else "(Mesajınız burada görünecek)"
    st.markdown(f'<div class="message-display">{msg_val}</div>', unsafe_allow_html=True)
    
    # === İLERLEME ÇUBUĞU ===
    progress_pct = int(STATE["progress"] * 100)
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Durum bilgisi
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        gesture = STATE["gesture_name"]
        st.info(f"🤚 {gesture}")
    with col_s2:
        st.info(f"📊 %{progress_pct}")
    
    # Son olay
    if STATE["last_event"]:
        if "✓" in STATE["last_event"]:
            st.success(STATE["last_event"])
        elif "✗" in STATE["last_event"]:
            st.error(STATE["last_event"])
        else:
            st.info(STATE["last_event"])
    
    # === BUTONLAR ===
    st.markdown("### 🎮 Kontroller")
    btn1, btn2, btn3 = st.columns(3)
    
    with btn1:
        if st.button("🗑️ TEMİZLE", use_container_width=True):
            STATE["text"] = ""
            STATE["morse"] = ""
            STATE["last_event"] = "✓ Temizlendi"
            st.rerun()
    
    with btn2:
        if st.button("↩️ GERİ", use_container_width=True):
            if STATE["morse"]:
                STATE["morse"] = STATE["morse"][:-1]
            elif STATE["text"]:
                STATE["text"] = STATE["text"][:-1]
            STATE["last_event"] = "✓ Silindi"
            st.rerun()
    
    with btn3:
        if st.button("🔊 OKU", use_container_width=True):
            if STATE["text"]:
                speak_js(STATE["text"])
                st.success("Okunuyor...")
    
    # Yenile butonu
    if st.button("🔄 PANELİ YENİLE", use_container_width=True, type="primary"):
        st.rerun()

# === KOMUT REHBERİ (ALT KISIM) ===
st.markdown("---")
st.markdown("### 📖 KOMUT REHBERİ")

cmd_cols = st.columns(6)
for i, (count, cmd) in enumerate(COMMANDS.items()):
    with cmd_cols[i]:
        st.markdown(f"""
        <div class="command-btn">
            <div style="font-size: 32px;">{cmd['emoji']}</div>
            <div style="font-size: 14px; font-weight: bold;">{cmd['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

# Ayarlar
with st.expander("⚙️ Hız Ayarları"):
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        STATE["hold_time"] = st.slider(
            "Komut süresi (saniye)", 
            0.5, 2.5, STATE["hold_time"], 0.1,
            help="Parmağı bu kadar süre tutunca komut çalışır"
        )
    with col_set2:
        STATE["cooldown"] = st.slider(
            "Bekleme süresi (saniye)", 
            0.3, 1.5, STATE["cooldown"], 0.1,
            help="Komutlar arası minimum bekleme"
        )
    
    STATE["confirm_time"] = st.slider(
        "Onaylama süresi (yumruk)", 
        1.0, 3.0, STATE["confirm_time"], 0.1,
        help="Harfi onaylamak için yumruğu tutma süresi"
    )

# Morse tablosu
with st.expander("📖 Morse Alfabesi Tablosu"):
    morse_cols = st.columns(4)
    letters = list(MORSE_TO_CHAR.items())[:26]  # Sadece harfler
    per_col = 7
    
    for i, col in enumerate(morse_cols):
        with col:
            start = i * per_col
            end = start + per_col
            for code, char in letters[start:end]:
                st.write(f"**{char}** → `{code}`")

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 10px;">
    ✋ <b>MORSE-HAND v2</b> | TÜBİTAK 2204-A | El hareketleri ile iletişim sistemi
</div>
""", unsafe_allow_html=True)

# Otomatik yenileme (video çalışırken)
if webrtc_ctx.state.playing:
    time.sleep(1)
    st.rerun()
