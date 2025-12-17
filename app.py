# MORSE-HAND — El Hareketleri ile Mors Kodu
# TÜBİTAK 2204-A Projesi
# Parmak sayısına göre komut sistemi

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
    0: {"name": "YUMRUK", "action": "CONFIRM", "desc": "Harfi Onayla", "color": (0, 255, 0)},
    1: {"name": "1 PARMAK", "action": "DOT", "desc": "Nokta (.)", "color": (255, 165, 0)},
    2: {"name": "2 PARMAK", "action": "DASH", "desc": "Çizgi (-)", "color": (255, 0, 100)},
    3: {"name": "3 PARMAK", "action": "SPACE", "desc": "Boşluk", "color": (100, 100, 255)},
    4: {"name": "4 PARMAK", "action": "BACKSPACE", "desc": "Son Sembolü Sil", "color": (255, 100, 100)},
    5: {"name": "5 PARMAK", "action": "CLEAR", "desc": "Tümünü Temizle", "color": (200, 200, 200)},
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
        # Ayarlar
        "hold_time": 0.6,  # Komut için bekleme süresi
        "cooldown": 0.4,   # Komutlar arası bekleme
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
        msg.rate = 0.9;
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
        
        # Parmak ucu indeksleri
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # Başparmak, işaret, orta, yüzük, serçe
        self.FINGER_PIPS = [3, 6, 10, 14, 18]  # Parmak orta eklemleri
        
        # Zamanlama
        self.gesture_start = 0
        self.current_fingers = -1
        self.last_action_time = 0
        self.finger_history = deque(maxlen=5)  # Stabilite için
        
    def count_fingers(self, hand_landmarks, handedness):
        """Açık parmak sayısını hesapla"""
        landmarks = hand_landmarks.landmark
        fingers_up = []
        
        # Elin sağ mı sol mu olduğunu kontrol et
        is_right = handedness.classification[0].label == "Right"
        
        # Başparmak - yatay kontrol (elin yönüne göre)
        thumb_tip = landmarks[self.FINGER_TIPS[0]]
        thumb_ip = landmarks[self.FINGER_PIPS[0]]
        
        if is_right:
            # Sağ el: başparmak sola açılır (x azalır)
            fingers_up.append(thumb_tip.x < thumb_ip.x)
        else:
            # Sol el: başparmak sağa açılır (x artar)
            fingers_up.append(thumb_tip.x > thumb_ip.x)
        
        # Diğer 4 parmak - dikey kontrol (y azalırsa açık)
        for i in range(1, 5):
            tip = landmarks[self.FINGER_TIPS[i]]
            pip = landmarks[self.FINGER_PIPS[i]]
            fingers_up.append(tip.y < pip.y)
        
        return sum(fingers_up)
    
    def get_hand_center(self, hand_landmarks, w, h):
        """El merkezini hesapla"""
        landmarks = hand_landmarks.landmark
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        return int(np.mean(x_coords)), int(np.mean(y_coords))
    
    def process_gesture(self, finger_count):
        """Parmak sayısına göre komut işle"""
        now = time.time()
        
        with LOCK:
            hold_time = STATE["hold_time"]
            cooldown = STATE["cooldown"]
            active = STATE["active"]
        
        if not active:
            return None, 0.0
        
        # Stabilite için geçmiş değerlere bak
        self.finger_history.append(finger_count)
        
        # Son 5 okumada en sık görülen değer
        if len(self.finger_history) >= 3:
            stable_count = max(set(self.finger_history), key=list(self.finger_history).count)
        else:
            stable_count = finger_count
        
        # Parmak sayısı değişti mi?
        if stable_count != self.current_fingers:
            self.current_fingers = stable_count
            self.gesture_start = now
            with LOCK:
                STATE["progress"] = 0.0
            return None, 0.0
        
        # Ne kadar süredir bu jest?
        held_time = now - self.gesture_start
        progress = min(1.0, held_time / hold_time)
        
        # Cooldown kontrolü
        if (now - self.last_action_time) < cooldown:
            return None, progress
        
        # Süre doldu mu?
        action = None
        if held_time >= hold_time:
            if stable_count in COMMANDS:
                action = COMMANDS[stable_count]["action"]
                self.last_action_time = now
                self.gesture_start = now  # Sıfırla
                
                # Aksiyonu uygula
                with LOCK:
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
                                STATE["last_event"] = f"✓ Harf: {STATE['morse']} → {char}"
                            else:
                                STATE["last_event"] = f"✗ Geçersiz: {STATE['morse']}"
                            STATE["morse"] = ""
                        else:
                            STATE["last_event"] = "Morse tamponu boş"
                    elif action == "SPACE":
                        if STATE["text"] and not STATE["text"].endswith(" "):
                            STATE["text"] += " "
                            STATE["last_event"] = "✓ Boşluk eklendi"
                    elif action == "BACKSPACE":
                        if STATE["morse"]:
                            STATE["morse"] = STATE["morse"][:-1]
                            STATE["last_event"] = "✓ Son sembol silindi"
                        elif STATE["text"]:
                            STATE["text"] = STATE["text"][:-1]
                            STATE["last_event"] = "✓ Son harf silindi"
                    elif action == "CLEAR":
                        STATE["morse"] = ""
                        STATE["text"] = ""
                        STATE["last_event"] = "✓ Tümü temizlendi"
        
        with LOCK:
            STATE["finger_count"] = stable_count
            STATE["progress"] = progress
            if stable_count in COMMANDS:
                STATE["gesture_name"] = COMMANDS[stable_count]["name"]
        
        return action, progress

    def draw_ui(self, img, finger_count, progress, hand_landmarks=None):
        """Ekran üzerine görsel arayüz çiz"""
        h, w = img.shape[:2]
        
        with LOCK:
            morse = STATE["morse"]
            text = STATE["text"]
            last_event = STATE["last_event"]
            active = STATE["active"]
        
        # === ÜST PANEL - Komut Göstergeleri ===
        panel_height = 100
        cv2.rectangle(img, (0, 0), (w, panel_height), (30, 30, 30), -1)
        
        # 6 komut kutusu
        box_width = w // 6
        for i, (count, cmd) in enumerate(COMMANDS.items()):
            x1 = i * box_width
            x2 = x1 + box_width - 2
            
            # Aktif komut vurgusu
            if finger_count == count:
                # İlerleme çubuğu (alttan yukarı dolan)
                fill_height = int(60 * progress)
                cv2.rectangle(img, (x1, 60 - fill_height), (x2, 60), cmd["color"], -1)
                border_color = (255, 255, 255)
                text_color = (255, 255, 255)
            else:
                border_color = (80, 80, 80)
                text_color = (150, 150, 150)
            
            # Kutu çerçevesi
            cv2.rectangle(img, (x1 + 2, 5), (x2, 60), border_color, 2)
            
            # Parmak sayısı (büyük)
            num_text = "✊" if count == 0 else str(count)
            cv2.putText(img, num_text, (x1 + box_width//2 - 15, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
            
            # Komut açıklaması
            cv2.putText(img, cmd["desc"], (x1 + 5, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
        
        # === ORTA KISIM - Mors ve El ===
        
        # Morse kodu (büyük, görünür)
        morse_display = morse if morse else "---"
        morse_size = 2.0 if len(morse_display) < 8 else 1.5
        text_size = cv2.getTextSize(morse_display, cv2.FONT_HERSHEY_SIMPLEX, morse_size, 4)[0]
        text_x = (w - text_size[0]) // 2
        
        # Morse arka planı
        cv2.rectangle(img, (text_x - 20, 110), (text_x + text_size[0] + 20, 180), (0, 0, 0), -1)
        cv2.rectangle(img, (text_x - 20, 110), (text_x + text_size[0] + 20, 180), (0, 255, 255), 2)
        cv2.putText(img, morse_display, (text_x, 165), cv2.FONT_HERSHEY_SIMPLEX, morse_size, (0, 255, 255), 4)
        
        # Anlık harf çözümü
        if morse:
            predicted = decode_morse(morse)
            if predicted and predicted != "?":
                cv2.putText(img, f"= {predicted}", (text_x + text_size[0] + 30, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # El landmark'larını çiz (varsa)
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                img, hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # El merkezinde parmak sayısı göster
            cx, cy = self.get_hand_center(hand_landmarks, w, h)
            cv2.circle(img, (cx, cy), 40, COMMANDS.get(finger_count, {}).get("color", (255,255,255)), 3)
            cv2.putText(img, str(finger_count), (cx - 15, cy + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # === ALT PANEL - Mesaj ===
        cv2.rectangle(img, (0, h-80), (w, h), (20, 20, 20), -1)
        
        # Mesaj başlığı
        cv2.putText(img, "MESAJ:", (15, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Mesaj metni
        display_text = text[-50:] if len(text) > 50 else text
        if not display_text:
            display_text = "(El hareketleriyle mesaj yazın...)"
            text_color = (100, 100, 100)
        else:
            text_color = (255, 255, 255)
        cv2.putText(img, display_text, (15, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Son olay (sağ altta)
        if last_event:
            event_color = (100, 255, 100) if "✓" in last_event else (255, 100, 100)
            cv2.putText(img, last_event, (w - 350, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, event_color, 1)
        
        # Sistem durumu (sol üst köşe)
        status_text = "AKTIF" if active else "PASIF"
        status_color = (0, 255, 0) if active else (0, 0, 255)
        cv2.circle(img, (w - 30, 25), 10, status_color, -1)
        
        return img

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Ayna görüntüsü
            h, w = img.shape[:2]
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            finger_count = -1
            progress = 0.0
            hand_landmarks = None
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                
                # Parmak say
                finger_count = self.count_fingers(hand_landmarks, handedness)
                
                # Jest işle
                action, progress = self.process_gesture(finger_count)
            else:
                # El bulunamadı - state sıfırla
                self.current_fingers = -1
                self.finger_history.clear()
                with LOCK:
                    STATE["finger_count"] = -1
                    STATE["gesture_name"] = "El bulunamadı"
                    STATE["progress"] = 0.0
            
            # Arayüzü çiz
            img = self.draw_ui(img, finger_count, progress, hand_landmarks)
            
            # El bulunamadı uyarısı
            if finger_count == -1:
                cv2.rectangle(img, (w//2-120, h//2-20), (w//2+120, h//2+20), (0, 0, 180), -1)
                cv2.putText(img, "ELINI GOSTER", (w//2-100, h//2+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
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
    
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    
    .morse-box {
        font-family: 'Courier New', monospace;
        font-size: 52px;
        font-weight: bold;
        color: #00ffcc;
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        border: 3px solid #00ffcc;
        margin: 10px 0;
        min-height: 90px;
        text-shadow: 0 0 20px #00ffcc;
    }
    
    .text-box {
        font-size: 26px;
        color: #ffffff;
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #555;
        min-height: 100px;
        margin: 10px 0;
    }
    
    .command-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .command-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: white;
        border: 2px solid #4a90d9;
    }
    
    .command-card h3 {
        margin: 0;
        font-size: 32px;
    }
    
    .command-card p {
        margin: 5px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("""
<div class="main-title">
    <h1>✋ MORSE-HAND</h1>
    <p>El Hareketleri ile Mors Kodu İletişim Sistemi | TÜBİTAK 2204-A</p>
</div>
""", unsafe_allow_html=True)

# Ana düzen
col_video, col_panel = st.columns([1.5, 1])

with col_video:
    # WebRTC kamera
    webrtc_streamer(
        key="morse-hand",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        video_processor_factory=MorseHandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Komut rehberi
    st.markdown("""
    <div class="command-grid">
        <div class="command-card" style="background: linear-gradient(135deg, #f5af19 0%, #f12711 100%);">
            <h3>☝️ 1</h3>
            <p>NOKTA (.)</p>
        </div>
        <div class="command-card" style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);">
            <h3>✌️ 2</h3>
            <p>ÇİZGİ (-)</p>
        </div>
        <div class="command-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h3>✊ 0</h3>
            <p>ONAYLA</p>
        </div>
        <div class="command-card" style="background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);">
            <h3>🤟 3</h3>
            <p>BOŞLUK</p>
        </div>
        <div class="command-card" style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);">
            <h3>🖖 4</h3>
            <p>GERİ SİL</p>
        </div>
        <div class="command-card" style="background: linear-gradient(135deg, #525252 0%, #3d3d3d 100%);">
            <h3>🖐️ 5</h3>
            <p>TEMİZLE</p>
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
        if predicted != "?":
            st.success(f"🔤 Tahmin: **{predicted}**")
        else:
            st.warning("❓ Geçersiz kod")
    
    st.subheader("💬 Mesaj")
    text_display = STATE["text"] if STATE["text"] else "(Henüz mesaj yok)"
    st.markdown(f'<div class="text-box">{text_display}</div>', unsafe_allow_html=True)
    
    # Durum göstergesi
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        finger_text = STATE["gesture_name"]
        st.info(f"🤚 **{finger_text}**")
    with col_s2:
        progress_pct = int(STATE["progress"] * 100)
        st.metric("İlerleme", f"%{progress_pct}")
    
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
        if st.button("↩️ Geri", use_container_width=True):
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
            "Komut süresi (saniye)", 
            0.3, 1.5, STATE["hold_time"], 0.1,
            help="Parmağı bu kadar süre tutunca komut çalışır"
        )
        
        STATE["cooldown"] = st.slider(
            "Bekleme süresi (saniye)", 
            0.2, 1.0, STATE["cooldown"], 0.1,
            help="Komutlar arası minimum bekleme"
        )
    
    # Morse tablosu
    with st.expander("📖 Morse Alfabesi"):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            **A** .-  
            **B** -...  
            **C** -.-.  
            **D** -..  
            **E** .  
            **F** ..-.  
            **G** --.  
            **H** ....  
            **I** ..  
            """)
        with col_m2:
            st.markdown("""
            **J** .---  
            **K** -.-  
            **L** .-..  
            **M** --  
            **N** -.  
            **O** ---  
            **P** .--.  
            **Q** --.-  
            **R** .-.  
            """)
        with col_m3:
            st.markdown("""
            **S** ...  
            **T** -  
            **U** ..-  
            **V** ...-  
            **W** .--  
            **X** -..-  
            **Y** -.--  
            **Z** --..  
            """)

# Son olay
if STATE["last_event"]:
    if "✓" in STATE["last_event"]:
        st.success(STATE["last_event"])
    elif "✗" in STATE["last_event"]:
        st.error(STATE["last_event"])

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>✋ <b>MORSE-HAND</b> | TÜBİTAK 2204-A Projesi</p>
    <p>Hareket kısıtlılığı olan bireyler için el hareketi ile iletişim sistemi</p>
</div>
""", unsafe_allow_html=True)
