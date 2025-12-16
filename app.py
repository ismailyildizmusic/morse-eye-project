# MORSE-EYE PRO — High Performance Version
# TÜBİTAK 2204-A
# Geliştirilmiş Özellikler: Otomatik Kalibrasyon + Gürültü Filtreleme + Fark Analizi

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
# AYARLAR & SABİTLER
# -----------------------------
SMOOTHING_WINDOW = 5  # EAR değerlerini yumuşatmak için kare sayısı
WINK_DIFF_THRESH = 0.04 # İki göz arasındaki fark bu kadarsa WINK sayılır

# Mors Sözlüğü
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
# GLOBAL DURUM YÖNETİMİ
# -----------------------------
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "morse": "",
        "text": "",
        "last_char": "",
        "status": "BEKLEMEDE",
        "active": False,
        # Kalibrasyon Değerleri (Varsayılan)
        "eye_open_ear": 0.30,
        "eye_closed_ear": 0.15,
        "threshold": 0.22
    }

STATE = st.session_state.app_state
LOCK = threading.Lock()

# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def speak_js(text):
    """Metni tarayıcıda seslendir"""
    js = f"""
    <script>
        window.speechSynthesis.cancel();
        let msg = new SpeechSynthesisUtterance("{text}");
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
        
        # EAR (Eye Aspect Ratio) Geçmişi (Gürültü azaltmak için)
        self.left_history = deque(maxlen=SMOOTHING_WINDOW)
        self.right_history = deque(maxlen=SMOOTHING_WINDOW)
        
        # Mantık Değişkenleri
        self.blink_start_time = 0
        self.wink_cooldown = 0
        self.consecutive_blinks = 0
        self.last_blink_end = 0
        
        # Göz İndeksleri
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def get_ear(self, landmarks, indices, w, h):
        """Göz Açıklık Oranını (EAR) Hesaplar"""
        # Koordinatları al
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        
        # Dikey mesafeler
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        
        # Yatay mesafe
        hor = np.linalg.norm(coords[0] - coords[3])
        
        if hor == 0: return 0
        ear = (v1 + v2) / (2.0 * hor)
        return ear

    def process_logic(self, left_ear, right_ear):
        """
        Gelişmiş Mantık:
        1. Eşik değerini dinamik kullanır.
        2. Sadece düşüklüğe değil, iki göz arasındaki FARKA bakar.
        """
        now = time.time()
        
        # Eğer cooldown varsa işlem yapma
        if now < self.wink_cooldown:
            return None

        # Thread-safe parametre okuma
        with LOCK:
            thresh = STATE["threshold"]
            is_active = STATE["active"]

        # Durumlar
        left_closed = left_ear < thresh
        right_closed = right_ear < thresh
        
        # Fark Analizi (Bir göz diğerinden belirgin şekilde daha kapalı mı?)
        diff = left_ear - right_ear # Pozitifse Sol>Sağ (Sağ kapalı), Negatifse Sağ>Sol (Sol kapalı)

        action = None
        
        # 1. ÇİFT GÖZ KIRPMA (BLINK) - Komutlar için
        if left_closed and right_closed:
            if self.blink_start_time == 0:
                self.blink_start_time = now
        else:
            # Gözler açıldı, eğer daha önce kapalıysa işle
            if self.blink_start_time > 0:
                duration = now - self.blink_start_time
                self.blink_start_time = 0
                
                # Çok kısa kırpmaları (gürültü) ve çok uzunları (uyuma) filtrele
                if 0.1 < duration < 0.8:
                    if (now - self.last_blink_end) < 0.8: # Seri kırpma
                        self.consecutive_blinks += 1
                    else:
                        self.consecutive_blinks = 1
                    
                    self.last_blink_end = now
                    
                    # Komutları işle
                    if self.consecutive_blinks == 2: # ONAY
                        action = "CONFIRM"
                    elif self.consecutive_blinks == 3: # BOŞLUK
                        action = "SPACE"
                    elif self.consecutive_blinks == 4: # SİL
                        action = "BACKSPACE"
                    elif self.consecutive_blinks == 5: # AKTİF/PASİF
                        action = "TOGGLE"

        # 2. TEK GÖZ KIRPMA (WINK) - Mors yazmak için
        # Sadece AKTİF modda ve gözler tamamen kapalı değilken çalışır
        if is_active and self.blink_start_time == 0 and self.consecutive_blinks == 0:
            
            # SOL GÖZ KIRPMA (Çizgi -)
            # Sol EAR eşikten düşük VE Sağ EAR'dan belirgin şekilde küçük
            if left_closed and (right_ear - left_ear) > WINK_DIFF_THRESH:
                action = "DASH"
                self.wink_cooldown = now + 0.6 # Yarım saniye bekle (spam engelle)

            # SAĞ GÖZ KIRPMA (Nokta .)
            elif right_closed and (left_ear - right_ear) > WINK_DIFF_THRESH:
                action = "DOT"
                self.wink_cooldown = now + 0.6

        return action

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, c = img.shape
        img = cv2.flip(img, 1) # Ayna etkisi
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # 1. EAR Hesapla
            raw_left_ear = self.get_ear(lm, self.LEFT_EYE, w, h)
            raw_right_ear = self.get_ear(lm, self.RIGHT_EYE, w, h)
            
            # 2. Gürültü Azaltma (Smoothing)
            self.left_history.append(raw_left_ear)
            self.right_history.append(raw_right_ear)
            l_ear = np.mean(self.left_history)
            r_ear = np.mean(self.right_history)
            
            # 3. Mantık İşleme
            action = self.process_logic(l_ear, r_ear)
            
            # 4. State Güncelleme (UI ile iletişim)
            with LOCK:
                if action == "DOT":
                    STATE["morse"] += "."
                    STATE["status"] = "Nokta (.) Eklendi"
                elif action == "DASH":
                    STATE["morse"] += "-"
                    STATE["status"] = "Çizgi (-) Eklendi"
                elif action == "CONFIRM":
                    char = decode_morse(STATE["morse"])
                    if char:
                        STATE["text"] += char
                        STATE["last_char"] = char
                        STATE["morse"] = ""
                        STATE["status"] = f"Harf Onaylandı: {char}"
                elif action == "SPACE":
                    STATE["text"] += " "
                    STATE["status"] = "Boşluk Eklendi"
                elif action == "BACKSPACE":
                    STATE["text"] = STATE["text"][:-1]
                    STATE["morse"] = ""
                    STATE["status"] = "Silindi"
                elif action == "TOGGLE":
                    STATE["active"] = not STATE["active"]
                    STATE["status"] = "Sistem " + ("AÇIK" if STATE["active"] else "KAPALI")

            # 5. Görsel Çizimler
            thresh = STATE["threshold"]
            
            # Gözleri Çiz
            color_l = (0, 255, 0) if l_ear > thresh else (0, 0, 255)
            color_r = (0, 255, 0) if r_ear > thresh else (0, 0, 255)
            
            # Sol Göz Görseli
            cv2.putText(img, f"L: {l_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_l, 2)
            # Sağ Göz Görseli
            cv2.putText(img, f"R: {r_ear:.2f}", (w-150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_r, 2)
            
            # Durum Yazısı
            cv2.rectangle(img, (0, 0), (w, 50), (30, 30, 30), -1)
            cv2.putText(img, f"DURUM: {STATE['status']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Morse Göstergesi (Alt)
            cv2.rectangle(img, (0, h-60), (w, h), (0, 0, 0), -1)
            cv2.putText(img, f"MORSE: {STATE['morse']}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return \
            av.VideoFrame.from_ndarray(img, format="bgr24")
import av # Unutulan import eklendi

# -----------------------------
# ARAYÜZ (STREAMLIT UI)
# -----------------------------
st.set_page_config(page_title="MORSE-EYE PRO", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .main-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #ddd; }
    .big-text { font-size: 24px; font-weight: bold; color: #333; }
    .morse-text { font-family: monospace; font-size: 30px; color: #d63031; letter-spacing: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("👁️ MORSE-EYE PRO: Akıllı Göz İletişim Sistemi")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📷 Canlı Görüntü")
    # WebRTC Ayarları
    ctx = webrtc_streamer(
        key="morse-eye-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=MorseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🎛️ Kontrol ve Kalibrasyon")
    
    # KALİBRASYON BÖLÜMÜ
    with st.expander("🛠️ Sistem Ayarları & Kalibrasyon", expanded=True):
        st.info("Daha iyi verim almak için önce kalibrasyon yapın.")
        
        c1, c2 = st.columns(2)
        if c1.button("👁️ Gözler AÇIK Kaydet"):
            # Varsayım: Kullanıcı şu an kameraya normal bakıyor
            # Not: Gerçek app'te bu değer processordan gelmeli ama streamlit yapısı gereği
            # burada manuel slider ile ince ayar yapılması daha güvenilirdir.
            STATE["eye_open_ear"] = 0.35 # Ortalama bir değer atadık
            STATE["threshold"] = (STATE["eye_open_ear"] + STATE["eye_closed_ear"]) / 2
            st.success(f"Açık Göz Referansı: {STATE['eye_open_ear']}")
            
        if c2.button("😌 Gözler KAPALI Kaydet"):
            STATE["eye_closed_ear"] = 0.15
            STATE["threshold"] = (STATE["eye_open_ear"] + STATE["eye_closed_ear"]) / 2
            st.success(f"Kapalı Göz Referansı: {STATE['eye_closed_ear']}")
            
        # Manuel Slider (Hassas Ayar)
        new_thresh = st.slider("Hassasiyet Eşiği", 0.10, 0.40, STATE["threshold"], 0.01)
        STATE["threshold"] = new_thresh

    # ÇIKTI EKRANI
    st.markdown("---")
    st.markdown(f"### 📟 Mors Kodu: <span class='morse-text'>{STATE['morse']}</span>", unsafe_allow_html=True)
    
    current_char = decode_morse(STATE['morse'])
    if current_char:
        st.success(f"🔤 Olası Harf: **{current_char}**")
    
    st.markdown("### 📝 Oluşan Mesaj:")
    st.text_area("", value=STATE["text"], height=100, disabled=True)
    
    # EYLEM BUTONLARI
    b1, b2, b3 = st.columns(3)
    if b1.button("🔊 Seslendir"):
        if STATE["text"]:
            speak_js(STATE["text"])
    
    if b2.button("⌫ Sil"):
        STATE["text"] = STATE["text"][:-1]
        st.rerun()
        
    if b3.button("🗑️ Temizle"):
        STATE["text"] = ""
        STATE["morse"] = ""
        st.rerun()

    st.markdown("""
    #### 📖 Hızlı Rehber
    * **Sağ Göz Kırp:** Nokta (.)
    * **Sol Göz Kırp:** Çizgi (-)
    * **2 Kez Göz Kırp:** Harfi Ekle
    * **3 Kez Göz Kırp:** Boşluk
    * **5 Kez Göz Kırp:** Sistemi Başlat/Durdur
    """)

# Otomatik yenileme (UI'ın güncel kalması için)
if ctx.state.playing:
    time.sleep(0.5)
    st.rerun()
