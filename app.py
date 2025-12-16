# MORSE-EYE — Eye Wink to Morse (Streamlit + WebRTC + MediaPipe)
# TÜBİTAK 2204-A için demo/prototip
# YENİ YAKLAŞIM: Sağ göz kırpma = Nokta, Sol göz kırpma = Çizgi

import time
import threading
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

import mediapipe as mp


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
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9"
}


# -----------------------------
# WINK TİPLERİ
# -----------------------------
class WinkType(Enum):
    NONE = 0
    LEFT_WINK = 1      # Sol göz kırpma (sağ açık)
    RIGHT_WINK = 2     # Sağ göz kırpma (sol açık)
    BOTH_BLINK = 3     # İki göz birden


# -----------------------------
# PAYLAŞILAN DURUM (thread-safe)
# -----------------------------
@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: bool = False

    morse: str = ""
    text: str = ""

    last_event: str = ""
    last_symbol: str = ""
    
    # Debug bilgileri
    left_ear: float = 0.25
    right_ear: float = 0.25
    left_eye_closed: bool = False
    right_eye_closed: bool = False
    
    # Blink sayacı (sistem açma/kapama için)
    both_blink_count: int = 0
    last_both_blink_ts: float = 0.0
    blink_sequence_start: float = 0.0
    
    # Wink cooldown (çift algılamayı önlemek için)
    last_wink_ts: float = 0.0
    
    # EAR eşikleri
    ear_threshold: float = 0.20


# Global state
if "morse_state" not in st.session_state:
    st.session_state.morse_state = SharedState()

STATE = st.session_state.morse_state


# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def decode_morse(m: str) -> str:
    return MORSE_TO_CHAR.get(m, "?") if m else ""


def speak_in_browser(text: str):
    """Tarayıcıda SpeechSynthesis ile seslendirir"""
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    components.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance("{safe}");
        msg.lang = "tr-TR";
        msg.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


# -----------------------------
# VIDEO PROCESSOR
# -----------------------------
class MorseEyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Timing parametreleri
        self.wink_cooldown = 0.4          # Wink'ler arası minimum süre
        self.blink_confirm_delay = 0.8    # Blink serisi onay süresi
        self.blink_sequence_timeout = 2.0 # Seri zaman aşımı
        
        # Önceki göz durumları (geçiş algılama için)
        self._prev_left_closed = False
        self._prev_right_closed = False
        self._prev_both_closed = False
        
        # EAR smoothing
        self.left_ear_buffer = deque(maxlen=3)
        self.right_ear_buffer = deque(maxlen=3)

        # ===== MediaPipe FaceMesh Landmark İndeksleri =====
        # Sol göz (kullanıcının solu, kameranın sağı)
        self.LEFT_EYE = {
            'top': [159, 158, 157, 173],      # Üst kapak
            'bottom': [145, 144, 153, 154],   # Alt kapak
            'left': 33,                        # Sol köşe
            'right': 133                       # Sağ köşe
        }
        
        # Sağ göz (kullanıcının sağı, kameranın solu)
        self.RIGHT_EYE = {
            'top': [386, 385, 384, 398],      # Üst kapak
            'bottom': [374, 373, 390, 249],   # Alt kapak
            'left': 362,                       # Sol köşe
            'right': 263                       # Sağ köşe
        }

    def _landmark_xy(self, lm, w, h, idx):
        p = lm[idx]
        return np.array([p.x * w, p.y * h])

    def _calculate_ear(self, lm, w, h, eye_indices):
        """
        Eye Aspect Ratio (EAR) hesaplar.
        EAR = (dikey mesafelerin ortalaması) / (yatay mesafe)
        Düşük EAR = göz kapalı
        """
        # Dikey mesafeler (birden fazla nokta ile daha doğru)
        vertical_dists = []
        for top_idx, bottom_idx in zip(eye_indices['top'], eye_indices['bottom']):
            top = self._landmark_xy(lm, w, h, top_idx)
            bottom = self._landmark_xy(lm, w, h, bottom_idx)
            vertical_dists.append(np.linalg.norm(top - bottom))
        
        avg_vertical = np.mean(vertical_dists)
        
        # Yatay mesafe
        left_corner = self._landmark_xy(lm, w, h, eye_indices['left'])
        right_corner = self._landmark_xy(lm, w, h, eye_indices['right'])
        horizontal = np.linalg.norm(left_corner - right_corner)
        
        ear = avg_vertical / (horizontal + 1e-6)
        return float(ear)

    def _detect_wink_type(self, left_ear: float, right_ear: float, threshold: float) -> WinkType:
        """
        Hangi gözün kırpıldığını algılar.
        
        Mantık:
        - Her iki göz de kapalı (düşük EAR) → BOTH_BLINK
        - Sadece sol göz kapalı, sağ açık → LEFT_WINK  
        - Sadece sağ göz kapalı, sol açık → RIGHT_WINK
        - İkisi de açık → NONE
        """
        left_closed = left_ear < threshold
        right_closed = right_ear < threshold
        
        # State'e kaydet
        with STATE.lock:
            STATE.left_eye_closed = left_closed
            STATE.right_eye_closed = right_closed
        
        if left_closed and right_closed:
            return WinkType.BOTH_BLINK
        elif left_closed and not right_closed:
            # Sol kapalı, sağ açık - ama sağın gerçekten açık olduğundan emin ol
            if right_ear > threshold * 1.3:  # Sağ göz kesinlikle açık
                return WinkType.LEFT_WINK
        elif right_closed and not left_closed:
            # Sağ kapalı, sol açık
            if left_ear > threshold * 1.3:  # Sol göz kesinlikle açık
                return WinkType.RIGHT_WINK
        
        return WinkType.NONE

    def _process_wink(self, wink_type: WinkType):
        """
        Wink tipine göre işlem yap.
        Geçiş algılama: kapalıdan açığa geçişte işlem yap.
        """
        now = time.time()
        
        with STATE.lock:
            active = STATE.active
            cooldown_ok = (now - STATE.last_wink_ts) > self.wink_cooldown
        
        # === İki göz birden kırpma (sistem kontrolü) ===
        if wink_type == WinkType.BOTH_BLINK:
            if not self._prev_both_closed:
                # Yeni kapanma başladı
                self._prev_both_closed = True
        else:
            if self._prev_both_closed:
                # Gözler açıldı = blink tamamlandı
                self._prev_both_closed = False
                with STATE.lock:
                    time_since_last = now - STATE.last_both_blink_ts
                    
                    if time_since_last > 0.2:  # Debounce
                        if STATE.both_blink_count == 0:
                            STATE.blink_sequence_start = now
                        
                        STATE.both_blink_count += 1
                        STATE.last_both_blink_ts = now
                        STATE.last_event = f"Çift kırpma #{STATE.both_blink_count}"
        
        # === Tek göz kırpma (sembol ekleme - sadece aktifken) ===
        if active and cooldown_ok:
            # SOL GÖZ WINK
            if wink_type == WinkType.LEFT_WINK:
                if not self._prev_left_closed:
                    self._prev_left_closed = True
            elif self._prev_left_closed and wink_type == WinkType.NONE:
                # Sol göz açıldı = wink tamamlandı
                self._prev_left_closed = False
                with STATE.lock:
                    STATE.morse += "-"  # Sol göz = Çizgi
                    STATE.last_symbol = "-"
                    STATE.last_wink_ts = now
                    STATE.last_event = "Sol göz kırpıldı → Çizgi (-)"
            
            # SAĞ GÖZ WINK
            if wink_type == WinkType.RIGHT_WINK:
                if not self._prev_right_closed:
                    self._prev_right_closed = True
            elif self._prev_right_closed and wink_type == WinkType.NONE:
                # Sağ göz açıldı = wink tamamlandı
                self._prev_right_closed = False
                with STATE.lock:
                    STATE.morse += "."  # Sağ göz = Nokta
                    STATE.last_symbol = "."
                    STATE.last_wink_ts = now
                    STATE.last_event = "Sağ göz kırpıldı → Nokta (.)"
        
        # Tek göz durumlarını sıfırla (wink değilse)
        if wink_type != WinkType.LEFT_WINK:
            self._prev_left_closed = False
        if wink_type != WinkType.RIGHT_WINK:
            self._prev_right_closed = False

    def _check_blink_command(self):
        """
        Çift göz kırpma serisini değerlendir
        """
        now = time.time()
        
        with STATE.lock:
            if STATE.both_blink_count == 0:
                return None
            
            time_since_last = now - STATE.last_both_blink_ts
            
            # Seri bitti mi?
            if time_since_last >= self.blink_confirm_delay:
                count = STATE.both_blink_count
                STATE.both_blink_count = 0
                STATE.blink_sequence_start = 0
                
                # Komut yorumlama
                if count >= 3:
                    return "TOGGLE_ACTIVE"
                elif count == 2:
                    return "SPACE"
                elif count == 1:
                    return "CONFIRM_CHAR"
            
            # Zaman aşımı kontrolü
            if STATE.blink_sequence_start > 0:
                if now - STATE.blink_sequence_start > self.blink_sequence_timeout:
                    STATE.both_blink_count = 0
                    STATE.blink_sequence_start = 0
        
        return None

    def _execute_command(self, cmd: str):
        """Komutu çalıştır"""
        if cmd is None:
            return
            
        with STATE.lock:
            if cmd == "TOGGLE_ACTIVE":
                STATE.active = not STATE.active
                status = "AKTİF ✅" if STATE.active else "PASİF ⛔"
                STATE.last_event = f"Sistem {status}"
                if not STATE.active:
                    STATE.morse = ""
                    
            elif cmd == "CONFIRM_CHAR":
                if STATE.morse:
                    ch = decode_morse(STATE.morse)
                    STATE.text += ch
                    STATE.last_event = f"Harf: {STATE.morse} → {ch}"
                    STATE.morse = ""
                else:
                    STATE.last_event = "Morse tamponu boş!"
                    
            elif cmd == "SPACE":
                STATE.text += " "
                STATE.last_event = "Boşluk eklendi"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # EAR eşiğini al
        with STATE.lock:
            ear_threshold = STATE.ear_threshold

        # Yüz algılama
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        left_ear = 0.25
        right_ear = 0.25
        wink_type = WinkType.NONE
        face_detected = False

        if res.multi_face_landmarks:
            face_detected = True
            lm = res.multi_face_landmarks[0].landmark

            # Her göz için EAR hesapla
            left_ear = self._calculate_ear(lm, w, h, self.LEFT_EYE)
            right_ear = self._calculate_ear(lm, w, h, self.RIGHT_EYE)
            
            # Smoothing
            self.left_ear_buffer.append(left_ear)
            self.right_ear_buffer.append(right_ear)
            left_ear = float(np.mean(self.left_ear_buffer))
            right_ear = float(np.mean(self.right_ear_buffer))
            
            # State'e kaydet
            with STATE.lock:
                STATE.left_ear = left_ear
                STATE.right_ear = right_ear

            # Wink tipini algıla
            wink_type = self._detect_wink_type(left_ear, right_ear, ear_threshold)
            
            # Wink işle
            self._process_wink(wink_type)
            
            # Göz çerçevelerini çiz
            for eye_name, eye_indices, color in [
                ("SOL", self.LEFT_EYE, (255, 100, 100)),
                ("SAG", self.RIGHT_EYE, (100, 100, 255))
            ]:
                left_corner = self._landmark_xy(lm, w, h, eye_indices['left'])
                right_corner = self._landmark_xy(lm, w, h, eye_indices['right'])
                cv2.line(img, tuple(left_corner.astype(int)), 
                        tuple(right_corner.astype(int)), color, 2)

        # Blink komutlarını kontrol et
        cmd = self._check_blink_command()
        self._execute_command(cmd)

        # State al
        with STATE.lock:
            active = STATE.active
            morse = STATE.morse
            text = STATE.text
            last_event = STATE.last_event
            both_blink_count = STATE.both_blink_count
            left_closed = STATE.left_eye_closed
            right_closed = STATE.right_eye_closed

        # =====================
        # EKRAN ÜZERİ GÖSTERGE
        # =====================
        
        # Üst panel
        cv2.rectangle(img, (5, 5), (w - 5, 175), (255, 255, 255), -1)
        cv2.rectangle(img, (5, 5), (w - 5, 175), (30, 41, 59), 2)

        # Başlık ve durum
        status_text = "AKTIF" if active else "PASIF"
        status_color = (0, 150, 0) if active else (0, 0, 200)
        cv2.putText(img, f"MORSE-EYE (Wink) | {status_text}", (15, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # EAR değerleri ve göz durumu
        left_status = "KAPALI" if left_closed else "ACIK"
        right_status = "KAPALI" if right_closed else "ACIK"
        
        cv2.putText(img, f"Sol Goz: {left_ear:.3f} ({left_status})", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 50, 50), 1)
        cv2.putText(img, f"Sag Goz: {right_ear:.3f} ({right_status})", (15, 68),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 150), 1)

        # Wink tipi
        wink_names = {
            WinkType.NONE: "---",
            WinkType.LEFT_WINK: "SOL GOZ KIRPTI!",
            WinkType.RIGHT_WINK: "SAG GOZ KIRPTI!",
            WinkType.BOTH_BLINK: "IKI GOZ KIRPTI!"
        }
        wink_colors = {
            WinkType.NONE: (100, 100, 100),
            WinkType.LEFT_WINK: (0, 0, 255),
            WinkType.RIGHT_WINK: (255, 0, 0),
            WinkType.BOTH_BLINK: (0, 150, 0)
        }
        cv2.putText(img, f"Algilanan: {wink_names[wink_type]}", (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, wink_colors[wink_type], 2)

        # Blink sayacı
        cv2.putText(img, f"Cift Kirpma Sayaci: {both_blink_count}", (15, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 0, 100), 1)

        # MORSE - BÜYÜK
        morse_display = morse if morse else "---"
        cv2.putText(img, f"MORSE: {morse_display}", (15, 138), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 50, 0), 2)
        
        # Anlık çözüm
        current_char = decode_morse(morse) if morse else "-"
        cv2.putText(img, f"Harf: {current_char}", (w - 120, 138),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

        # MESAJ
        text_display = text[-35:] if text else "(bos)"
        cv2.putText(img, f"MESAJ: {text_display}", (15, 168),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 150), 2)

        # Alt bilgi
        if last_event:
            cv2.rectangle(img, (5, h - 35), (w - 5, h - 5), (240, 255, 240), -1)
            cv2.putText(img, last_event[:55], (15, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)

        # Aktif değilse uyarı
        if not active:
            cv2.rectangle(img, (w//2 - 160, h//2 - 25), (w//2 + 160, h//2 + 25), (0, 0, 180), -1)
            cv2.putText(img, "3x CIFT KIRP veya BUTON ile AC", (w//2 - 150, h//2 + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Yüz bulunamadıysa
        if not face_detected:
            cv2.rectangle(img, (w//2 - 100, h//2 + 40), (w//2 + 100, h//2 + 70), (0, 0, 200), -1)
            cv2.putText(img, "YUZ BULUNAMADI!", (w//2 - 85, h//2 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="MORSE-EYE | TÜBİTAK 2204-A", page_icon="👁️", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#f8fafc; }
.block-container { max-width: 1200px; padding-top: 1rem; }
.header {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border: 1px solid rgba(148,163,184,0.25);
  padding: 18px 18px;
  border-radius: 16px;
  color: white;
}
.badge {
  display:inline-block; margin-top:8px;
  padding:6px 12px; border-radius:999px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 0.9rem;
}
.big-text {
  font-size: 2rem;
  font-weight: bold;
  color: #1e3a5f;
  padding: 10px;
  background: #e8f4f8;
  border-radius: 10px;
  margin: 10px 0;
}
.morse-display {
  font-size: 2.5rem;
  font-family: monospace;
  letter-spacing: 8px;
  color: #c0392b;
  padding: 15px;
  background: #fdf2f2;
  border-radius: 10px;
  text-align: center;
}
.command-box {
  background: #f0f9ff;
  border: 2px solid #0ea5e9;
  border-radius: 12px;
  padding: 15px;
  margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1 style="margin:0; font-weight:900;">👁️ MORSE-EYE — Göz Kırpma ile Mors Kod İletişimi</h1>
  <div class="badge">🏆 TÜBİTAK 2204-A • Wink Detection Version</div>
  <p style="margin:10px 0 0 0; color:#cbd5e1;">
    Sağ göz = Nokta (.) • Sol göz = Çizgi (-) • Çift kırpma = Komutlar
  </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    # MANUEL KONTROL
    st.subheader("🚀 Hızlı Başlat")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ AKTİF ET", use_container_width=True, type="primary"):
            with STATE.lock:
                STATE.active = True
                STATE.last_event = "Manuel: AKTİF"
    with col_btn2:
        if st.button("⏹️ PASİF", use_container_width=True):
            with STATE.lock:
                STATE.active = False
                STATE.morse = ""
                STATE.last_event = "Manuel: PASİF"
    
    # Durum göstergesi
    with STATE.lock:
        is_active = STATE.active
    if is_active:
        st.success("✅ Sistem AKTİF")
    else:
        st.warning("⛔ Sistem PASİF")
    
    st.divider()
    
    # EAR Eşiği ayarı
    st.subheader("🎯 Hassasiyet Ayarı")
    ear_thr = st.slider(
        "EAR Eşiği (Göz Kapanma)", 
        0.12, 0.30, 0.20, 0.01,
        help="Düşük = daha hassas (yanlış algılama riski), Yüksek = daha az hassas"
    )
    
    with STATE.lock:
        STATE.ear_threshold = float(ear_thr)
    
    # Debug bilgileri
    st.divider()
    st.subheader("📊 Canlı Değerler")
    with STATE.lock:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("Sol EAR", f"{STATE.left_ear:.3f}")
        with col_d2:
            st.metric("Sağ EAR", f"{STATE.right_ear:.3f}")
        
        st.write(f"**Eşik:** {STATE.ear_threshold:.2f}")
        st.write(f"**Çift Kırpma:** {STATE.both_blink_count}")
    
    st.divider()
    
    # KOMUTLAR
    st.subheader("⌨️ Komutlar")
    st.markdown("""
    <div class="command-box">
    <b>🔵 Sağ Göz Kırp:</b> Nokta (.) ekle<br>
    <b>🔴 Sol Göz Kırp:</b> Çizgi (-) ekle<br>
    <b>🟢 1x Çift Kırp:</b> Harfi onayla<br>
    <b>🟡 2x Çift Kırp:</b> Boşluk ekle<br>
    <b>⚪ 3x Çift Kırp:</b> Sistemi Aç/Kapat
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🗑️ Mesaj Kontrol")
    if st.button("↩️ Son karakteri sil", use_container_width=True):
        with STATE.lock:
            STATE.text = STATE.text[:-1]
            STATE.last_event = "Son karakter silindi"

    if st.button("🧹 Morse temizle", use_container_width=True):
        with STATE.lock:
            STATE.morse = ""
            STATE.last_event = "Morse temizlendi"

    if st.button("🧾 Tümünü temizle", use_container_width=True):
        with STATE.lock:
            STATE.text = ""
            STATE.morse = ""
            STATE.last_event = "Temizlendi"

    st.divider()
    st.subheader("🔊 Sesli Oku")
    if st.button("▶️ Mesajı seslendir", use_container_width=True):
        with STATE.lock:
            t = STATE.text.strip()
        if t:
            speak_in_browser(t)
        else:
            st.warning("Mesaj boş!")


# --- Main layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 🎥 Kamera")
    st.info("💡 **Kullanım:** Sağ gözü kapat = nokta, Sol gözü kapat = çizgi. Harfi onaylamak için iki gözü bir kez kırp.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="morse-eye-wink",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MorseEyeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 📌 Çıktı Paneli")
    
    with STATE.lock:
        active = STATE.active
        morse = STATE.morse
        text = STATE.text
        last_event = STATE.last_event
        current_char = decode_morse(morse) if morse else ""

    # Durum
    if active:
        st.success("### ✅ SİSTEM AKTİF")
    else:
        st.error("### ⛔ SİSTEM PASİF")
    
    # Morse
    st.markdown("#### 📟 Morse Tamponu:")
    morse_html = f'<div class="morse-display">{morse if morse else "---"}</div>'
    st.markdown(morse_html, unsafe_allow_html=True)
    
    # Anlık harf
    if morse:
        st.markdown(f"#### 🔤 Anlık Çözüm: **{current_char}**")
    
    # Mesaj
    st.markdown("#### 💬 Mesaj:")
    text_html = f'<div class="big-text">{text if text else "(Henüz mesaj yok)"}</div>'
    st.markdown(text_html, unsafe_allow_html=True)
    
    # Son olay
    if last_event:
        st.info(f"📢 {last_event}")
    
    # Yenile butonu
    if st.button("🔄 Paneli Yenile", use_container_width=True):
        st.rerun()

    # Morse tablosu
    with st.expander("📖 Morse Alfabesi"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            A: .-&nbsp;&nbsp;|&nbsp;&nbsp;B: -...  
            C: -.-.&nbsp;&nbsp;|&nbsp;&nbsp;D: -..  
            E: .&nbsp;&nbsp;|&nbsp;&nbsp;F: ..-.  
            G: --.&nbsp;&nbsp;|&nbsp;&nbsp;H: ....  
            I: ..&nbsp;&nbsp;|&nbsp;&nbsp;J: .---  
            K: -.-&nbsp;&nbsp;|&nbsp;&nbsp;L: .-..  
            M: --&nbsp;&nbsp;|&nbsp;&nbsp;N: -.  
            """)
        with col_m2:
            st.markdown("""
            O: ---&nbsp;&nbsp;|&nbsp;&nbsp;P: .--.  
            Q: --.-&nbsp;&nbsp;|&nbsp;&nbsp;R: .-.  
            S: ...&nbsp;&nbsp;|&nbsp;&nbsp;T: -  
            U: ..-&nbsp;&nbsp;|&nbsp;&nbsp;V: ...-  
            W: .--&nbsp;&nbsp;|&nbsp;&nbsp;X: -..-  
            Y: -.--&nbsp;&nbsp;|&nbsp;&nbsp;Z: --..  
            """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b;'>"
    "MORSE-EYE • TÜBİTAK 2204-A Demo • Göz kırpma ile iletişim sistemi"
    "</div>",
    unsafe_allow_html=True,
)
