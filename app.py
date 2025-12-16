# MORSE-EYE — Eye Gaze to Morse (Streamlit + WebRTC + MediaPipe)
# TÜBİTAK 2204-A için demo/prototip
# Düzeltilmiş ve iyileştirilmiş versiyon

import time
import threading
from dataclasses import dataclass, field
from collections import deque

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
# PAYLAŞILAN DURUM (thread-safe)
# -----------------------------
@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: bool = False

    morse: str = ""
    text: str = ""

    last_dir: str = "CENTER"
    last_symbol: str = ""
    last_event: str = ""
    
    # Gaze ratio (debug için)
    current_gaze_ratio: float = 0.5
    
    # EAR değeri (debug için)
    current_ear: float = 0.25

    # Blink için state machine
    eye_closed: bool = False
    blink_count: int = 0
    last_blink_ts: float = 0.0
    blink_sequence_start: float = 0.0

    # Kalibrasyon / eşik
    center_ratio: float = 0.5
    ratio_left: float = 0.40
    ratio_right: float = 0.60
    
    # EAR eşiği (göz kırpma için)
    ear_threshold: float = 0.21

    # hız/kararlılık
    hold_start_ts: float = 0.0
    hold_dir: str = "CENTER"
    last_symbol_ts: float = 0.0


# Global state - uygulama başladığında bir kere oluşturulur
if "morse_state" not in st.session_state:
    st.session_state.morse_state = SharedState()

STATE = st.session_state.morse_state


# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def clamp(x, a, b):
    return max(a, min(b, x))


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
# VIDEO PROCESSOR (GÖZ + KIRPMA)
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

        # Blink tespiti için parametreler
        self.blink_min_interval = 0.15      # İki blink arası minimum süre
        self.blink_sequence_timeout = 1.5   # Blink serisinin zaman aşımı
        self.blink_confirm_delay = 0.7      # Seri bittikten sonra bekleme

        # Sembol eklemek için bakışı sabit tutma
        self.dwell_time = 0.40              # saniye
        self.symbol_cooldown = 0.50         # saniye

        # Yön filtresi (smooth için)
        self.ratio_smooth = deque(maxlen=5)
        
        # Önceki EAR değeri (geçiş algılama için)
        self._prev_ear = 0.25
        self._eye_was_closed = False

        # Göz landmark indeksleri (MediaPipe FaceMesh 468+10 iris)
        # Sol göz köşeleri
        self.LEFT_EYE_LEFT = 33      # Sol gözün sol köşesi
        self.LEFT_EYE_RIGHT = 133    # Sol gözün sağ köşesi
        # Sağ göz köşeleri  
        self.RIGHT_EYE_LEFT = 362    # Sağ gözün sol köşesi
        self.RIGHT_EYE_RIGHT = 263   # Sağ gözün sağ köşesi
        
        # Göz kapağı (EAR hesabı için)
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

        # İris indeksleri (refine_landmarks=True ile aktif)
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def _landmark_xy(self, lm, w, h, idx):
        p = lm[idx]
        return (p.x * w, p.y * h)

    def _iris_center(self, lm, w, h, idxs):
        pts = [self._landmark_xy(lm, w, h, i) for i in idxs]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _gaze_ratio(self, lm, w, h):
        """
        Gaze ratio hesaplar: 0.0 = tam sol, 1.0 = tam sağ, 0.5 = orta
        """
        # Sol göz köşeleri
        left_eye_left = self._landmark_xy(lm, w, h, self.LEFT_EYE_LEFT)
        left_eye_right = self._landmark_xy(lm, w, h, self.LEFT_EYE_RIGHT)
        
        # Sağ göz köşeleri
        right_eye_left = self._landmark_xy(lm, w, h, self.RIGHT_EYE_LEFT)
        right_eye_right = self._landmark_xy(lm, w, h, self.RIGHT_EYE_RIGHT)

        # İris merkezleri
        try:
            left_iris = self._iris_center(lm, w, h, self.LEFT_IRIS)
            right_iris = self._iris_center(lm, w, h, self.RIGHT_IRIS)
        except Exception:
            # Fallback
            left_iris = ((left_eye_left[0] + left_eye_right[0]) / 2,
                        (left_eye_left[1] + left_eye_right[1]) / 2)
            right_iris = ((right_eye_left[0] + right_eye_right[0]) / 2,
                         (right_eye_left[1] + right_eye_right[1]) / 2)

        # Her göz için iris'in göz genişliği içindeki pozisyonunu hesapla
        left_eye_width = abs(left_eye_right[0] - left_eye_left[0])
        right_eye_width = abs(right_eye_right[0] - right_eye_left[0])
        
        if left_eye_width < 1:
            left_eye_width = 1
        if right_eye_width < 1:
            right_eye_width = 1

        # İris'in göz içindeki yatay pozisyonu (0-1 arası)
        left_ratio = (left_iris[0] - left_eye_left[0]) / left_eye_width
        right_ratio = (right_iris[0] - right_eye_left[0]) / right_eye_width

        # İki gözün ortalaması
        ratio = (left_ratio + right_ratio) / 2.0
        ratio = clamp(ratio, 0.0, 1.0)

        return ratio, left_iris, right_iris, \
               (left_eye_left, left_eye_right), (right_eye_left, right_eye_right)

    def _eye_aspect_ratio(self, lm, w, h):
        """
        Eye Aspect Ratio (EAR) hesaplar.
        Düşük değer = göz kapalı, yüksek değer = göz açık
        """
        # Sol göz
        left_top = self._landmark_xy(lm, w, h, self.LEFT_EYE_TOP)
        left_bottom = self._landmark_xy(lm, w, h, self.LEFT_EYE_BOTTOM)
        left_left = self._landmark_xy(lm, w, h, self.LEFT_EYE_LEFT)
        left_right = self._landmark_xy(lm, w, h, self.LEFT_EYE_RIGHT)
        
        # Sağ göz
        right_top = self._landmark_xy(lm, w, h, self.RIGHT_EYE_TOP)
        right_bottom = self._landmark_xy(lm, w, h, self.RIGHT_EYE_BOTTOM)
        right_left = self._landmark_xy(lm, w, h, self.RIGHT_EYE_LEFT)
        right_right = self._landmark_xy(lm, w, h, self.RIGHT_EYE_RIGHT)

        # EAR = dikey mesafe / yatay mesafe
        left_vertical = _dist(left_top, left_bottom)
        left_horizontal = _dist(left_left, left_right)
        right_vertical = _dist(right_top, right_bottom)
        right_horizontal = _dist(right_left, right_right)

        left_ear = left_vertical / (left_horizontal + 1e-6)
        right_ear = right_vertical / (right_horizontal + 1e-6)

        return (left_ear + right_ear) / 2.0

    def _process_blink(self, ear: float):
        """
        Göz kırpma algılama - state machine yaklaşımı
        """
        now = time.time()
        
        with STATE.lock:
            ear_threshold = STATE.ear_threshold
        
        # Göz kapalı mı?
        eye_is_closed = ear < ear_threshold
        
        # Geçiş algılama: kapalıdan açığa geçiş = 1 blink
        if self._eye_was_closed and not eye_is_closed:
            # Göz açıldı = blink tamamlandı
            with STATE.lock:
                time_since_last = now - STATE.last_blink_ts
                
                if time_since_last > self.blink_min_interval:
                    # Yeni bir blink
                    if STATE.blink_count == 0:
                        # Yeni seri başlıyor
                        STATE.blink_sequence_start = now
                    
                    STATE.blink_count += 1
                    STATE.last_blink_ts = now
                    STATE.last_event = f"Blink #{STATE.blink_count} algılandı"
        
        self._eye_was_closed = eye_is_closed

    def _check_blink_command(self):
        """
        Blink serisini değerlendir ve komut üret
        """
        now = time.time()
        
        with STATE.lock:
            if STATE.blink_count == 0:
                return None
            
            time_since_last = now - STATE.last_blink_ts
            
            # Seri bitti mi? (son blink'ten bu yana yeterli süre geçti mi?)
            if time_since_last >= self.blink_confirm_delay:
                count = STATE.blink_count
                STATE.blink_count = 0
                STATE.blink_sequence_start = 0
                
                # Komut yorumlama
                if count >= 5:
                    return "TOGGLE_ACTIVE"
                elif count == 3:
                    return "SPACE"
                elif count == 2:
                    return "CONFIRM_CHAR"
                else:
                    # 1 veya 4 blink - bir şey yapma
                    return None
            
            # Zaman aşımı kontrolü
            if STATE.blink_sequence_start > 0:
                if now - STATE.blink_sequence_start > self.blink_sequence_timeout:
                    # Seri zaman aşımına uğradı, sıfırla
                    STATE.blink_count = 0
                    STATE.blink_sequence_start = 0
        
        return None

    def _execute_command(self, cmd: str):
        """
        Komutu çalıştır
        """
        if cmd is None:
            return
            
        with STATE.lock:
            if cmd == "TOGGLE_ACTIVE":
                STATE.active = not STATE.active
                STATE.last_event = "Sistem " + ("AKTİF ✅" if STATE.active else "PASİF ⛔")
                if not STATE.active:
                    STATE.morse = ""
                    
            elif cmd == "CONFIRM_CHAR":
                if STATE.morse:
                    ch = decode_morse(STATE.morse)
                    STATE.text += ch
                    STATE.last_event = f"Harf eklendi: {STATE.morse} → {ch}"
                    STATE.morse = ""
                else:
                    STATE.last_event = "Morse tamponu boş!"
                    
            elif cmd == "SPACE":
                STATE.text += " "
                STATE.last_event = "Boşluk eklendi"

    def _update_gaze_logic(self, direction: str, mapping_right_dot: bool):
        """
        Bakış yönüne göre sembol ekleme
        """
        now = time.time()
        
        with STATE.lock:
            STATE.last_dir = direction
            
            if not STATE.active:
                return
            
            # Sağ/Sol bakışla sembol ekleme
            if direction in ("LEFT", "RIGHT"):
                if STATE.hold_dir != direction:
                    # Yön değişti, timer sıfırla
                    STATE.hold_dir = direction
                    STATE.hold_start_ts = now
                else:
                    # Aynı yöne bakmaya devam
                    held_time = now - STATE.hold_start_ts
                    time_since_symbol = now - STATE.last_symbol_ts
                    
                    if held_time >= self.dwell_time and time_since_symbol > self.symbol_cooldown:
                        # Sembol ekle
                        if mapping_right_dot:
                            sym = "." if direction == "RIGHT" else "-"
                        else:
                            sym = "-" if direction == "RIGHT" else "."
                        
                        STATE.morse += sym
                        STATE.last_symbol = sym
                        STATE.last_symbol_ts = now
                        STATE.last_event = f"Sembol: {sym} (Morse: {STATE.morse})"
                        
                        # Aynı bakışta sürekli eklemeyi önle
                        STATE.hold_start_ts = now + 0.1
            else:
                # Ortaya bakıyor
                STATE.hold_dir = direction
                STATE.hold_start_ts = now

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Parametreleri al
        with STATE.lock:
            ratio_left = STATE.ratio_left
            ratio_right = STATE.ratio_right
            active = STATE.active
            morse = STATE.morse
            text = STATE.text
            last_event = STATE.last_event
            blink_count = STATE.blink_count

        # Yüz algılama
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        direction = "NO_FACE"
        gaze_ratio = 0.5
        ear = 0.25

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # Gaze ratio hesapla
            gaze_ratio, left_iris, right_iris, left_eye, right_eye = self._gaze_ratio(lm, w, h)
            
            # Smooth
            self.ratio_smooth.append(gaze_ratio)
            smooth_ratio = float(np.mean(self.ratio_smooth))

            # Yön belirleme
            if smooth_ratio < ratio_left:
                direction = "LEFT ◀"
            elif smooth_ratio > ratio_right:
                direction = "RIGHT ▶"
            else:
                direction = "CENTER ●"

            # EAR hesapla
            ear = self._eye_aspect_ratio(lm, w, h)
            
            # State'e kaydet (debug için)
            with STATE.lock:
                STATE.current_gaze_ratio = smooth_ratio
                STATE.current_ear = ear

            # Blink işleme
            self._process_blink(ear)

            # Görselleştirme - iris noktaları
            cv2.circle(img, (int(left_iris[0]), int(left_iris[1])), 3, (0, 255, 255), -1)
            cv2.circle(img, (int(right_iris[0]), int(right_iris[1])), 3, (0, 255, 255), -1)
            
            # Göz çerçeveleri
            cv2.line(img, (int(left_eye[0][0]), int(left_eye[0][1])), 
                    (int(left_eye[1][0]), int(left_eye[1][1])), (255, 100, 100), 2)
            cv2.line(img, (int(right_eye[0][0]), int(right_eye[0][1])), 
                    (int(right_eye[1][0]), int(right_eye[1][1])), (255, 100, 100), 2)

        # Blink komutlarını kontrol et
        cmd = self._check_blink_command()
        self._execute_command(cmd)

        # Gaze mantığı
        mapping_right_dot = True  # Varsayılan: sağ = nokta
        self._update_gaze_logic(direction.split()[0] if direction != "NO_FACE" else "CENTER", mapping_right_dot)

        # State'i tekrar al (güncellenmiş olabilir)
        with STATE.lock:
            active = STATE.active
            morse = STATE.morse
            text = STATE.text
            last_event = STATE.last_event
            last_symbol = STATE.last_symbol

        # =====================
        # EKRAN ÜZERİ GÖSTERGE (HUD)
        # =====================
        
        # Üst panel - arka plan
        cv2.rectangle(img, (5, 5), (w - 5, 160), (255, 255, 255), -1)
        cv2.rectangle(img, (5, 5), (w - 5, 160), (30, 41, 59), 2)

        # Başlık ve durum
        status_text = "AKTIF" if active else "PASIF"
        status_color = (0, 150, 0) if active else (0, 0, 200)
        cv2.putText(img, f"MORSE-EYE | {status_text}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Gaze bilgisi
        cv2.putText(img, f"Yon: {direction} | Ratio: {gaze_ratio:.3f} | EAR: {ear:.3f}", 
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

        # Blink sayısı
        cv2.putText(img, f"Blink Sayaci: {blink_count}", 
                   (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 100), 1)

        # MORSE - BÜYÜK VE NET
        morse_display = morse if morse else "---"
        cv2.putText(img, f"MORSE: {morse_display}", (15, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 50, 0), 2)
        
        # Anlık çözüm
        current_char = decode_morse(morse) if morse else "-"
        cv2.putText(img, f"Anlık Harf: {current_char}", (15, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

        # MESAJ - BÜYÜK
        text_display = text[-30:] if text else "(bos)"
        cv2.putText(img, f"MESAJ: {text_display}", (15, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)

        # Alt bilgi - son olay
        if last_event:
            cv2.rectangle(img, (5, h - 35), (w - 5, h - 5), (240, 255, 240), -1)
            cv2.putText(img, last_event[:60], (15, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)

        # Aktif değilse uyarı
        if not active:
            cv2.rectangle(img, (w//2 - 180, h//2 - 25), (w//2 + 180, h//2 + 25), (0, 0, 200), -1)
            cv2.putText(img, "5x KIRP veya BUTON ile AKTIF ET", (w//2 - 170, h//2 + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1 style="margin:0; font-weight:900;">👁️ MORSE-EYE — Göz Hareketleri ile Mors Kod İletişimi</h1>
  <div class="badge">🏆 TÜBİTAK 2204-A • Demo Web App</div>
  <p style="margin:10px 0 0 0; color:#cbd5e1;">
    Sağ/Sol bakış ile nokta-çizgi üret, kırpma komutlarıyla harfi onayla ve mesaj oluştur.
  </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ayarları
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    # MANUEL AKTİF/PASİF BUTONU
    st.subheader("🚀 Hızlı Başlat")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ AKTİF ET", use_container_width=True, type="primary"):
            with STATE.lock:
                STATE.active = True
                STATE.last_event = "Manuel olarak AKTİF edildi"
    with col_btn2:
        if st.button("⏹️ PASİF YAP", use_container_width=True):
            with STATE.lock:
                STATE.active = False
                STATE.morse = ""
                STATE.last_event = "Manuel olarak PASİF yapıldı"
    
    # Mevcut durum göstergesi
    with STATE.lock:
        is_active = STATE.active
    if is_active:
        st.success("✅ Sistem AKTİF - Bakışlarınız algılanıyor")
    else:
        st.warning("⛔ Sistem PASİF - Butona basın veya 5x göz kırpın")
    
    st.divider()
    
    # Mapping ayarı
    mapping = st.toggle("Sağ = Nokta (.)  |  Sol = Çizgi (-)", value=True)
    st.session_state["mapping_right_dot"] = mapping
    
    st.divider()
    st.subheader("🎯 Kalibrasyon")
    
    # Eşik ayarları
    left_thr = st.slider("Sol Eşik", 0.20, 0.50, 0.40, 0.01)
    right_thr = st.slider("Sağ Eşik", 0.50, 0.80, 0.60, 0.01)
    ear_thr = st.slider("EAR Eşiği (Göz Kırpma)", 0.10, 0.35, 0.21, 0.01, 
                       help="Düşük = daha hassas kırpma algılama")
    
    with STATE.lock:
        STATE.ratio_left = float(left_thr)
        STATE.ratio_right = float(right_thr)
        STATE.ear_threshold = float(ear_thr)
    
    # Debug bilgileri
    st.divider()
    st.subheader("📊 Debug Bilgileri")
    with STATE.lock:
        st.write(f"**Gaze Ratio:** {STATE.current_gaze_ratio:.3f}")
        st.write(f"**EAR:** {STATE.current_ear:.3f}")
        st.write(f"**Blink Sayacı:** {STATE.blink_count}")
        st.write(f"**Son Yön:** {STATE.last_dir}")
    
    st.divider()
    st.subheader("⌨️ Komutlar")
    st.markdown("""
- **5 kırpma**: Sistemi Aç/Kapat  
- **Sağ/Sol bak (0.4 sn)**: Nokta/Çizgi ekle  
- **2 kırpma**: Harfi onayla  
- **3 kırpma**: Boşluk ekle
    """)

    st.divider()
    st.subheader("🗑️ Mesaj Kontrol")
    if st.button("↩️ Son karakteri sil", use_container_width=True):
        with STATE.lock:
            STATE.text = STATE.text[:-1]
            STATE.last_event = "Son karakter silindi"

    if st.button("🧹 Morse tamponunu temizle", use_container_width=True):
        with STATE.lock:
            STATE.morse = ""
            STATE.last_event = "Morse temizlendi"

    if st.button("🧾 Tüm mesajı temizle", use_container_width=True):
        with STATE.lock:
            STATE.text = ""
            STATE.morse = ""
            STATE.last_event = "Her şey temizlendi"

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
    st.markdown("### 🎥 Kamera (Canlı)")
    st.info("💡 Kamera izni verdikten sonra, önce **AKTİF ET** butonuna basın veya **5 kere göz kırpın**.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="morse-eye",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MorseEyeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 📌 Çıktı Paneli")
    
    # Durumu göster
    with STATE.lock:
        active = STATE.active
        morse = STATE.morse
        text = STATE.text
        last_event = STATE.last_event
        current_char = decode_morse(morse) if morse else ""

    # Durum kartı
    if active:
        st.success("### ✅ SİSTEM AKTİF")
    else:
        st.error("### ⛔ SİSTEM PASİF")
    
    # Morse gösterimi - BÜYÜK
    st.markdown("#### 📟 Morse Tamponu:")
    morse_html = f'<div class="morse-display">{morse if morse else "---"}</div>'
    st.markdown(morse_html, unsafe_allow_html=True)
    
    # Anlık harf
    if morse:
        st.markdown(f"#### 🔤 Anlık Çözüm: **{current_char}**")
    
    # Mesaj
    st.markdown("#### 💬 Oluşturulan Mesaj:")
    text_html = f'<div class="big-text">{text if text else "(Henüz mesaj yok)"}</div>'
    st.markdown(text_html, unsafe_allow_html=True)
    
    # Son olay
    if last_event:
        st.info(f"📢 {last_event}")
    
    # Yenileme butonu (Streamlit state güncellemesi için)
    if st.button("🔄 Paneli Yenile", use_container_width=True):
        st.rerun()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b;'>"
    "MORSE-EYE • TÜBİTAK 2204-A Demo • Göz takibi ile iletişim sistemi"
    "</div>",
    unsafe_allow_html=True,
)
