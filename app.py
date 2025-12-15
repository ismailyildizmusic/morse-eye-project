# MORSE-EYE — Eye Gaze to Morse (Streamlit + WebRTC + MediaPipe)
# TÜBİTAK 2204-A için demo/prototip
# Varsayılan: SAĞ = NOKTA (.) , SOL = ÇİZGİ (-)

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

    blink_count_window: deque = field(default_factory=deque)  # timestamps
    last_blink_ts: float = 0.0

    # Kalibrasyon / eşik
    center_ratio: float = 0.5
    ratio_left: float = 0.40
    ratio_right: float = 0.60

    # hız/kararlılık
    hold_start_ts: float = 0.0
    hold_dir: str = "CENTER"
    last_symbol_ts: float = 0.0


STATE = SharedState()


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
    # Tarayıcıda SpeechSynthesis ile seslendirir (server bağımsız)
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    components.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance(`{safe}`);
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
            refine_landmarks=True,  # iris noktaları için önemli
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Blink tespiti için eşik (panelden ayarlanabilir)
        self.ear_thresh = 0.20
        self.blink_min_interval = 0.25  # saniye
        self.blink_window = 1.2         # 2-3-5 kırpma komut penceresi

        # Sembol eklemek için bakışı sabit tutma
        self.dwell_time = 0.35          # saniye (sağ/solu bu kadar tutunca 1 sembol ekler)
        self.symbol_cooldown = 0.55     # saniye

        # Yukarı/aşağı komutları (opsiyonel)
        self.enable_updown = False
        self.updown_hold = 1.8

        # yön filtresi
        self.ratio_smooth = deque(maxlen=5)

        # göz landmark indeksleri (MediaPipe FaceMesh)
        self.LEFT_CORNER = (33, 133)
        self.RIGHT_CORNER = (362, 263)
        self.LEFT_LID = (159, 145)
        self.RIGHT_LID = (386, 374)

        # iris indeksleri (refine_landmarks=True ile)
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
        # göz köşeleri
        lx1, ly1 = self._landmark_xy(lm, w, h, self.LEFT_CORNER[0])
        lx2, ly2 = self._landmark_xy(lm, w, h, self.LEFT_CORNER[1])
        rx1, ry1 = self._landmark_xy(lm, w, h, self.RIGHT_CORNER[0])
        rx2, ry2 = self._landmark_xy(lm, w, h, self.RIGHT_CORNER[1])

        # iris merkezleri
        try:
            lcx, lcy = self._iris_center(lm, w, h, self.LEFT_IRIS)
            rcx, rcy = self._iris_center(lm, w, h, self.RIGHT_IRIS)
        except Exception:
            # fallback: köşelerin orta noktası (zayıf ama crash olmaz)
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2

        # ratio: iris_x, gözün sol köşesine ne kadar yakın?
        lden = max(1.0, (lx2 - lx1))
        rden = max(1.0, (rx2 - rx1))
        lr = (lcx - lx1) / lden
        rr = (rcx - rx1) / rden

        ratio = float((lr + rr) / 2.0)
        ratio = clamp(ratio, 0.0, 1.0)
        return ratio, (lcx, lcy), (rcx, rcy), (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2)

    def _eye_aspect_ratio(self, lm, w, h):
        # basit EAR: dikey / yatay (iki göz ortalaması)
        # sol
        l_up = self._landmark_xy(lm, w, h, self.LEFT_LID[0])
        l_dn = self._landmark_xy(lm, w, h, self.LEFT_LID[1])
        l_c1 = self._landmark_xy(lm, w, h, self.LEFT_CORNER[0])
        l_c2 = self._landmark_xy(lm, w, h, self.LEFT_CORNER[1])
        # sağ
        r_up = self._landmark_xy(lm, w, h, self.RIGHT_LID[0])
        r_dn = self._landmark_xy(lm, w, h, self.RIGHT_LID[1])
        r_c1 = self._landmark_xy(lm, w, h, self.RIGHT_CORNER[0])
        r_c2 = self._landmark_xy(lm, w, h, self.RIGHT_CORNER[1])

        l_v = _dist(l_up, l_dn)
        l_h = _dist(l_c1, l_c2)
        r_v = _dist(r_up, r_dn)
        r_h = _dist(r_c1, r_c2)

        l_ear = l_v / (l_h + 1e-9)
        r_ear = r_v / (r_h + 1e-9)
        return float((l_ear + r_ear) / 2.0)

    def _register_blink(self):
        now = time.time()
        with STATE.lock:
            # debounce
            if now - STATE.last_blink_ts < self.blink_min_interval:
                return
            STATE.last_blink_ts = now
            STATE.blink_count_window.append(now)
            # pencere dışını at
            while STATE.blink_count_window and (now - STATE.blink_count_window[0] > self.blink_window):
                STATE.blink_count_window.popleft()

    def _interpret_blinks_if_ready(self):
        # Son kırpmanın üzerinden biraz zaman geçtiyse komut üret
        now = time.time()
        with STATE.lock:
            if not STATE.blink_count_window:
                return None
            last = STATE.blink_count_window[-1]
            if now - last < 0.55:  # hala seri kırpma geliyor olabilir
                return None
            count = len(STATE.blink_count_window)
            STATE.blink_count_window.clear()

        # Komutlar:
        # 5 kırpma: başlat/durdur
        # 2 kırpma: harf onayla
        # 3 kırpma: boşluk
        if count >= 5:
            return "TOGGLE_ACTIVE"
        if count == 2:
            return "CONFIRM_CHAR"
        if count == 3:
            return "SPACE"
        return None

    def _update_logic(self, direction: str, mapping_right_dot: bool):
        now = time.time()

        # blink komutu var mı?
        cmd = self._interpret_blinks_if_ready()
        if cmd:
            with STATE.lock:
                if cmd == "TOGGLE_ACTIVE":
                    STATE.active = not STATE.active
                    STATE.last_event = "Sistem " + ("AKTİF ✅" if STATE.active else "PASİF ⛔")
                    STATE.morse = "" if not STATE.active else STATE.morse

                elif cmd == "CONFIRM_CHAR":
                    if STATE.morse:
                        ch = decode_morse(STATE.morse)
                        STATE.text += ch
                        STATE.last_event = f"Onay: {STATE.morse} → {ch}"
                        STATE.morse = ""
                    else:
                        STATE.last_event = "Onay: boş (morse yok)"

                elif cmd == "SPACE":
                    STATE.text += " "
                    STATE.last_event = "Boşluk eklendi"

            return

        # aktif değilse sadece yön bilgisini yazalım
        with STATE.lock:
            STATE.last_dir = direction
            active = STATE.active

        if not active:
            return

        # sağ/sol ile sembol ekleme (dwell)
        if direction in ("LEFT", "RIGHT"):
            with STATE.lock:
                if STATE.hold_dir != direction:
                    STATE.hold_dir = direction
                    STATE.hold_start_ts = now
                else:
                    held = now - STATE.hold_start_ts
                    can_add = (now - STATE.last_symbol_ts) > self.symbol_cooldown
                    if held >= self.dwell_time and can_add:
                        # mapping
                        if mapping_right_dot:
                            sym = "." if direction == "RIGHT" else "-"
                        else:
                            sym = "." if direction == "LEFT" else "-"

                        STATE.morse += sym
                        STATE.last_symbol = sym
                        STATE.last_symbol_ts = now
                        STATE.last_event = f"Sembol eklendi: {sym}"
                        # aynı bakışta tekrar eklemeyi zorlaştır
                        STATE.hold_start_ts = now + 0.15
        else:
            with STATE.lock:
                STATE.hold_dir = direction
                STATE.hold_start_ts = now

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # parametreleri ana thread’den al (state)
        with STATE.lock:
            ratio_left = STATE.ratio_left
            ratio_right = STATE.ratio_right

        # UI ayarları session_state üzerinden (webrtc thread’inde erişmek riskli),
        # bu yüzden eşikleri STATE üstünden yönetiyoruz.

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        direction = "NO_FACE"
        gaze_ratio = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # gaze
            gaze_ratio, l_iris, r_iris, l_c, r_c = self._gaze_ratio(lm, w, h)

            self.ratio_smooth.append(gaze_ratio)
            g = float(np.mean(self.ratio_smooth))

            if g < ratio_left:
                direction = "LEFT"
            elif g > ratio_right:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            # blink
            ear = self._eye_aspect_ratio(lm, w, h)
            if ear < self.ear_thresh:
                # göz kapalı say
                pass
            else:
                # göz açık; kapanma-açılma geçişini burada yakalamak için
                # basit yaklaşım: ear düşükten yükseğe çıktığında blink say
                # (daha iyi için state tutulabilir; burada pratik tutuyoruz)
                # EAR çok titreşirse yanlış sayabilir; eşik ve interval ile azaltıyoruz.
                if hasattr(self, "_prev_ear") and self._prev_ear < self.ear_thresh:
                    self._register_blink()
                self._prev_ear = ear

            # çizimler (göz köşeleri + iris)
            (lx1, ly1, lx2, ly2) = l_c
            (rx1, ry1, rx2, ry2) = r_c
            cv2.circle(img, (int(l_iris[0]), int(l_iris[1])), 2, (255, 255, 0), -1)
            cv2.circle(img, (int(r_iris[0]), int(r_iris[1])), 2, (255, 255, 0), -1)
            cv2.line(img, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (80, 80, 200), 1)
            cv2.line(img, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (80, 80, 200), 1)

        # mapping (varsayılan sağ=nokta)
        mapping_right_dot = st.session_state.get("mapping_right_dot", True)

        # mantık güncelle
        self._update_logic(direction, mapping_right_dot)

        # üst yazılar (overlay)
        with STATE.lock:
            active = STATE.active
            morse = STATE.morse
            text = STATE.text
            last_event = STATE.last_event
            last_symbol = STATE.last_symbol

        # HUD
        cv2.rectangle(img, (10, 10), (w - 10, 135), (255, 255, 255), -1)
        cv2.rectangle(img, (10, 10), (w - 10, 135), (30, 41, 59), 2)

        status = "AKTİF ✅" if active else "PASİF ⛔ (5 kırpma ile başlat)"
        cv2.putText(img, f"MORSE-EYE | {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 41, 59), 2)

        if gaze_ratio is not None:
            cv2.putText(img, f"Yon: {direction} | GazeRatio: {gaze_ratio:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 41, 59), 2)
        else:
            cv2.putText(img, f"Yon: {direction}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 41, 59), 2)

        cv2.putText(img, f"Morse: {morse}   (Son: {last_symbol})", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 41, 59), 2)
        cv2.putText(img, f"Mesaj: {text[-45:]}", (20, 127),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 41, 59), 2)

        # küçük event
        if last_event:
            cv2.putText(img, last_event[:60], (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="MORSE-EYE | TÜBİTAK 2204-A", page_icon="👁️", layout="wide")

st.markdown(
    """
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
.card { background:white; border:1px solid rgba(15,23,42,0.10);
  border-radius: 14px; padding: 14px; box-shadow: 0 6px 16px rgba(15,23,42,0.06);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header">
  <h1 style="margin:0; font-weight:900;">👁️ MORSE-EYE — Göz Hareketleri ile Mors Kod İletişimi</h1>
  <div class="badge">🏆 TÜBİTAK 2204-A • Demo Web App (GitHub + Streamlit)</div>
  <p style="margin:10px 0 0 0; color:#cbd5e1;">
    Sağ/Sol bakış ile nokta-çizgi üret, kırpma komutlarıyla harfi onayla ve mesaj oluştur.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ayarları
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")

    st.session_state["mapping_right_dot"] = st.toggle("Sağ = Nokta (.)  |  Sol = Çizgi (-)", value=True)
    st.caption("İstersen tersine çevirip sol=nokta yapabilirsin.")

    st.divider()
    st.subheader("🎯 Kalibrasyon / Eşik")
    st.write("İpucu: Kameraya karşı **tam ortaya bak** ve ışığı sabitle.")

    if st.button("📌 MERKEZ KALİBRASYONU AL (Ortaya bak)", use_container_width=True):
        # o anki center_ratio'yu son ölçümden almak için:
        # (güvenli yol: mevcut oranı sabit 0.5 yapıp, kullanıcıya eşikleri sliderla ayarlatmak)
        with STATE.lock:
            STATE.center_ratio = 0.5
            # default eşikler (orta 0.5 çevresi)
            STATE.ratio_left = 0.42
            STATE.ratio_right = 0.58

    left_thr = st.slider("Sol Eşik (ratio_left)", 0.20, 0.50, 0.42, 0.01)
    right_thr = st.slider("Sağ Eşik (ratio_right)", 0.50, 0.80, 0.58, 0.01)

    with STATE.lock:
        STATE.ratio_left = float(left_thr)
        STATE.ratio_right = float(right_thr)

    st.divider()
    st.subheader("⌨️ Komutlar")
    st.markdown(
        """
- **5 kırpma**: sistemi Aç/Kapat  
- **Sağ/Sol bak (0.35 sn)**: **Nokta / Çizgi** ekler  
- **2 kırpma**: Harfi onayla  
- **3 kırpma**: Boşluk ekle
        """
    )

    st.divider()
    st.subheader("🗑️ Mesaj Kontrol")
    if st.button("↩️ Son karakteri sil", use_container_width=True):
        with STATE.lock:
            STATE.text = STATE.text[:-1]
            STATE.last_event = "Silindi: son karakter"

    if st.button("🧹 Morse tamponunu temizle", use_container_width=True):
        with STATE.lock:
            STATE.morse = ""
            STATE.last_event = "Morse temizlendi"

    if st.button("🧾 Mesajı temizle", use_container_width=True):
        with STATE.lock:
            STATE.text = ""
            STATE.last_event = "Mesaj temizlendi"

    st.divider()
    st.subheader("🔊 Sesli Oku (Tarayıcı)")
    if st.button("▶️ Mesajı seslendir", use_container_width=True):
        with STATE.lock:
            t = STATE.text.strip()
        if t:
            speak_in_browser(t)

# --- Main layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 🎥 Kamera (Canlı) + Üstte Anlık Mors & Mesaj")
    st.info("Not: İlk çalıştırmada tarayıcı kamera izni ister. Video üstündeki beyaz panelde durum ve mesajı görürsün.")

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
    with STATE.lock:
        active = STATE.active
        morse = STATE.morse
        text = STATE.text
        last_event = STATE.last_event

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**Durum:**", "AKTİF ✅" if active else "PASİF ⛔ (5 kırpma)")
    st.write("**Morse tamponu:**", f"`{morse}`")
    st.write("**Anlık çözüm:**", decode_morse(morse) if morse else "-")
    st.write("**Mesaj:**")
    st.code(text if text else "—", language="text")
    if last_event:
        st.success(last_event)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("İpucu: Eşikler tutmazsa (yanlış sağ/sol algılıyorsa) sidebar’dan ratio_left / ratio_right ile ayarla.")


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b;'>"
    "MORSE-EYE • TÜBİTAK 2204-A Demo • Yerel çalıştırma gizlilik açısından önerilir."
    "</div>",
    unsafe_allow_html=True,
)
