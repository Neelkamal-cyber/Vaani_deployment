"""
VAANI – Real-Time Indian Sign Language Recognition
Hugging Face Spaces · Streamlit + streamlit-webrtc + gTTS
=====================================================
Files needed in your Space root:
  app.py                 ← this file
  requirements.txt
  packages.txt
  README.md
  vaani_endec_deploy.h5  ← your trained model
  label_map.json         ← your label map
"""

import base64
import json
import queue
import time
from io import BytesIO

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from gtts import gTTS
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VAANI · Sign Language AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Header ── */
.vaani-header {
    background: linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 60%,#0d1b2a 100%);
    border: 1px solid #00ff9525;
    border-radius: 16px;
    padding: 26px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.vaani-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #00ff95;
    letter-spacing: -1px;
    margin: 0;
    text-shadow: 0 0 28px #00ff9545;
}
.vaani-sub {
    color: #7777aa;
    font-size: 0.88rem;
    margin: 5px 0 0;
    letter-spacing: 0.4px;
}

/* ── Cards ── */
.sign-card {
    background: #0f0f1a;
    border: 1px solid #00ff9530;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #00ff95;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.sign-word {
    font-size: 2.3rem;
    font-weight: 600;
    color: #ffffff;
    min-height: 54px;
    line-height: 1.2;
}
.sentence-card {
    background: #0d1b2a;
    border: 1px solid #0066cc35;
    border-radius: 14px;
    padding: 20px 24px;
    min-height: 96px;
    margin-bottom: 14px;
}
.sentence-text {
    font-size: 1.1rem;
    color: #cce4ff;
    line-height: 1.75;
    font-weight: 300;
}

/* ── Stats pills ── */
.stat-pill {
    display: inline-block;
    background: #00ff9510;
    border: 1px solid #00ff9530;
    border-radius: 20px;
    padding: 4px 13px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #00ff95;
    margin: 3px 3px 3px 0;
}

/* ── Tip box ── */
.tip-box {
    background: #1a1a2e;
    border-left: 3px solid #00ff95;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 0.84rem;
    color: #8888aa;
    line-height: 1.65;
    margin-top: 12px;
}

/* ── Vocab chips ── */
.vocab-wrap { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
.vocab-chip {
    background: #1a1a2e;
    border: 1px solid #333355;
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 0.76rem;
    color: #8888aa;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    background: #0f0f1a !important;
    border: 1px solid #00ff9540 !important;
    color: #00ff95 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.76rem !important;
    border-radius: 8px !important;
    transition: all .2s !important;
}
div[data-testid="stButton"] button:hover {
    background: #00ff9515 !important;
    border-color: #00ff95 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
MODEL_PATH        = "vaani_endec_deploy.h5"
LABEL_MAP_FILE    = "label_map.json"
SEQUENCE_LENGTH   = 60
CONFIDENCE_THRESH = 0.90
MAX_SENTENCE      = 8

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL + LABELS  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳  Loading VAANI model…")
def load_resources():
    with open(LABEL_MAP_FILE) as f:
        label_map = json.load(f)
    # label_map can be  { "word": index }  or  { index: "word" }
    # Normalise to  { int_index: "WORD" }
    idx_to_word = {}
    for k, v in label_map.items():
        if isinstance(v, int):          # {"Hello": 0, ...}
            idx_to_word[v] = k
        else:                           # {"0": "Hello", ...}
            idx_to_word[int(k)] = v
    model = load_model(MODEL_PATH, compile=False)
    # Warm-up: first predict() call always takes extra time
    model.predict(np.zeros((1, SEQUENCE_LENGTH, 258)), verbose=0)
    return model, idx_to_word

model, idx_to_word = load_resources()

# ─────────────────────────────────────────────────────────────────
# MEDIAPIPE  (drawing styles)
# ─────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
_DOT  = mp_draw.DrawingSpec(color=(0, 255, 149), thickness=2, circle_radius=3)
_LINE = mp_draw.DrawingSpec(color=(0, 180, 100), thickness=1)

# ─────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_features(results) -> np.ndarray:
    """Return a shoulder-normalised 258-dim keypoint vector."""
    sx, sy = 0.0, 0.0
    if results.pose_landmarks:
        l = results.pose_landmarks.landmark[11]
        r = results.pose_landmarks.landmark[12]
        sx, sy = (l.x + r.x) / 2, (l.y + r.y) / 2

    pose = (np.array([[lm.x-sx, lm.y-sy, lm.z, lm.visibility]
                       for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33*4))
    lh   = (np.array([[lm.x-sx, lm.y-sy, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21*3))
    rh   = (np.array([[lm.x-sx, lm.y-sy, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21*3))
    return np.concatenate([pose, lh, rh])   # (258,)

# ─────────────────────────────────────────────────────────────────
# THREAD-SAFE WORD QUEUE  (video thread → UI thread)
# ─────────────────────────────────────────────────────────────────
_word_q: queue.Queue = queue.Queue()

# ─────────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH  (gTTS → base-64 MP3 → hidden <audio autoplay>)
# gTTS replaces pyttsx3/PowerShell — works on every OS including
# the Linux containers that Hugging Face Spaces runs on.
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=300)
def _word_to_b64(word: str) -> str:
    """Convert a word to a base-64 MP3 string (cached per unique word)."""
    buf = BytesIO()
    gTTS(text=word, lang="en", slow=False).write_to_fp(buf)
    return base64.b64encode(buf.getvalue()).decode()

def play_word(word: str, slot) -> None:
    """
    Inject a hidden autoplay <audio> tag into *slot*.
    The browser plays the MP3 without any server-side audio driver.
    Falls back to the browser's built-in speechSynthesis if gTTS fails
    (e.g. the Space is offline / gTTS quota hit).
    """
    try:
        b64 = _word_to_b64(word)
        slot.markdown(
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
            f'</audio>',
            unsafe_allow_html=True,
        )
    except Exception:
        safe = word.replace("'", "").replace('"', "")
        slot.empty()
        # Browser Web Speech API fallback — no internet needed
        st.components.v1.html(
            f"<script>speechSynthesis.cancel();"
            f"speechSynthesis.speak("
            f"Object.assign(new SpeechSynthesisUtterance('{safe}'),"
            f"{{rate:1,lang:'en-US'}}));</script>",
            height=0,
        )

# ─────────────────────────────────────────────────────────────────
# VIDEO PROCESSOR  (runs inside the streamlit-webrtc worker thread)
# ─────────────────────────────────────────────────────────────────
class VANIProcessor:
    """Stateful per-session video processor."""

    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.sequence:           list = []
        self.predictions:        list = []
        self.current_word:        str = ""
        self.sentence:           list = []
        self.frames_since_sign:   int = 0

    # Called once per frame by streamlit-webrtc
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # ── MediaPipe processing ──────────────────────────────
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        # ── Draw hand skeletons ───────────────────────────────
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                img, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, _DOT, _LINE,
            )
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                img, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, _DOT, _LINE,
            )

        # ── Feature extraction → sequence buffer ─────────────
        kp = extract_features(results)
        self.sequence.append(kp)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:]

        # ── Model inference ───────────────────────────────────
        if len(self.sequence) == SEQUENCE_LENGTH:
            inp  = np.expand_dims(self.sequence, axis=0)   # (1,60,258)
            res  = model.predict(inp, verbose=0)[0]
            pi   = int(np.argmax(res))
            conf = float(res[pi])

            if conf > CONFIDENCE_THRESH:
                self.predictions.append(pi)
                self.predictions = self.predictions[-15:]

                # Confirm only when majority of last 15 frames agree
                if self.predictions.count(pi) > 10:
                    word = idx_to_word.get(pi, "?")
                    if word != self.current_word:
                        self.current_word      = word
                        self.frames_since_sign = 0
                        if not self.sentence or self.sentence[-1] != word:
                            self.sentence.append(word)
                            self.sentence = self.sentence[-MAX_SENTENCE:]
                            _word_q.put(word)   # ← send to UI thread
            else:
                self.frames_since_sign += 1

        if self.frames_since_sign > 60:
            self.current_word = ""

        # ── On-frame UI overlay ───────────────────────────────
        # Top bar — current sign
        cv2.rectangle(img, (0, 0), (w, 52), (10, 10, 22), -1)
        cv2.putText(
            img,
            f"VAANI  |  {self.current_word or 'waiting...'}",
            (14, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 149), 2, cv2.LINE_AA,
        )
        # Bottom bar — sentence
        cv2.rectangle(img, (0, h-50), (w, h), (10, 28, 18), -1)
        cv2.putText(
            img,
            " ".join(self.sentence) or "start signing...",
            (14, h-16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (200, 255, 235), 2, cv2.LINE_AA,
        )
        # Live indicator dot (top-right)
        color = (0, 255, 149) if self.current_word else (70, 70, 70)
        cv2.circle(img, (w-22, 26), 9, color, -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────────
# WebRTC / STUN config  — required for NAT traversal on HF Spaces
# ─────────────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
for _k, _v in [("sentence",[]), ("last_word",""), ("word_count",0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────
# ──────────────────────────  UI  ─────────────────────────────────
# ─────────────────────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="vaani-header">
  <div style="font-size:3rem;line-height:1">🤟</div>
  <div>
    <div class="vaani-title">VAANI</div>
    <div class="vaani-sub">Real-Time Indian Sign Language → Speech &nbsp;·&nbsp;
       MediaPipe Holistic + Encoder-Decoder LSTM</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────
col_cam, col_ui = st.columns([3, 2], gap="large")

# ── LEFT — camera ─────────────────────────────────────────────────
with col_cam:
    st.markdown("#### 📷 Live Camera")
    ctx = webrtc_streamer(
        key              = "vaani",
        mode             = WebRtcMode.SENDRECV,
        rtc_configuration= RTC_CONFIG,
        video_processor_factory = VANIProcessor,
        media_stream_constraints= {
            "video": {"width": 1280, "height": 720},
            "audio": False,
        },
        async_processing = True,
    )
    st.markdown("""
    <div class="tip-box">
      <span style="color:#00ff95;font-weight:600">Tips for best results:</span><br>
      ✦ Use good, even lighting — avoid sitting with a window behind you<br>
      ✦ Keep both hands clearly visible in the frame<br>
      ✦ Hold each sign steadily for about 1.5 seconds<br>
      ✦ A plain or simple background improves accuracy
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT — dashboard ─────────────────────────────────────────────
with col_ui:

    # Placeholders updated in the polling loop below
    _slot_word  = st.empty()
    _slot_audio = st.empty()   # invisible audio element lives here
    _slot_sent  = st.empty()
    _slot_stats = st.empty()

    st.markdown("---")

    # Control buttons
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        if st.button("🗑️ Clear"):
            st.session_state.sentence   = []
            st.session_state.last_word  = ""
            st.session_state.word_count = 0
            if ctx.video_processor:
                ctx.video_processor.sentence = []
            st.rerun()
    with _c2:
        if st.button("🔄 Refresh"):
            st.rerun()
    with _c3:
        _read_all = st.button("🔊 Read All")

    # Vocabulary chips
    st.markdown("**Recognised vocabulary**")
    chips = "".join(
        f'<span class="vocab-chip">{w}</span>'
        for w in sorted(idx_to_word.values())
    )
    st.markdown(f'<div class="vocab-wrap">{chips}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# RENDER PANEL HELPER
# ─────────────────────────────────────────────────────────────────
def _render_panel():
    word = st.session_state.last_word
    _slot_word.markdown(f"""
    <div class="sign-card">
      <div class="card-label">Current Sign</div>
      <div class="sign-word">{word or "—"}</div>
    </div>""", unsafe_allow_html=True)

    sent = st.session_state.sentence
    body = " &nbsp;›&nbsp; ".join(
        f'<b style="color:#fff">{w}</b>' for w in sent
    ) if sent else '<i style="color:#44446a">Waiting for signs…</i>'
    _slot_sent.markdown(f"""
    <div class="sentence-card">
      <div class="card-label">Sentence</div>
      <div class="sentence-text">{body}</div>
    </div>""", unsafe_allow_html=True)

    _slot_stats.markdown(
        f'<span class="stat-pill">Words detected: {st.session_state.word_count}</span>'
        f'<span class="stat-pill">Accuracy: 95 %</span>'
        f'<span class="stat-pill">Enc-Dec LSTM</span>',
        unsafe_allow_html=True,
    )

# Initial render before webcam starts
_render_panel()

# ── "Read All" button handler ─────────────────────────────────────
if _read_all and st.session_state.sentence:
    play_word(" ".join(st.session_state.sentence), _slot_audio)

# ─────────────────────────────────────────────────────────────────
# LIVE POLLING LOOP
# Drains the word queue that VANIProcessor fills,
# updates session state, triggers TTS, re-renders the panel.
# st.rerun() is called every 0.8 s — the webcam video itself
# streams independently at full frame rate via WebRTC.
# ─────────────────────────────────────────────────────────────────
if ctx.state.playing and ctx.video_processor:
    new_words = []
    while True:
        try:
            new_words.append(_word_q.get_nowait())
        except queue.Empty:
            break

    if new_words:
        for w in new_words:
            if (not st.session_state.sentence
                    or st.session_state.sentence[-1] != w):
                st.session_state.sentence.append(w)
                st.session_state.word_count += 1
            st.session_state.last_word = w

        st.session_state.sentence = st.session_state.sentence[-MAX_SENTENCE:]

        # Sync sentence back into the video processor
        ctx.video_processor.sentence = list(st.session_state.sentence)

        # Speak the most recent confirmed word
        play_word(st.session_state.last_word, _slot_audio)
        _render_panel()

    time.sleep(0.8)
    st.rerun()
