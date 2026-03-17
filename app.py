"""
╔══════════════════════════════════════════════════════════╗
║         Lokit AI – Personal AI Assistant                 ║
║         Built with RAG, LangChain & Streamlit            ║
╚══════════════════════════════════════════════════════════╝

app.py  –  Complete Streamlit frontend for Lokit's AI chatbot.
The only function you need to wire in is ask_bot(user_input) → str.
Everything else is self-contained UI logic.
"""

import streamlit as st
import time
import os
import sys

# Ensure project root is on sys.path so `backend` is importable when running from frontend/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.chatbot import ask_bot

# ──────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lokit AI – Personal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────
# GLOBAL CSS  –  dark premium theme with teal accent
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Mono:wght@400;700&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2230;
    --bg-input:      #21273a;
    --accent:        #00d4aa;
    --accent-dim:    #00d4aa22;
    --accent-hover:  #00ffca;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
    --border:        #30363d;
    --user-bubble:   #1f3a5f;
    --bot-bubble:    #1c2230;
    --gradient:      linear-gradient(135deg, #00d4aa 0%, #0080ff 100%);
    --shadow:        0 8px 32px rgba(0,0,0,0.4);
    --radius:        14px;
    --font-main:     'DM Sans', sans-serif;
    --font-mono:     'Space Mono', monospace;
}

/* ── Base & App Shell ── */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background: var(--bg-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Chat message bubbles ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 18px;
    padding: 24px 32px;
    overflow-y: auto;
}

.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    animation: fadeSlideUp 0.3s ease;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.avatar.user { background: var(--gradient); }
.avatar.bot  { background: var(--bg-card); border: 1px solid var(--border); }

.bubble {
    max-width: 68%;
    padding: 13px 18px;
    border-radius: var(--radius);
    line-height: 1.65;
    font-size: 0.95rem;
    position: relative;
}
.bubble.user {
    background: var(--user-bubble);
    border: 1px solid #1f4080;
    border-bottom-right-radius: 4px;
    color: #cce3ff;
}
.bubble.bot {
    background: var(--bot-bubble);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text-primary);
}

.msg-time {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: var(--font-mono);
}
.msg-row.user  .msg-time { text-align: right; }
.msg-row.bot   .msg-time { text-align: left; }

/* ── Typing indicator ── */
.typing-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: blink 1.2s infinite;
    margin: 0 2px;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.9); }
    40%           { opacity: 1;   transform: scale(1.1); }
}

/* ── Input bar ── */
.stChatInput > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
.stChatInput > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
.stChatInput textarea {
    color: var(--text-primary) !important;
    background: transparent !important;
    font-family: var(--font-main) !important;
}
.stChatInput textarea::placeholder { color: var(--text-muted) !important; }

/* ── Buttons (quick questions) ── */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    font-size: 0.82rem !important;
    font-family: var(--font-main) !important;
    transition: all 0.2s !important;
    width: 100% !important;
    text-align: left !important;
}
.stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateX(3px) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--gradient) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    width: 100% !important;
    font-family: var(--font-main) !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.85 !important; }

/* ── Skill badge ── */
.skill-badge {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: var(--accent);
    margin: 3px 2px;
    font-family: var(--font-mono);
    transition: all 0.2s;
}
.skill-badge:hover {
    background: var(--accent-dim);
    border-color: var(--accent);
}

/* ── Section headers in sidebar ── */
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 6px 0 10px 0;
    font-family: var(--font-mono);
}

/* ── Project card ── */
.project-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.project-card:hover { border-color: var(--accent); }
.project-title {
    font-size: 0.87rem;
    font-weight: 600;
    color: var(--text-primary);
}
.project-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 3px;
    line-height: 1.5;
}

/* ── Welcome card (empty state) ── */
.welcome-card {
    text-align: center;
    padding: 60px 40px;
    max-width: 560px;
    margin: auto;
}
.welcome-icon {
    font-size: 52px;
    margin-bottom: 16px;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.06); }
}
.welcome-title {
    font-size: 1.7rem;
    font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}
.welcome-sub {
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

/* ── Header strip ── */
.header-strip {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}
.header-badge {
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-family: var(--font-mono);
}
.online-dot {
    width: 8px; height: 8px;
    background: #3fb950;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px #3fb950;
    animation: glow 2s ease-in-out infinite;
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 4px #3fb950; }
    50%       { box-shadow: 0 0 10px #3fb950; }
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 12px;
    font-size: 0.75rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
    font-family: var(--font-mono);
}
.footer span { color: var(--accent); }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 14px 0 !important;
}

/* ── Achievement item ── */
.achievement-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
}
.achievement-item:last-child { border-bottom: none; }
.achievement-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.achievement-text { font-size: 0.83rem; color: var(--text-secondary); line-height: 1.5; }
.achievement-text strong { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": "user"|"assistant", "content": str}

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ──────────────────────────────────────────────────────────
# DATA – Lokit's portfolio content
# ──────────────────────────────────────────────────────────
SKILLS = [
    "Python", "Machine Learning", "Deep Learning", "Generative AI",
    "LangChain", "RAG", "FAISS", "Streamlit", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Hugging Face", "SQL", "Git",
]

PROJECTS = [
    {
        "icon": "🤖",
        "title": "Personal AI Chatbot (RAG)",
        "desc": "LangChain + FAISS + Groq API chatbot that answers questions about Lokit using vector search.",
    },
    {
        "icon": "🔍",
        "title": "Document Q&A System",
        "desc": "Upload any PDF and get instant, cited answers powered by Ollama embeddings & LLaMA.",
    },
    {
        "icon": "🧠",
        "title": "Fine-tuned Sentiment Analyser",
        "desc": "BERT fine-tuned on domain-specific reviews; 94% accuracy on held-out test set.",
    },
    {
        "icon": "🎯",
        "title": "Object Detection Pipeline",
        "desc": "Real-time YOLOv8 detection system deployed as a Streamlit web app.",
    },
    {
        "icon": "📊",
        "title": "ML Ops Dashboard",
        "desc": "Experiment tracking, model comparison & drift monitoring dashboard.",
    },
]

ACHIEVEMENTS = [
    {
        "icon": "🏆",
        "title": "Amazon ML Summer School 2024",
        "desc": "Selected among top students nationwide for Amazon's prestigious ML programme.",
    },
    {
        "icon": "🥇",
        "title": "Hackathon Winner",
        "desc": "1st place at college-level AI Hackathon – built a real-time sign-language translator.",
    },
    {
        "icon": "📜",
        "title": "Google Data Analytics Certificate",
        "desc": "Completed the full professional certificate on Coursera with distinction.",
    },
    {
        "icon": "⭐",
        "title": "Open-Source Contributor",
        "desc": "Active contributor to LangChain & community RAG projects on GitHub.",
    },
]

QUICK_QUESTIONS = [
    "💡  What are Lokit's core skills?",
    "🛠️  What projects has he built?",
    "🎓  Tell me about Lokit's background.",
    "🏆  What are his top achievements?",
    "🚀  What are his career goals?",
    "📬  How can I contact Lokit?",
]

# Dummy resume bytes (replace with open("resume.pdf","rb").read() in production)
RESUME_BYTES = b"%PDF-1.4 placeholder - replace with real resume bytes"


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:

    # ── Profile header ──────────────────────────────────
    st.markdown("""
    <div style="padding:24px 20px 16px 20px; text-align:center;">
        <div style="width:72px;height:72px;border-radius:50%;
                    background:linear-gradient(135deg,#00d4aa,#0080ff);
                    display:flex;align-items:center;justify-content:center;
                    font-size:32px;margin:0 auto 12px auto;
                    box-shadow:0 4px 20px rgba(0,212,170,0.3);">
            L
        </div>
        <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;">Lokit Patidar</div>
        <div style="font-size:0.78rem;color:#00d4aa;margin-top:3px;font-family:'Space Mono',monospace;">
            AI / ML Engineer
        </div>
        <div style="margin-top:10px;display:flex;justify-content:center;gap:8px;">
            <span style="background:#1c2230;border:1px solid #30363d;border-radius:4px;
                         padding:2px 8px;font-size:0.72rem;color:#8b949e;">
                🎓 B.Tech CS
            </span>
            <span style="background:#1c2230;border:1px solid #30363d;border-radius:4px;
                         padding:2px 8px;font-size:0.72rem;color:#8b949e;">
                📍 India
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── About ────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem;color:#8b949e;line-height:1.7;padding-bottom:4px;">
        Lokit is a passionate AI & ML engineer specialising in
        <span style="color:#00d4aa;">Generative AI</span>,
        <span style="color:#00d4aa;">RAG pipelines</span>, and
        <span style="color:#00d4aa;">LLM applications</span>.
        He loves building intelligent systems that solve real problems.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Skills ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Skills</div>', unsafe_allow_html=True)
    badges_html = "".join(f'<span class="skill-badge">{s}</span>' for s in SKILLS)
    st.markdown(f'<div style="line-height:2;">{badges_html}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Projects ─────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Featured Projects</div>', unsafe_allow_html=True)
    for p in PROJECTS:
        st.markdown(f"""
        <div class="project-card">
            <div class="project-title">{p['icon']} {p['title']}</div>
            <div class="project-desc">{p['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Achievements ─────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Achievements</div>', unsafe_allow_html=True)
    for a in ACHIEVEMENTS:
        st.markdown(f"""
        <div class="achievement-item">
            <div class="achievement-icon">{a['icon']}</div>
            <div class="achievement-text">
                <strong>{a['title']}</strong><br>{a['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Resume download ──────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Resume</div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download Resume (PDF)",
        data=RESUME_BYTES,
        file_name="Lokit_Patidar_Resume.pdf",
        mime="application/pdf",
    )

    # ── Social links ─────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:10px;justify-content:center;padding:14px 0 8px 0;">
        <a href="https://github.com/" target="_blank"
           style="color:#8b949e;font-size:0.78rem;text-decoration:none;
                  background:#1c2230;border:1px solid #30363d;
                  border-radius:6px;padding:5px 12px;transition:all 0.2s;"
           onmouseover="this.style.borderColor='#00d4aa';this.style.color='#00d4aa';"
           onmouseout="this.style.borderColor='#30363d';this.style.color='#8b949e';">
            🐙 GitHub
        </a>
        <a href="https://linkedin.com/" target="_blank"
           style="color:#8b949e;font-size:0.78rem;text-decoration:none;
                  background:#1c2230;border:1px solid #30363d;
                  border-radius:6px;padding:5px 12px;transition:all 0.2s;"
           onmouseover="this.style.borderColor='#00d4aa';this.style.color='#00d4aa';"
           onmouseout="this.style.borderColor='#30363d';this.style.color='#8b949e';">
            💼 LinkedIn
        </a>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# MAIN CONTENT AREA
# ──────────────────────────────────────────────────────────

# ── Header strip ──────────────────────────────────────────
st.markdown("""
<div class="header-strip">
    <div class="header-title">
        <div class="online-dot"></div>
        🤖 Lokit <span style="color:#00d4aa;">AI</span>
        &nbsp;–&nbsp; Personal AI Assistant
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
        <div class="header-badge">RAG Powered</div>
        <div class="header-badge">v1.0.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Description ───────────────────────────────────────────
st.markdown("""
<div style="padding:12px 32px 0 32px;">
    <p style="font-size:0.88rem;color:#8b949e;margin:0;">
        Ask anything about <strong style="color:#e6edf3;">Lokit Patidar</strong> —
        his projects, skills, education, achievements, and career goals.
        Powered by a <span style="color:#00d4aa;">RAG pipeline</span> over his personal knowledge base.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:10px 32px !important;'>", unsafe_allow_html=True)

# ── Quick Questions ───────────────────────────────────────
st.markdown("""
<div style="padding:0 32px 4px 32px;">
    <div class="sidebar-section-title">Quick Questions</div>
</div>
""", unsafe_allow_html=True)

# Render quick-question buttons in a 3-column grid
cols = st.columns(3)
for idx, question in enumerate(QUICK_QUESTIONS):
    with cols[idx % 3]:
        if st.button(question, key=f"qq_{idx}"):
            # Strip the leading emoji + spaces from display label
            clean_q = question.split("  ", 1)[-1] if "  " in question else question
            st.session_state.pending_question = clean_q

st.markdown("<hr style='margin:4px 32px 0 32px !important;'>", unsafe_allow_html=True)

# ── Chat history display area ─────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # ── Welcome / empty state ────────────────────────
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">✨</div>
            <div class="welcome-title">Hi, I'm Lokit AI</div>
            <div class="welcome-sub">
                I'm an AI assistant that knows everything about
                <strong>Lokit Patidar</strong>. Ask me about his skills,
                projects, education, achievements, or career goals — I'm
                here to help you learn more about him!
            </div>
            <div style="margin-top:22px;font-size:0.8rem;color:#484f58;font-family:'Space Mono',monospace;">
                ↑ Click a Quick Question or type below to start
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Render each message bubble ───────────────────
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role      = msg["role"]
            content   = msg["content"]
            timestamp = msg.get("time", "")
            row_class = "user" if role == "user" else "bot"
            avatar    = "👤" if role == "user" else "🤖"

            st.markdown(f"""
            <div class="msg-row {row_class}">
                <div class="avatar {row_class}">{avatar}</div>
                <div style="max-width:70%;">
                    <div class="bubble {row_class}">{content}</div>
                    <div class="msg-time">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# HANDLE PENDING QUESTION  (from quick-question buttons)
# ──────────────────────────────────────────────────────────
if st.session_state.pending_question:
    user_text = st.session_state.pending_question
    st.session_state.pending_question = None

    timestamp = time.strftime("%H:%M")

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "time": timestamp,
    })

    # Get bot response
    with st.spinner(""):
        bot_reply = ask_bot(user_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "time": time.strftime("%H:%M"),
    })
    st.rerun()


# ──────────────────────────────────────────────────────────
# CHAT INPUT  (pinned at bottom by Streamlit)
# ──────────────────────────────────────────────────────────
user_input = st.chat_input(
    placeholder="Ask me anything about Lokit Patidar…",
)

if user_input and user_input.strip():
    timestamp = time.strftime("%H:%M")

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "time": timestamp,
    })

    # Show typing indicator briefly then fetch response
    with st.spinner("Lokit AI is thinking…"):
        bot_reply = ask_bot(user_input.strip())

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "time": time.strftime("%H:%M"),
    })

    st.rerun()


# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <span>Generative AI</span> · <span>RAG</span> ·
    <span>LangChain</span> · <span>FAISS</span> · <span>Streamlit</span>
    &nbsp;|&nbsp; © 2024 Lokit Patidar
</div>
""", unsafe_allow_html=True)

