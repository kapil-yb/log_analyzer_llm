# local_chat_ui.py
# -------------------------------------------------------------------
# Streamlit Chat UI for interacting with local Ollama LLMs like LLaMA3
# with selectable personas (e.g., Yugabyte Expert, Linux Expert, etc.)
# and context-specific log analysis instruction dropdowns per message.
#
# Features:
# - Uses `ollama.chat` for LLM interaction
# - Selectable LLM model (mistral, llama3, dolphin-mistral)
# - Persona-based system prompts
# - Message-specific instruction type (e.g., JSON format, RCA-style, etc.)

#
# Requirements:
# pip install streamlit
# Start Ollama locally (https://ollama.com/download)
# Run: `streamlit run local_chat_ui.py`
# -------------------------------------------------------------------

# local_chat_ui.py
# -------------------------------------------------------------------
# Streamlit Chat UI - Dark Theme Version
# -------------------------------------------------------------------

import streamlit as st
import ollama

# --------------------------
# Config: Persona Prompts
# --------------------------

PERSONA_PROMPTS = {
    "SQL Developer": "You are an expert SQL developer. Always provide complete SQL queries without using ellipses (\"...\"). Do not summarize or skip lines. Respond in full, clearly, and formally.",
    "Linux Expert": "You are a Linux system expert. Your answers should be detailed, technically accurate, and complete. Avoid using ellipses (\"...\"). Respond in a formal tone.",
    "Yugabyte Expert": "You are a knowledgeable support engineer specialized in YugabyteDB. Always give full, formal responses without skipping steps or using ellipses (\"...\").",
    "Networking Expert": "You are a networking expert. Provide complete and technically accurate answers. Never use ellipses (\"...\") or skip information.",
    "SSL Expert": "You are an SSL and security protocol expert. Be detailed and formal. Never summarize with ellipses (\"...\").",
    "Custom": "You are a highly knowledgeable assistant. Please assist with detailed and formal responses. Never use ellipses (\"...\") to skip content."
}

# --------------------------
# Config: Log Task Prompts
# --------------------------

TASK_INSTRUCTIONS = {
    "Do your best": "",
    "Format logs in JSON": (
        "Please return the log entry in JSON format which is easy to read and understand. No need to add any logs explanation\n"
    ),
    "Explain the log in detail": (
        "Provide a comprehensive, line-by-line explanation of the provided log entries.\n"
        "Describe what each log line indicates about the system's state and actions at that specific time.\n"
        "Pay attention to timestamps, source components (e.g., application names, services), log levels (e.g., INFO, WARNING, ERROR), and any specific messages or error codes.\n"
        "Trace the sequence of events as indicated by the logs.\n"
        "Identify any patterns, anomalies, or recurring messages.\n"
        "Explain the significance of any specific parameters or values present in the log lines."
    )
}

# --------------------------
# Dark Theme CSS Overrides
# --------------------------

st.markdown("""
<style>
    /* 1. Dark theme base */
    :root {
        --primary-bg: #0e1117;
        --secondary-bg: #1a1d24;
        --text-color: #f0f0f0;
        --border-color: #2d3748;
    }
    
    /* 2. Complete reset and dark background */
    html, body, #root, .stApp {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 100vh !important;
        background: var(--primary-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* 3. Chat message area */
    .chat-area {
        padding: 15px;
        padding-bottom: 160px !important;
        background: var(--primary-bg);
    }
    
    /* 4. Fixed input container */
    .input-container {
        position: fixed !important;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--secondary-bg);
        padding: 15px;
        border-top: 1px solid var(--border-color);
        box-shadow: 0 -2px 15px rgba(0,0,0,0.3);
        z-index: 1000;
    }
    
    /* 5. Hide all Streamlit decorative elements */
    footer, .stDecoration, hr {
        display: none !important;
    }
    
    /* 6. Dark theme for all components */
    .stTextInput, .stSelectbox, .stTextArea, .stButton>button {
        background: var(--secondary-bg) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* 7. Chat message styling */
    .stChatMessage {
        background: var(--secondary-bg) !important;
        border-color: var(--border-color) !important;
    }
    
    /* 8. Sidebar dark theme */
    .css-1d391kg, .st-emotion-cache-1cypcdb {
        background: var(--secondary-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar Configuration
# --------------------------

st.sidebar.title("🧠 Persona & Model Settings")
persona = st.sidebar.selectbox("Select Persona", list(PERSONA_PROMPTS.keys()), index=2)

if persona == "Custom":
    custom_prompt = st.sidebar.text_area("Enter your custom system prompt:", value=PERSONA_PROMPTS["Custom"], height=150)
else:
    custom_prompt = PERSONA_PROMPTS[persona]

model_name = st.sidebar.selectbox("Select Model", ["llama3", "mistral", "dolphin-mistral"])
clear_chat = st.sidebar.button("🧹 Clear Chat History")

# --------------------------
# Session State Initialization
# --------------------------

if clear_chat or "messages" not in st.session_state:
    st.session_state.messages = []

# Update system prompt
if not st.session_state.messages or st.session_state.messages[0]["role"] != "system":
    st.session_state.messages.insert(0, {"role": "system", "content": custom_prompt})
else:
    st.session_state.messages[0]["content"] = custom_prompt

# --------------------------
# Main Chat UI
# --------------------------

st.title("🤖 Local Log Expert (via Ollama)")

# Chat Messages Area
with st.container():
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    
    # Display messages (newest at bottom)
    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fixed Input Area
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    cols = st.columns([4, 1.5])
    with cols[0]:
        user_input = st.chat_input("Type your message here...", key="main_input")
    with cols[1]:
        selected_instruction = st.selectbox(
            "Task Type",
            list(TASK_INSTRUCTIONS.keys()),
            index=0,
            key="main_instruction"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Message Handling
# --------------------------

if user_input:
    full_prompt = user_input
    if TASK_INSTRUCTIONS[selected_instruction]:
        full_prompt += f"\n\nAdditional Guidance:\n{TASK_INSTRUCTIONS[selected_instruction]}"

    st.session_state.messages.append({"role": "user", "content": full_prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.spinner("Generating response..."):
        response = ollama.chat(
            model=model_name,
            messages=st.session_state.messages,
            options={"temperature": 0.2, "num_predict": 2048},
        )
        reply = response["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Force UI update
    st.rerun()

# --------------------------
# JavaScript for Final Cleanup
# --------------------------

st.markdown("""
<script>
    // Final dark theme enforcement
    document.addEventListener('DOMContentLoaded', function() {
        // Ensure all text is visible
        document.querySelectorAll('*').forEach(el => {
            el.style.color = '#f0f0f0';
        });
        
        // Remove any light backgrounds
        document.querySelectorAll('.stApp, .stChatMessage').forEach(el => {
            el.style.backgroundColor = '#0e1117';
        });
    });
</script>
""", unsafe_allow_html=True)
