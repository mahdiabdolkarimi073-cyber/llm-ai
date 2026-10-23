"""ChatGPT-style chat interface built with NiceGUI.

Features:
  - Persian, RTL, dark-mode, responsive design
  - Sidebar: new chat, history, delete chats
  - User/model message bubbles with code formatting
  - Multi-line input: Enter to send, Shift+Enter for newline
  - Streaming token-by-token generation, stop button
  - Copy response/code, auto-scroll, real model status
  - Thread-safe generation (UI never freezes)
  - Port 8080 (avoids 5173 and 8000)

Usage:
    pip install nicegui torch
    python llm/chat_app.py
"""

import os
import sys
import json
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nicegui import ui, app

from chat_model import ChatModelEngine, ModelLoadError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAT_PORT = 8080
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(HISTORY_DIR, "conversations.json")

# ---------------------------------------------------------------------------
# Global model engine
# ---------------------------------------------------------------------------
engine = ChatModelEngine()

# ---------------------------------------------------------------------------
# Conversation storage
# ---------------------------------------------------------------------------

def load_conversations() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_conversations(conversations: list[dict]):
    tmp = HISTORY_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# CSS: modern, minimal, dark, RTL, Persian
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
  --bg-primary: #0f0f0f;
  --bg-secondary: #1a1a1a;
  --bg-tertiary: #242424;
  --bg-input: #2a2a2a;
  --text-primary: #e8e8e8;
  --text-secondary: #a0a0a0;
  --text-muted: #6b6b6b;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --accent-light: #60a5fa;
  --user-bubble: #2563eb;
  --model-bubble: #242424;
  --border: #333333;
  --error: #ef4444;
  --success: #22c55e;
  --warning: #f59e0b;
  --code-bg: #1e1e1e;
  --code-border: #383838;
}

* { box-sizing: border-box; }

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Vazirmatn', 'Segoe UI', Tahoma, sans-serif;
  margin: 0;
  padding: 0;
  direction: rtl;
  height: 100vh;
  overflow: hidden;
}

/* Main layout */
.app-container {
  display: flex;
  height: 100vh;
  width: 100vw;
}

/* Sidebar */
.sidebar {
  width: 280px;
  min-width: 280px;
  background: var(--bg-secondary);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}

.sidebar-header h2 {
  font-size: 16px;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary);
}

.sidebar-new-chat {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}

.sidebar-history {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.sidebar-footer {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--text-muted);
}

/* Conversation list items */
.conv-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 4px;
  transition: background 0.15s;
  font-size: 13px;
  color: var(--text-secondary);
}

.conv-item:hover {
  background: var(--bg-tertiary);
}

.conv-item.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-right: 3px solid var(--accent);
}

.conv-item-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-delete-btn {
  opacity: 0;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  font-size: 14px;
  transition: opacity 0.15s, color 0.15s;
}

.conv-item:hover .conv-delete-btn {
  opacity: 1;
}

.conv-delete-btn:hover {
  color: var(--error);
}

/* Chat area */
.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
}

.chat-header {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-secondary);
}

.chat-header-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.loaded {
  background: rgba(34, 197, 94, 0.15);
  color: var(--success);
}

.status-badge.unloaded {
  background: rgba(239, 68, 68, 0.15);
  color: var(--error);
}

.status-badge.loading {
  background: rgba(245, 158, 11, 0.15);
  color: var(--warning);
}

.status-badge.generating {
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-light);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.loaded { background: var(--success); }
.status-dot.unloaded { background: var(--error); }
.status-dot.loading { background: var(--warning); animation: pulse 1s infinite; }
.status-dot.generating { background: var(--accent-light); animation: pulse 0.8s infinite; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

/* Messages area */
.messages-area {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message-bubble {
  max-width: 80%;
  padding: 14px 18px;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.7;
  word-wrap: break-word;
  position: relative;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-user {
  background: var(--user-bubble);
  color: #ffffff;
  align-self: flex-end;
  border-bottom-left-radius: 4px;
}

.message-model {
  background: var(--model-bubble);
  color: var(--text-primary);
  border: 1px solid var(--border);
  align-self: flex-start;
  border-bottom-right-radius: 4px;
}

.message-error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--error);
  color: var(--error);
  align-self: center;
  text-align: center;
  font-size: 13px;
}

.message-role-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 6px;
  font-weight: 600;
}

.message-user .message-role-label {
  color: rgba(255,255,255,0.7);
}

/* Code blocks */
.message-bubble pre {
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 8px;
  padding: 12px;
  overflow-x: auto;
  margin: 8px 0;
  font-size: 13px;
  font-family: 'Fira Code', 'Consolas', monospace;
  direction: ltr;
  text-align: left;
}

.message-bubble code {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 13px;
}

.message-bubble p {
  margin: 6px 0;
}

.message-bubble p:first-child { margin-top: 0; }
.message-bubble p:last-child { margin-bottom: 0; }

/* Copy button */
.copy-btn {
  position: absolute;
  top: 8px;
  left: 8px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 11px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
}

.message-bubble:hover .copy-btn {
  opacity: 1;
}

.copy-btn:hover {
  background: var(--accent);
  color: #fff;
}

.copy-btn.copied {
  background: var(--success);
  color: #fff;
}

/* Typing indicator */
.typing-indicator {
  display: inline-flex;
  gap: 4px;
  padding: 8px 0;
}

.typing-indicator span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--text-muted);
  animation: typingBounce 1s infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

@keyframes typingBounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}

/* Input area */
.input-area {
  padding: 16px 20px;
  border-top: 1px solid var(--border);
  background: var(--bg-secondary);
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.input-wrapper {
  flex: 1;
  position: relative;
}

.text-input {
  width: 100%;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 12px;
  color: var(--text-primary);
  padding: 12px 16px;
  font-size: 14px;
  font-family: inherit;
  resize: none;
  min-height: 48px;
  max-height: 160px;
  line-height: 1.6;
  transition: border-color 0.2s;
  direction: rtl;
}

.text-input:focus {
  outline: none;
  border-color: var(--accent);
}

.send-btn {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  font-size: 14px;
  cursor: pointer;
  font-weight: 600;
  transition: background 0.2s;
  min-width: 80px;
}

.send-btn:hover:not(:disabled) {
  background: var(--accent-hover);
}

.send-btn:disabled {
  background: var(--bg-tertiary);
  color: var(--text-muted);
  cursor: not-allowed;
}

.stop-btn {
  background: var(--error);
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  font-size: 14px;
  cursor: pointer;
  font-weight: 600;
  transition: background 0.2s;
  min-width: 80px;
}

.stop-btn:hover {
  background: #dc2626;
}

/* Scrollbar */
.messages-area::-webkit-scrollbar,
.sidebar-history::-webkit-scrollbar {
  width: 6px;
}

.messages-area::-webkit-scrollbar-track,
.sidebar-history::-webkit-scrollbar-track {
  background: transparent;
}

.messages-area::-webkit-scrollbar-thumb,
.sidebar-history::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}

.messages-area::-webkit-scrollbar-thumb:hover,
.sidebar-history::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

/* Welcome screen */
.welcome-screen {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);
  text-align: center;
}

.welcome-screen h3 {
  font-size: 22px;
  color: var(--text-secondary);
  font-weight: 600;
}

.welcome-screen p {
  font-size: 14px;
  max-width: 400px;
}

/* Model selector */
.model-selector {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 10px;
}

.model-option {
  padding: 8px 14px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.model-option:hover {
  border-color: var(--accent);
  color: var(--text-primary);
}

.model-option.active {
  border-color: var(--accent);
  background: rgba(59,130,246,0.15);
  color: var(--accent-light);
}

.model-option.unavailable {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Responsive */
@media (max-width: 768px) {
  .sidebar {
    position: fixed;
    right: 0;
    top: 0;
    z-index: 1000;
    height: 100vh;
    transform: translateX(100%);
  }
  .sidebar.open {
    transform: translateX(0);
  }
  .message-bubble {
    max-width: 92%;
  }
  .chat-header-title {
    font-size: 13px;
  }
}

/* Toggle button for mobile sidebar */
.sidebar-toggle {
  display: none;
  background: none;
  border: none;
  color: var(--text-primary);
  font-size: 20px;
  cursor: pointer;
  padding: 4px 8px;
}

@media (max-width: 768px) {
  .sidebar-toggle {
    display: block;
  }
}
</style>
"""

# ---------------------------------------------------------------------------
# Helper: escape HTML
# ---------------------------------------------------------------------------
def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def format_message_content(text: str) -> str:
    """Format message text with basic code block support."""
    import re
    escaped = escape_html(text)
    # Code blocks: ```lang\ncode``` -> <pre><code>code</code></pre>
    parts = re.split(r"```(\w*)\n(.*?)```", escaped, flags=re.DOTALL)
    if len(parts) == 1:
        # No code blocks — convert newlines to <br> and wrap in <p>
        paragraphs = escaped.split("\n\n")
        return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())

    result = []
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Regular text
            if parts[i].strip():
                paragraphs = parts[i].strip().split("\n\n")
                for p in paragraphs:
                    if p.strip():
                        result.append(f"<p>{p.replace(chr(10), '<br>')}</p>")
        elif i % 3 == 1:
            # Language tag — skip
            pass
        else:
            # Code content
            lang = parts[i - 1] if i >= 1 else ""
            code = parts[i]
            result.append(f'<pre><code>{code}</code></pre>')
        i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# UI state
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.conversations: list[dict] = load_conversations()
        self.current_conv_id: str | None = None
        self.is_generating: bool = False
        self.gen_thread: threading.Thread | None = None


state = AppState()


def get_current_conversation() -> dict | None:
    if state.current_conv_id is None:
        return None
    for conv in state.conversations:
        if conv["id"] == state.current_conv_id:
            return conv
    return None


# ---------------------------------------------------------------------------
# UI rebuild helpers
# ---------------------------------------------------------------------------

def render_sidebar_history(container):
    """Re-render the conversation list in the sidebar."""
    container.clear()
    with container:
        if not state.conversations:
            ui.label("هنوز گفت‌وگویی وجود ندارد.").classes("text-sm text-gray-500 p-3")
        for conv in state.conversations:
            item_classes = "conv-item"
            if conv["id"] == state.current_conv_id:
                item_classes += " active"
            with ui.row().classes(item_classes).on("click", lambda c=conv: select_conversation(c["id"])):
                ui.label(conv.get("title", "گفت‌وگوی جدید")).classes("conv-item-title")
                ui.button(icon="delete", on_click=lambda e, c=conv: delete_conversation(c["id"])).classes("conv-delete-btn").props("flat dense size=sm")


def select_conversation(conv_id: str):
    state.current_conv_id = conv_id
    ui.navigate.reload()


def delete_conversation(conv_id: str):
    state.conversations = [c for c in state.conversations if c["id"] != conv_id]
    if state.current_conv_id == conv_id:
        state.current_conv_id = None
    save_conversations(state.conversations)
    ui.navigate.reload()


def new_conversation():
    import uuid
    conv = {
        "id": str(uuid.uuid4())[:8],
        "title": "گفت‌وگوی جدید",
        "messages": [],
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    state.conversations.insert(0, conv)
    state.current_conv_id = conv["id"]
    save_conversations(state.conversations)
    ui.navigate.reload()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

@ui.page("/")
def main_page():
    ui.add_head_html(CUSTOM_CSS)
    ui.add_head_html('<link rel="preconnect" href="https://fonts.googleapis.com">')
    ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap" rel="stylesheet">')

    # --- Layout ---
    with ui.column().classes("app-container"):

        # --- Sidebar ---
        with ui.column().classes("sidebar") as sidebar:
            with ui.row().classes("sidebar-header"):
                ui.icon("chat", color="primary").style("font-size: 22px;")
                ui.html("<h2>دستیار هوش مصنوعی</h2>")

            with ui.column().classes("sidebar-new-chat"):
                ui.button("گفت‌وگوی جدید", icon="add", on_click=new_conversation).props("unelevated color=primary").classes("w-full")

            history_container = ui.column().classes("sidebar-history")
            render_sidebar_history(history_container)

            with ui.column().classes("sidebar-footer"):
                status_text = "مدل: بارگذاری نشده"
                ui.label(status_text).bind_text_from(
                    engine, "_loaded",
                    backward=lambda v: "مدل: بارگذاری شده" if v else "مدل: بارگذاری نشده"
                )

        # --- Chat area ---
        with ui.column().classes("chat-area"):
            # Header
            with ui.row().classes("chat-header"):
                ui.button(icon="menu", on_click=lambda: sidebar.classes(add="open")).classes("sidebar-toggle").props("flat dense")
                title_label = ui.label("").classes("chat-header-title")

                status_badge_container = ui.row().classes("status-badge-container")

            # Messages
            messages_scroll = ui.scroll_area().classes("messages-area")
            messages_container = ui.column().classes("messages-container w-full")

            # Input
            with ui.row().classes("input-area"):
                input_wrapper = ui.column().classes("input-wrapper")
                with input_wrapper:
                    text_input = ui.textarea(
                        placeholder="پیام خود را بنویسید... (Enter برای ارسال، Shift+Enter برای خط جدید)",
                    ).classes("text-input").props("autofocus")

                send_btn = ui.button("ارسال", icon="send").classes("send-btn").props("unelevated color=primary")
                stop_btn = ui.button("توقف", icon="stop").classes("stop-btn").props("unelevated color=negative")
                stop_btn.set_visibility(False)

    # --- State setup ---
    current_conv = get_current_conversation()

    # Update header title
    if current_conv:
        title_label.text = current_conv.get("title", "گفت‌وگوی جدید")
    else:
        title_label.text = "دستیار هوش مصنوعی"

    # Render status badge
    def render_status_badge():
        status_badge_container.clear()
        status_info = engine.get_status()
        with status_badge_container:
            if state.is_generating:
                with ui.row().classes("status-badge generating"):
                    ui.html('<div class="status-dot generating"></div>')
                    ui.label("در حال تولید...")
            elif status_info["loaded"]:
                with ui.row().classes("status-badge loaded"):
                    ui.html('<div class="status-dot loaded"></div>')
                    ui.label(f"مدل آماده — {status_info.get('label', '')}")
            else:
                with ui.row().classes("status-badge unloaded"):
                    ui.html('<div class="status-dot unloaded"></div>')
                    ui.label("مدل بارگذاری نشده")

    # --- Render messages or welcome screen ---
    def render_messages():
        messages_container.clear()
        with messages_container:
            if not current_conv or not current_conv.get("messages"):
                # Welcome / model selection screen
                with ui.column().classes("welcome-screen"):
                    ui.icon("hub", color="primary").style("font-size: 48px;")
                    ui.html("<h3>به دستیار هوش مصنوعی خوش آمدید</h3>")
                    ui.html("<p>یک گفت‌وگوی جدید شروع کنید و مدل خود را انتخاب کنید.</p>")

                    # Model selector
                    available = engine.list_available()
                    with ui.column().classes("model-selector"):
                        for ckpt in available:
                            cls = "model-option"
                            if not ckpt["available"]:
                                cls += " unavailable"
                            if ckpt["available"]:
                                btn = ui.button(ckpt["label"]).classes(cls)
                                btn.on("click", lambda k=ckpt["key"]: load_model_action(k))
                            else:
                                ui.button(f"{ckpt['label']} (غیرفعال)").classes(cls)
                return

            for msg in current_conv["messages"]:
                if msg["role"] == "user":
                    with ui.column().classes("message-bubble message-user"):
                        ui.html("<div class='message-role-label'>شما</div>")
                        ui.html(format_message_content(msg["content"]))
                elif msg["role"] == "model":
                    with ui.column().classes("message-bubble message-model"):
                        ui.html("<div class='message-role-label'>مدل</div>")
                        content_html = format_message_content(msg["content"])
                        ui.html(content_html)
                        # Copy button
                        copy_btn = ui.button("کپی", icon="content_copy").classes("copy-btn").props("flat dense size=sm")
                        copy_btn.on("click", lambda c=msg["content"]: copy_to_clipboard(c))
                elif msg["role"] == "error":
                    with ui.column().classes("message-bubble message-error"):
                        ui.label(msg["content"])

        # Auto-scroll to bottom
        ui.run_javascript("""
            setTimeout(function() {
                var el = document.querySelector('.messages-area');
                if (el) el.scrollTop = el.scrollHeight;
            }, 50);
        """)

    def copy_to_clipboard(text: str):
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(text)})")
        ui.notify("کپی شد!", type="positive", position="top")

    # --- Model loading ---
    def load_model_action(checkpoint_key: str):
        if state.is_generating:
            ui.notify("در حال تولید پاسخ است. صبر کنید.", type="warning", position="top")
            return
        try:
            meta = engine.load(checkpoint_key)
            ui.notify(
                f"مدل بارگذاری شد: {meta['checkpoint_label']} "
                f"({meta['params']:,} پارامتر)",
                type="positive", position="top",
            )
            render_status_badge()
            render_messages()
        except ModelLoadError as e:
            ui.notify(str(e), type="negative", position="top", multi_line=True)
            render_status_badge()

    # --- Send message ---
    def send_message():
        if state.is_generating:
            return
        text = text_input.value.strip()
        if not text:
            return
        if not engine.is_loaded:
            ui.notify("ابتدا یک مدل را بارگذاری کنید.", type="warning", position="top")
            return

        # Create conversation if none active
        nonlocal current_conv
        if current_conv is None:
            import uuid
            current_conv = {
                "id": str(uuid.uuid4())[:8],
                "title": text[:40] + ("..." if len(text) > 40 else ""),
                "messages": [],
                "created": time.strftime("%Y-%m-%d %H:%M"),
            }
            state.conversations.insert(0, current_conv)
            state.current_conv_id = current_conv["id"]
            title_label.text = current_conv["title"]

        # Add user message
        current_conv["messages"].append({"role": "user", "content": text})
        save_conversations(state.conversations)
        text_input.value = ""

        # Add placeholder model message
        model_msg = {"role": "model", "content": ""}
        current_conv["messages"].append(model_msg)

        render_messages()
        render_sidebar_history(history_container)

        # Start generation thread
        state.is_generating = True
        send_btn.set_visibility(False)
        stop_btn.set_visibility(True)
        render_status_badge()

        def generation_worker():
            full_response = ""
            try:
                for event in engine.generate_stream(
                    prompt=text,
                    max_new_tokens=engine.config.max_new_tokens if engine.config else 200,
                    temperature=engine.config.temperature if engine.config else 0.7,
                    top_k=engine.config.top_k if engine.config else 40,
                ):
                    if event["type"] == "token":
                        full_response += event["text"]
                        model_msg["content"] = full_response
                        # Update UI — re-render just the last message
                        ui.update(messages_container)

                    elif event["type"] == "done":
                        full_response = event["text"]
                        # Strip the prompt prefix from the response
                        if full_response.startswith(text):
                            full_response = full_response[len(text):]
                        model_msg["content"] = full_response
                        save_conversations(state.conversations)

                    elif event["type"] == "stopped":
                        full_response = event["text"]
                        if full_response.startswith(text):
                            full_response = full_response[len(text):]
                        model_msg["content"] = full_response + "\n\n [تولید متوقف شد]"
                        save_conversations(state.conversations)

                    elif event["type"] == "error":
                        model_msg["role"] = "error"
                        model_msg["content"] = event["text"]
                        save_conversations(state.conversations)

            except Exception as e:
                model_msg["role"] = "error"
                model_msg["content"] = f"خطای غیرمنتظره: {e}"
                save_conversations(state.conversations)
            finally:
                state.is_generating = False
                send_btn.set_visibility(True)
                stop_btn.set_visibility(False)
                render_status_badge()
                render_messages()

        state.gen_thread = threading.Thread(target=generation_worker, daemon=True)
        state.gen_thread.start()

    def stop_generation():
        engine.stop()
        ui.notify("در حال توقف...", type="warning", position="top")

    # --- Bindings ---
    send_btn.on("click", send_message)
    stop_btn.on("click", stop_generation)

    # Enter to send, Shift+Enter for newline
    text_input.on("keydown.enter", lambda e: send_message() if not e.args.get("shiftKey") else None)

    # --- Initial render ---
    render_status_badge()
    render_messages()
    render_sidebar_history(history_container)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Chat Interface for Custom LLM")
    print(f"  Port: {CHAT_PORT}")
    print(f"  Model checkpoints dir: {os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    print("=" * 60)
    print()
    print("Available checkpoints:")
    for ckpt in engine.list_available():
        status = "READY" if ckpt["available"] else "NOT FOUND"
        print(f"  [{status}] {ckpt['label']}")
        print(f"           Model: {ckpt['model_path']}")
        print(f"           Tokenizer: {ckpt['tokenizer_path']}")
    print()
    print(f"Open http://localhost:{CHAT_PORT} in your browser.")
    print()

    ui.run(host="0.0.0.0", port=CHAT_PORT, title="دستیار هوش مصنوعی", reload=False, show=False)
