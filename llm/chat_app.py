"""Persian RTL chat UI for the local LLM, served on port 8080."""

import html
import json
import os
import queue
import re
import threading
import time
import uuid

from nicegui import ui

from chat_model import ChatModelEngine, ModelLoadError


CHAT_PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "conversations.json")
engine = ChatModelEngine()


CSS = """
<style>
:root {
  --bg: #111213; --panel: #18191b; --panel-2: #202225; --panel-3: #292b2f;
  --line: #303238; --text: #f2f3f5; --muted: #9da1aa; --dim: #696e78;
  --blue: #4b8df8; --blue-2: #2f6fe4; --green: #35c77a; --red: #ed6a75;
}
* { box-sizing: border-box; }
html, body, #__nuxt, #q-app { width: 100%; height: 100%; margin: 0; }
body { background: var(--bg); color: var(--text); direction: rtl; overflow: hidden;
  font-family: Vazirmatn, Tahoma, Arial, sans-serif; }
.app-shell { width: 100%; height: 100vh; display: flex; flex-direction: row;
  direction: rtl; background: var(--bg); }
.sidebar { width: 286px; min-width: 286px; height: 100%; display: flex; flex-direction: column;
  background: var(--panel); border-left: 1px solid var(--line); }
.brand { height: 70px; padding: 0 22px; display: flex; align-items: center; gap: 10px;
  border-bottom: 1px solid var(--line); }
.brand-mark { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
  background: linear-gradient(145deg, #5ca1ff, #2764ca); color: white; font-size: 17px; }
.brand-title { font-size: 15px; font-weight: 700; color: var(--text); }
.sidebar-content { flex: 1; min-height: 0; display: flex; flex-direction: column; padding: 14px; gap: 16px; }
.new-chat { height: 44px; border: 1px solid #3b78d7; border-radius: 10px; background: var(--blue-2);
  color: white; font-family: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }
.new-chat:hover { background: var(--blue); }
.section-label { color: var(--dim); font-size: 11px; font-weight: 600; padding: 0 8px; }
.history { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
.history-empty { color: var(--dim); font-size: 12px; padding: 16px 8px; text-align: center; }
.history-row { min-height: 42px; display: flex; align-items: center; gap: 4px; padding: 0 8px;
  border-radius: 9px; color: var(--muted); cursor: pointer; }
.history-row:hover, .history-row.active { background: var(--panel-3); color: var(--text); }
.history-row.active { box-shadow: inset -2px 0 var(--blue); }
.history-title { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-size: 12px; }
.history-delete { width: 26px; height: 26px; border: 0; border-radius: 6px; background: transparent;
  color: var(--dim); cursor: pointer; opacity: 0; }
.history-row:hover .history-delete { opacity: 1; }
.history-delete:hover { color: var(--red); background: #3b272b; }
.model-panel { border-top: 1px solid var(--line); padding-top: 14px; }
.model-select { width: 100%; margin-top: 8px; padding: 10px; border: 1px solid var(--line); border-radius: 9px;
  background: var(--panel-2); color: var(--text); font: 12px Vazirmatn; }
.model-select option { background: var(--panel-2); }
.model-state { margin-top: 9px; color: var(--muted); font-size: 11px; line-height: 1.8; }
.status-dot { display: inline-block; width: 7px; height: 7px; margin-left: 5px; border-radius: 50%; background: var(--dim); }
.status-dot.ready { background: var(--green); box-shadow: 0 0 0 3px #1d3c2b; }
.status-dot.busy { background: #e7aa45; animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity: .35; } }
.sidebar-footer { padding: 13px 22px; border-top: 1px solid var(--line); color: var(--dim); font-size: 10px; }
.chat-main { min-width: 0; flex: 1; height: 100%; display: flex; flex-direction: column; background: var(--bg); }
.chat-header { height: 70px; min-height: 70px; padding: 0 28px; display: flex; align-items: center;
  justify-content: space-between; border-bottom: 1px solid var(--line); background: rgba(24,25,27,.86); }
.chat-heading { min-width: 0; display: flex; align-items: center; gap: 10px; }
.chat-title { margin: 0; font-size: 15px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chat-subtitle { color: var(--muted); font-size: 11px; }
.header-status { display: flex; align-items: center; color: var(--muted); font-size: 11px; white-space: nowrap; }
.messages-scroll { flex: 1; min-height: 0; width: 100%; background: var(--bg); }
.messages-inner { width: min(920px, calc(100% - 48px)); min-height: 100%; margin: 0 auto; padding: 34px 0 30px;
  display: flex; flex-direction: column; gap: 22px; }
.welcome { flex: 1; min-height: 420px; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; color: var(--muted); gap: 12px; }
.welcome-icon { width: 58px; height: 58px; border-radius: 18px; display: grid; place-items: center;
  background: #1d3155; color: #75aaff; font-size: 27px; }
.welcome-title { margin: 6px 0 0; color: var(--text); font-size: 22px; }
.welcome p { margin: 0; max-width: 480px; font-size: 13px; line-height: 1.9; }
.message-line { width: 100%; display: flex; }
.message-line.user { justify-content: flex-start; }
.message-line.model, .message-line.error { justify-content: flex-end; }
.message { max-width: min(790px, 88%); padding: 15px 18px; border-radius: 14px; line-height: 1.85; font-size: 14px;
  overflow-wrap: anywhere; position: relative; }
.message.user { background: var(--blue-2); color: white; border-bottom-right-radius: 4px; }
.message.model { background: var(--panel-2); border: 1px solid var(--line); color: var(--text); border-bottom-left-radius: 4px; }
.message.error { background: #3a2024; border: 1px solid #804049; color: #ffadb3; }
.role { margin-bottom: 7px; color: rgba(255,255,255,.66); font-size: 10px; font-weight: 700; }
.model .role { color: var(--muted); }
.message p { margin: 0 0 9px; }
.message p:last-child { margin-bottom: 0; }
.message pre { direction: ltr; text-align: left; overflow-x: auto; margin: 12px 0 2px; padding: 14px; border-radius: 9px;
  background: #111315; border: 1px solid #3a3d43; color: #e7e9ed; font: 13px/1.65 Consolas, "Fira Code", monospace; white-space: pre; }
.copy { position: absolute; left: 8px; top: 8px; padding: 4px 8px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--panel-3); color: var(--muted); font: 10px Vazirmatn; cursor: pointer; opacity: 0; }
.message:hover .copy { opacity: 1; }
.copy:hover { color: white; background: var(--blue-2); }
.composer-wrap { padding: 0 24px 22px; background: var(--bg); }
.composer { width: min(920px, 100%); margin: 0 auto; display: flex; align-items: flex-end; gap: 9px; padding: 8px;
  background: var(--panel-2); border: 1px solid var(--line); border-radius: 15px; box-shadow: 0 9px 30px rgba(0,0,0,.22); }
.composer textarea { flex: 1; min-height: 46px; max-height: 170px; padding: 11px 12px; resize: none; border: 0; outline: 0;
  background: transparent; color: var(--text); font: 13px/1.7 Vazirmatn; direction: rtl; }
.composer textarea::placeholder { color: var(--dim); }
.send, .stop { width: 42px; height: 42px; flex: 0 0 42px; border: 0; border-radius: 11px; color: white; cursor: pointer; font-size: 17px; }
.send { background: var(--blue-2); }.send:hover { background: var(--blue); }.stop { background: #9e3541; }.stop:hover { background: #c44755; }
.composer-hint { width: min(920px, 100%); margin: 8px auto 0; color: var(--dim); text-align: center; font-size: 10px; }
@media (max-width: 760px) {
  .sidebar { width: 230px; min-width: 230px; }
  .chat-header { padding: 0 15px; }.messages-inner { width: calc(100% - 24px); padding-top: 22px; }
  .message { max-width: 94%; }.composer-wrap { padding: 0 10px 12px; }.composer-hint { display: none; }
}
@media (max-width: 560px) {
  .sidebar { width: 205px; min-width: 205px; }.brand { padding: 0 14px; }.sidebar-content { padding: 10px; }
  .chat-subtitle { display: none; }.header-status { font-size: 10px; }.message { font-size: 13px; }
}
</style>
"""


def load_conversations() -> list[dict]:
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_conversations(conversations: list[dict]) -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    temporary = HISTORY_FILE + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(conversations, file, ensure_ascii=False, indent=2)
    os.replace(temporary, HISTORY_FILE)


def format_message_content(text: str) -> str:
    escaped = html.escape(text)
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    result: list[str] = []
    cursor = 0
    for match in pattern.finditer(escaped):
        result.append(_format_plain(escaped[cursor:match.start()]))
        result.append(f"<pre><code>{match.group(2)}</code></pre>")
        cursor = match.end()
    result.append(_format_plain(escaped[cursor:]))
    return "".join(part for part in result if part)


def _format_plain(text: str) -> str:
    if not text.strip():
        return ""
    return "".join(
        f"<p>{html.escape(part) if False else part.replace(chr(10), '<br>')}</p>"
        for part in text.split("\n\n") if part.strip()
    )


@ui.page("/")
def main_page() -> None:
    ui.add_head_html(CSS)
    conversations = load_conversations()
    current_id: str | None = None
    current_conv: dict | None = None
    events: queue.Queue[dict] = queue.Queue()
    generating = False
    worker: threading.Thread | None = None

    with ui.row().classes("app-shell"):
        with ui.column().classes("sidebar"):
            with ui.row().classes("brand"):
                ui.html('<div class="brand-mark">✦</div>')
                ui.label("دستیار هوش مصنوعی").classes("brand-title")
            with ui.column().classes("sidebar-content"):
                new_button = ui.button("گفت‌وگوی جدید  ＋").classes("new-chat")
                ui.label("گفت‌وگوهای اخیر").classes("section-label")
                history = ui.column().classes("history")
                with ui.column().classes("model-panel"):
                    ui.label("مدل محلی").classes("section-label")
                    model_select = ui.select({}, value=None).classes("model-select")
                    model_state = ui.label("").classes("model-state")
            ui.label("فقط اجرای محلی؛ بدون API خارجی").classes("sidebar-footer")

        with ui.column().classes("chat-main"):
            with ui.row().classes("chat-header"):
                with ui.column().classes("chat-heading"):
                    heading = ui.label("گفت‌وگوی جدید").classes("chat-title")
                    subtitle = ui.label("مدل زبانی اختصاصی شما").classes("chat-subtitle")
                header_status = ui.label("").classes("header-status")

            with ui.scroll_area().classes("messages-scroll"):
                messages = ui.column().classes("messages-inner")

            with ui.column().classes("composer-wrap"):
                with ui.row().classes("composer"):
                    text_input = ui.textarea(placeholder="پیام خود را بنویسید...").props("autofocus")
                    send_button = ui.button(icon="arrow_upward").classes("send")
                    stop_button = ui.button(icon="stop").classes("stop")
                ui.label("Enter برای ارسال · Shift+Enter برای خط جدید").classes("composer-hint")

    def find_conversation(conversation_id: str | None) -> dict | None:
        return next((item for item in conversations if item.get("id") == conversation_id), None)

    def render_history() -> None:
        history.clear()
        with history:
            if not conversations:
                ui.label("گفت‌وگویی ثبت نشده است.").classes("history-empty")
            for item in conversations:
                with ui.row().classes("history-row" + (" active" if item.get("id") == current_id else "")):
                    title = item.get("title", "گفت‌وگوی جدید")
                    ui.label(title).classes("history-title").on("click", lambda _event, cid=item["id"]: select_conversation(cid))
                    ui.button(icon="delete", on_click=lambda _event, cid=item["id"]: delete_conversation(cid)).classes("history-delete").props("flat dense")

    def render_model_choices() -> None:
        choices = {item["key"]: item["label"] for item in engine.list_available() if item["available"]}
        model_select.options = choices
        model_select.update()
        status = engine.get_status()
        if status.get("loaded"):
            model_select.value = status["checkpoint"]
            model_state.text = f"● آماده · {status['context_length']} توکن زمینه · {status['params']:,} پارامتر"
        else:
            model_state.text = "● مدل بارگذاری نشده است"
        model_select.update()

    def render_messages() -> None:
        messages.clear()
        with messages:
            if current_conv is None or not current_conv.get("messages"):
                with ui.column().classes("welcome"):
                    ui.html('<div class="welcome-icon">✦</div>')
                    ui.label("چطور می‌توانم کمک کنم؟").classes("welcome-title")
                    ui.label("یک checkpoint محلی را از نوار کناری انتخاب کنید، سپس سؤال یا درخواست کدنویسی خود را بنویسید.")
                return
            for item in current_conv["messages"]:
                role = item.get("role")
                if role not in {"user", "model", "error"}:
                    continue
                line_class = "user" if role == "user" else "error" if role == "error" else "model"
                with ui.row().classes(f"message-line {line_class}"):
                    with ui.column().classes(f"message {line_class}"):
                        label = "شما" if role == "user" else "خطا" if role == "error" else "مدل محلی"
                        ui.label(label).classes("role")
                        ui.html(format_message_content(item.get("content", "")))
                        if role == "model" and item.get("content"):
                            ui.button("کپی", on_click=lambda _event, value=item["content"]: copy_message(value)).classes("copy").props("flat dense")

    def copy_message(value: str) -> None:
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(value, ensure_ascii=False)})")
        ui.notify("متن پاسخ کپی شد.", type="positive", position="top")

    def select_conversation(conversation_id: str) -> None:
        nonlocal current_id, current_conv
        if generating:
            ui.notify("تا پایان تولید پاسخ امکان جابه‌جایی وجود ندارد.", type="warning", position="top")
            return
        current_id = conversation_id
        current_conv = find_conversation(current_id)
        heading.text = current_conv.get("title", "گفت‌وگوی جدید") if current_conv else "گفت‌وگوی جدید"
        render_history()
        render_messages()

    def new_conversation() -> None:
        nonlocal current_id, current_conv
        if generating:
            return
        current_id = None
        current_conv = None
        heading.text = "گفت‌وگوی جدید"
        render_history()
        render_messages()

    def delete_conversation(conversation_id: str) -> None:
        nonlocal current_id, current_conv
        if generating:
            return
        conversations[:] = [item for item in conversations if item.get("id") != conversation_id]
        save_conversations(conversations)
        if current_id == conversation_id:
            current_id = None
            current_conv = None
            heading.text = "گفت‌وگوی جدید"
        render_history()
        render_messages()

    def load_selected_model(event) -> None:
        key = event.value
        if not key or generating:
            return
        try:
            metadata = engine.load(key)
            header_status.text = f"● آماده · {metadata['checkpoint_label']}"
            ui.notify("مدل محلی بارگذاری شد.", type="positive", position="top")
        except ModelLoadError as error:
            header_status.text = "● خطا در بارگذاری مدل"
            ui.notify(str(error), type="negative", position="top", multi_line=True)
        render_model_choices()

    def send_message() -> None:
        nonlocal current_id, current_conv, generating, worker
        if generating:
            return
        text = (text_input.value or "").strip()
        if not text:
            return
        if not engine.is_loaded:
            ui.notify("ابتدا یک checkpoint محلی را انتخاب کنید.", type="warning", position="top")
            return
        if current_conv is None:
            current_conv = {"id": uuid.uuid4().hex[:8], "title": text[:42], "messages": [], "created": time.strftime("%Y-%m-%d %H:%M")}
            current_id = current_conv["id"]
            conversations.insert(0, current_conv)
            heading.text = current_conv["title"]
        current_conv["messages"].append({"role": "user", "content": text})
        model_message = {"role": "model", "content": ""}
        current_conv["messages"].append(model_message)
        save_conversations(conversations)
        text_input.value = ""
        text_input.update()
        render_history()
        render_messages()
        generating = True
        send_button.set_visibility(False)
        stop_button.set_visibility(True)
        header_status.text = "● در حال تولید پاسخ..."

        def generate() -> None:
            for event in engine.generate_stream(text, max_new_tokens=120, temperature=0.35, top_k=20):
                events.put(event)

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()

    def stop_generation() -> None:
        if generating:
            engine.stop()
            ui.notify("درخواست توقف ارسال شد.", type="warning", position="top")

    def poll_generation() -> None:
        nonlocal generating
        changed = False
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if current_conv is None:
                continue
            target = current_conv["messages"][-1]
            event_type = event.get("type")
            if event_type in {"token", "done", "stopped"}:
                target["content"] = event.get("text", "")
                if event_type == "stopped":
                    target["content"] += "\n\n[تولید متوقف شد]"
                if event_type in {"done", "stopped"}:
                    generating = False
                    save_conversations(conversations)
            elif event_type == "error":
                target["role"] = "error"
                target["content"] = event.get("text", "خطای نامشخص")
                generating = False
                save_conversations(conversations)
        if changed:
            render_messages()
            if not generating:
                send_button.set_visibility(True)
                stop_button.set_visibility(False)
                header_status.text = "● آماده"

    new_button.on("click", new_conversation)
    send_button.on("click", send_message)
    stop_button.on("click", stop_generation)
    model_select.on_value_change(load_selected_model)
    text_input.on("keydown.enter", lambda event: send_message() if not event.args.get("shiftKey", False) else None)
    ui.timer(0.08, poll_generation)

    initial = conversations[0] if conversations else None
    if initial:
        current_id = initial.get("id")
        current_conv = initial
        heading.text = initial.get("title", "گفت‌وگوی جدید")
    render_history()
    render_model_choices()
    render_messages()
    header_status.text = "● مدل بارگذاری نشده"


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=CHAT_PORT, title="دستیار هوش مصنوعی", reload=False, show=False)
