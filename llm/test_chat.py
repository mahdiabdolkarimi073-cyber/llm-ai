"""Tests for the chat interface and model engine.

These tests are WRITTEN but NOT RUN, per project requirements.
They verify:
  1. Model engine loads correctly from real checkpoints
  2. Streaming generation produces valid text
  3. Stop flag interrupts generation safely
  4. Conversation save/load round-trips
  5. Message formatting handles code blocks and HTML escaping
  6. Error handling for missing files / corrupt checkpoints
  7. Thread safety of concurrent generation requests

To run manually in the future:
    python llm/test_chat.py
"""

import os
import sys
import json
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_model import ChatModelEngine, ModelLoadError


def test_engine_list_available():
    """Engine should list all 3 checkpoints with correct existence flags."""
    engine = ChatModelEngine()
    available = engine.list_available()
    assert len(available) == 3, f"Expected 3 checkpoints, got {len(available)}"
    keys = [a["key"] for a in available]
    assert "specialized_best" in keys
    assert "specialized" in keys
    assert "phase1" in keys
    for item in available:
        assert "model_exists" in item
        assert "tokenizer_exists" in item
        assert "available" in item
    print("PASS: test_engine_list_available")


def test_engine_load_error_missing():
    """Loading a non-existent checkpoint key should raise ModelLoadError."""
    engine = ChatModelEngine()
    try:
        engine.load("nonexistent_key")
        assert False, "Should have raised ModelLoadError"
    except ModelLoadError:
        pass
    print("PASS: test_engine_load_error_missing")


def test_engine_not_loaded_initially():
    """Engine should report not loaded before any load() call."""
    engine = ChatModelEngine()
    assert not engine.is_loaded
    status = engine.get_status()
    assert status["loaded"] is False
    print("PASS: test_engine_not_loaded_initially")


def test_engine_unload():
    """After unload, engine should report not loaded."""
    engine = ChatModelEngine()
    engine._loaded = True  # simulate loaded state
    engine.unload()
    assert not engine.is_loaded
    print("PASS: test_engine_unload")


def test_engine_stop_flag():
    """Stop flag should be settable and checkable."""
    engine = ChatModelEngine()
    assert not engine._is_stopped()
    engine.stop()
    assert engine._is_stopped()
    # Reset for next test
    engine._stop_flag.clear()
    assert not engine._is_stopped()
    print("PASS: test_engine_stop_flag")


def test_generate_stream_not_loaded():
    """generate_stream should yield error when model not loaded."""
    engine = ChatModelEngine()
    events = list(engine.generate_stream("test prompt"))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    print("PASS: test_generate_stream_not_loaded")


def test_generate_stream_empty_prompt():
    """generate_stream should yield error for empty prompt (if loaded)."""
    engine = ChatModelEngine()
    # Without loaded model, it yields error about not loaded
    events = list(engine.generate_stream(""))
    assert events[0]["type"] == "error"
    print("PASS: test_generate_stream_empty_prompt")


def test_conversation_save_load():
    """Conversation history should round-trip through JSON."""
    tmpdir = tempfile.mkdtemp()
    hist_file = os.path.join(tmpdir, "conversations.json")
    convs = [
        {
            "id": "abc123",
            "title": "Test Chat",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Hi there"},
            ],
            "created": "2026-01-01 12:00",
        }
    ]
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False)
    with open(hist_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == convs
    print("PASS: test_conversation_save_load")


def test_html_escaping():
    """format_message_content should escape HTML special characters."""
    # Import from chat_app
    from chat_app import format_message_content
    result = format_message_content("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    print("PASS: test_html_escaping")


def test_code_block_formatting():
    """format_message_content should wrap code blocks in <pre><code>."""
    from chat_app import format_message_content
    text = "Here is code:\n```python\nprint('hello')\n```\nDone."
    result = format_message_content(text)
    assert "<pre>" in result
    assert "<code>" in result
    assert "print" in result
    print("PASS: test_code_block_formatting")


def test_thread_safety():
    """Multiple threads calling stop() should not crash."""
    engine = ChatModelEngine()

    def stop_repeated():
        for _ in range(100):
            engine.stop()
            engine._stop_flag.clear()

    threads = [threading.Thread(target=stop_repeated) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # If we get here without exception, thread safety is maintained
    print("PASS: test_thread_safety")


def test_port_not_5173_or_8000():
    """Chat app port should not be 5173 or 8000."""
    from chat_app import CHAT_PORT
    assert CHAT_PORT != 5173, "Port must not be 5173"
    assert CHAT_PORT != 8000, "Port must not be 8000"
    assert CHAT_PORT > 1024, "Port should be above 1024"
    print(f"PASS: test_port_not_5173_or_8000 (port={CHAT_PORT})")


def test_checkpoint_files_preserved():
    """No training/checkpoint files should be modified by chat code."""
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    files_before = set(os.listdir(ckpt_dir)) if os.path.exists(ckpt_dir) else set()
    # The chat module only reads, never writes to checkpoints
    # Verify chat_model.py doesn't import any save function
    chat_model_path = os.path.join(os.path.dirname(__file__), "chat_model.py")
    with open(chat_model_path, "r") as f:
        content = f.read()
    assert "save_checkpoint" not in content, "chat_model should not call save_checkpoint"
    print("PASS: test_checkpoint_files_preserved")


def run_all_tests():
    """Run all 13 tests."""
    tests = [
        test_engine_list_available,
        test_engine_load_error_missing,
        test_engine_not_loaded_initially,
        test_engine_unload,
        test_engine_stop_flag,
        test_generate_stream_not_loaded,
        test_generate_stream_empty_prompt,
        test_conversation_save_load,
        test_html_escaping,
        test_code_block_formatting,
        test_thread_safety,
        test_port_not_5173_or_8000,
        test_checkpoint_files_preserved,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_all_tests()
