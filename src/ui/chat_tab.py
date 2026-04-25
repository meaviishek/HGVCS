"""
ChatTab  –  AI chat with Browse Mode + TTS Listen feature.

New in this version:
  • Browse Mode toggle: OFF = local knowledge + Ollama only
                        ON  = DuckDuckGo web search → LLM synthesises answer
  • 🔊 Listen button on every V reply bubble (reads it aloud via TTSEngine)
  • Auto-Speak toggle: automatically reads every V reply aloud
  • Knowledge-only mode: if Ollama unavailable, still answers from KnowledgeStore
"""

import logging
import threading


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

log = logging.getLogger("hgvcs.chat_tab")

# ── colour tokens ──────────────────────────────────────────
BG       = "#0d0f14"
CARD     = "#1a1f2e"
SURFACE2 = "#1e2435"
BORDER   = "#2a3245"
ACCENT   = "#6c63ff"
ACCENT2  = "#a78bfa"
GREEN    = "#22d3a5"
AMBER    = "#f59e0b"
RED      = "#ef4444"
TEAL     = "#0ea5e9"
TEXT     = "#e2e8f0"
DIMTEXT  = "#64748b"


def _font(size=12, weight=QFont.Normal):
    return QFont("Segoe UI", size, weight)


# ── web search helper (DuckDuckGo Instant Answer, no API key) ──
# ── web search helper ──────────────────────────────────────
def _web_search(query: str, max_results: int = 4) -> str:
    """
    Searches the web and returns a text summary to inject into the LLM.
    Tries DuckDuckGo Instant Answer first, then HTML search snippets.
    Never opens a browser — all results are returned as text.
    """
    import urllib.parse
    import urllib.request
    import json
    import re as _re

    parts = []

    # ── Source 1: DuckDuckGo Instant Answer API ────────────
    try:
        url = (
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1"
            .format(urllib.parse.quote(query))
        )
        req = urllib.request.Request(url, headers={"User-Agent": "HGVCS-AI/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        if data.get("AbstractText"):
            parts.append(f"Summary: {data['AbstractText']}")
        if data.get("Answer"):
            parts.append(f"Answer: {data['Answer']}")
        for topic in data.get("RelatedTopics", [])[:2]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
    except Exception as e:
        log.debug(f"DDG instant answer failed: {e}")

    # ── Source 2: DuckDuckGo HTML search (snippets) ────────
    if len(parts) < 2:
        try:
            url2 = (
                "https://html.duckduckgo.com/html/?q={}"
                .format(urllib.parse.quote(query))
            )
            req2 = urllib.request.Request(
                url2, headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                    )
                }
            )
            with urllib.request.urlopen(req2, timeout=7) as resp2:
                html = resp2.read().decode("utf-8", errors="replace")

            # Extract result snippets with simple regex
            snippets = _re.findall(
                r'<a class="result__snippet"[^>]*>(.*?)</a>', html, _re.DOTALL
            )
            if not snippets:
                # Try alternate class name
                snippets = _re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>', html, _re.DOTALL
                )

            for s in snippets[:max_results]:
                clean = _re.sub(r'<[^>]+>', '', s).strip()
                clean = _re.sub(r'\s+', ' ', clean)
                if clean and len(clean) > 20:
                    parts.append(clean)

        except Exception as e:
            log.debug(f"DDG HTML search failed: {e}")

    if not parts:
        return ""

    # Deduplicate and join
    seen = set()
    unique = []
    for p in parts:
        key = p[:50]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return "\n\n".join(unique[:max_results])



# ── KB sentence extractor (no external deps) ───────────────
def _extract_answer(query: str, results: list, max_sentences: int = 4) -> str:
    """
    Given KB search results, pick only the sentences most relevant to *query*.
    Uses keyword-overlap scoring — identical approach to KnowledgeStore, no model needed.
    Returns a short focused answer string.
    """
    import re as _re

    _STOP = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "about", "from", "into", "through", "and",
        "or", "but", "if", "then", "that", "this", "these", "those", "it",
        "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
        "they", "them", "their", "what", "which", "who", "how", "when",
        "where", "why", "not", "no", "so", "just", "very", "also",
        "tell", "show", "give", "explain", "describe", "whats", "define",
    }

    def _tok(t):
        return {w for w in _re.findall(r"[a-z0-9]+", t.lower())
                if w not in _STOP and len(w) > 1}

    q_tok = _tok(query)
    if not q_tok:
        return ""

    # Collect all sentences from all results with relevance scores
    scored = []
    for r in results:
        raw = r.get("text", "")
        sentences = _re.split(r'(?<=[.!?])\s+|\n', raw)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            s_tok = _tok(sent)
            if not s_tok:
                continue
            overlap = len(q_tok & s_tok)
            if overlap == 0:
                continue
            score = overlap / (len(q_tok | s_tok) + 1e-9)
            scored.append((score, sent))

    if not scored:
        return ""

    # Take top-N by score
    scored.sort(key=lambda x: x[0], reverse=True)
    top_set = {s for _, s in scored[:max_sentences]}

    # Restore original reading order from first result
    ordered = []
    seen_s: set = set()
    for r in results:
        raw = r.get("text", "")
        for sent in _re.split(r'(?<=[.!?])\s+|\n', raw):
            s = sent.strip()
            if s in top_set and s not in seen_s:
                seen_s.add(s)
                ordered.append(s)

    return " ".join(ordered[:max_sentences])


# ══════════════════════════════════════════════════════════
# CHAT BUBBLE  (with optional Listen button)
# ══════════════════════════════════════════════════════════
class ChatBubble(QFrame):
    listen_clicked = pyqtSignal(str)   # emits the bubble text

    def __init__(self, text: str, is_user: bool,
                 action_info: str = "", show_listen: bool = False,
                 parent=None):
        super().__init__(parent)
        self._text = text
        self._is_user = is_user
        self.setContentsMargins(0, 2, 0, 2)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(0)

        bubble = QFrame()
        bubble.setMaximumWidth(500)
        b_lay = QVBoxLayout(bubble)
        b_lay.setContentsMargins(12, 8, 12, 8)
        b_lay.setSpacing(4)

        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setFont(_font(10))
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse)

        if is_user:
            msg.setStyleSheet(f"color:{TEXT}; background:transparent;")
            bubble.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 rgba(108,99,255,200), stop:1 rgba(139,92,246,200));
                    border-radius: 14px;
                    border-bottom-right-radius: 3px;
                }}
            """)
        else:
            msg.setStyleSheet(f"color:{TEXT}; background:transparent;")
            bubble.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 rgba(26,31,46,240), stop:1 rgba(30,36,53,240));
                    border: 1px solid {BORDER};
                    border-radius: 14px;
                    border-bottom-left-radius: 3px;
                }}
            """)

        b_lay.addWidget(msg)

        if action_info:
            act = QLabel(action_info)
            act.setFont(_font(8))
            act.setStyleSheet(f"color:{GREEN}; background:transparent;")
            b_lay.addWidget(act)

        # Listen button (only on V bubbles)
        if show_listen and not is_user:
            listen_row = QHBoxLayout()
            listen_row.setContentsMargins(0, 2, 0, 0)
            btn = QPushButton("🔊 Listen")
            btn.setFixedHeight(22)
            btn.setFont(_font(8))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:transparent; color:{TEAL};
                    border:1px solid {TEAL}; border-radius:5px; padding:0 6px;
                }}
                QPushButton:hover {{ background:rgba(14,165,233,30); }}
            """)
            btn.clicked.connect(lambda: self.listen_clicked.emit(self._text))
            listen_row.addWidget(btn)
            listen_row.addStretch()
            b_lay.addLayout(listen_row)

        sender = QLabel("You" if is_user else "V")
        sender.setFont(_font(8))
        sender.setStyleSheet(
            f"color:{ACCENT2 if is_user else GREEN}; background:transparent;"
        )

        if is_user:
            outer.addStretch()
            v2 = QVBoxLayout()
            v2.addWidget(sender, 0, Qt.AlignRight)
            v2.addWidget(bubble)
            outer.addLayout(v2)
        else:
            v2 = QVBoxLayout()
            v2.addWidget(sender, 0, Qt.AlignLeft)
            v2.addWidget(bubble)
            outer.addLayout(v2)
            outer.addStretch()


# ══════════════════════════════════════════════════════════
# THINKING INDICATOR
# ══════════════════════════════════════════════════════════
class ThinkingBubble(QFrame):
    def __init__(self, browse_mode=False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)

        bubble = QFrame()
        bubble.setMaximumWidth(280)
        bubble.setStyleSheet(f"""
            QFrame {{
                background: rgba(26,31,46,200);
                border: 1px solid {BORDER};
                border-radius: 14px;
                border-bottom-left-radius: 3px;
            }}
        """)
        b_lay = QHBoxLayout(bubble)
        b_lay.setContentsMargins(14, 10, 14, 10)

        msg = "🌐 Searching & thinking..." if browse_mode else "V is thinking..."
        self._label = QLabel(msg)
        self._label.setFont(_font(10))
        self._label.setStyleSheet(f"color:{DIMTEXT}; background:transparent;")
        b_lay.addWidget(self._label)

        lay.addWidget(bubble)
        lay.addStretch()

        self._dot = 0
        self._base = msg.rstrip(".")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)

    def _tick(self):
        self._dot = (self._dot + 1) % 4
        self._label.setText(self._base + "." * self._dot)


# ══════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════
class ChatTab(QWidget):
    """
    Full AI text-chat interface with:
      - Browse Mode (web search integration)
      - Listen / Auto-Speak (TTS for replies)
      - Knowledge-store context injection
    """

    _append_signal  = pyqtSignal(str, bool, str)  # text, is_user, action_info
    _thinking_signal = pyqtSignal(bool, bool)      # show, browse_mode

    _ACTION_LABELS = {
        "open_browser": "▶ Opening browser…",
        "search_web":   "▶ Searching on Google…",
        "close_tab":    "▶ Closing tab…",
        "new_tab":      "▶ Opening new tab…",
        "volume_up":    "▶ Volume up",
        "volume_down":  "▶ Volume down",
        "mute":         "▶ Muted",
        "screenshot":   "▶ Screenshot taken!",
        "scroll_up":    "▶ Scrolling up…",
        "scroll_down":  "▶ Scrolling down…",
        "none":         "",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ollama      = None
        self._knowledge   = None
        self._voice_ctrl  = None
        self._tts         = None          # TTSEngine (injected)
        self._thinking_w  = None
        self._history     = []
        self._browse_mode = False
        self._auto_speak  = False
        self._build()
        self._append_signal.connect(self._append_bubble)
        self._thinking_signal.connect(self._set_thinking)

    # ── dependency injection ───────────────────────────────
    def set_ollama_client(self, ollama):
        self._ollama = ollama
        if ollama:
            model = getattr(ollama, '_model', '')
            if model:
                self._model_lbl.setText(f"Model: {model}")

    def set_knowledge_store(self, ks):
        self._knowledge = ks

    def set_voice_controller(self, vc):
        self._voice_ctrl = vc
        # Grab TTS from voice controller
        if vc and getattr(vc, '_tts', None):
            self._tts = vc._tts

    def set_tts_engine(self, tts):
        self._tts = tts

    def update_model_label(self, model_name: str):
        self._model_lbl.setText(f"Model: {model_name}")

    # ── UI construction ────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────
        hdr = QFrame()
        hdr.setFixedHeight(56)
        hdr.setStyleSheet(f"background:{CARD}; border-bottom:1px solid {BORDER};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(16, 0, 16, 0)
        hdr_lay.setSpacing(10)

        title = QLabel("🤖  Chat with V")
        title.setFont(_font(13, QFont.Bold))
        title.setStyleSheet(f"color:{TEXT};")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()

        self._model_lbl = QLabel("Model: —")
        self._model_lbl.setFont(_font(9))
        self._model_lbl.setStyleSheet(f"color:{DIMTEXT};")
        hdr_lay.addWidget(self._model_lbl)

        # Auto-Speak toggle
        self._speak_btn = QPushButton("🔇 Auto-Speak")
        self._speak_btn.setFixedHeight(30)
        self._speak_btn.setCheckable(True)
        self._speak_btn.setCursor(Qt.PointingHandCursor)
        self._speak_btn.clicked.connect(self._toggle_auto_speak)
        self._speak_btn.setStyleSheet(self._toggle_style(False))
        hdr_lay.addWidget(self._speak_btn)

        # Browse Mode toggle
        self._browse_btn = QPushButton("🌐 Browse: OFF")
        self._browse_btn.setFixedHeight(30)
        self._browse_btn.setCheckable(True)
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.clicked.connect(self._toggle_browse)
        self._browse_btn.setStyleSheet(self._toggle_style(False))
        hdr_lay.addWidget(self._browse_btn)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(56, 30)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{DIMTEXT};
                border:1px solid {BORDER}; border-radius:6px; font-size:10px;
            }}
            QPushButton:hover {{ color:{TEXT}; border-color:{ACCENT}; }}
        """)
        clear_btn.clicked.connect(self._clear_chat)
        hdr_lay.addWidget(clear_btn)
        root.addWidget(hdr)

        # ── Browse Mode info bar (hidden by default) ──────
        self._browse_bar = QFrame()
        self._browse_bar.setFixedHeight(30)
        self._browse_bar.setStyleSheet(f"""
            QFrame {{
                background: rgba(14,165,233,20);
                border-bottom: 1px solid rgba(14,165,233,80);
            }}
        """)
        bar_lay = QHBoxLayout(self._browse_bar)
        bar_lay.setContentsMargins(16, 0, 16, 0)
        bar_info = QLabel("🌐 Browse Mode ON — V will search the web and answer from live results")
        bar_info.setFont(_font(8))
        bar_info.setStyleSheet(f"color:{TEAL};")
        bar_lay.addWidget(bar_info)
        self._browse_bar.hide()
        root.addWidget(self._browse_bar)

        # ── Chat scroll area ──────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{ border:none; background:{BG}; }}
            QScrollBar:vertical {{
                background:{CARD}; width:5px; border-radius:3px;
            }}
            QScrollBar::handle:vertical {{
                background:{BORDER}; border-radius:3px; min-height:20px;
            }}
        """)

        self._chat_container = QWidget()
        self._chat_container.setStyleSheet(f"background:{BG};")
        self._chat_lay = QVBoxLayout(self._chat_container)
        self._chat_lay.setContentsMargins(0, 12, 0, 12)
        self._chat_lay.setSpacing(4)
        self._chat_lay.addStretch()

        self._scroll.setWidget(self._chat_container)
        root.addWidget(self._scroll, 1)
        self._show_welcome()

        # ── Input area ────────────────────────────────────
        input_frame = QFrame()
        input_frame.setStyleSheet(f"""
            QFrame {{
                background:{CARD};
                border-top:1px solid {BORDER};
            }}
        """)
        input_lay = QVBoxLayout(input_frame)
        input_lay.setContentsMargins(16, 8, 16, 12)
        input_lay.setSpacing(6)

        hint = QLabel(
            "📚 Browse OFF: answers strictly from your knowledge base  •  "
            "Browse ON: searches web and answers with live data"
        )
        hint.setFont(_font(8))
        hint.setStyleSheet(f"color:{DIMTEXT};")
        input_lay.addWidget(hint)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self._input = QTextEdit()
        self._input.setFixedHeight(54)
        self._input.setPlaceholderText("Type a message or command…")
        self._input.setFont(_font(10))
        self._input.setStyleSheet(f"""
            QTextEdit {{
                background:{SURFACE2}; color:{TEXT};
                border:1px solid {BORDER}; border-radius:10px; padding:8px 12px;
            }}
            QTextEdit:focus {{ border-color:{ACCENT}; }}
        """)
        self._input.installEventFilter(self)
        input_row.addWidget(self._input, 1)

        send_btn = QPushButton("Send")
        send_btn.setFixedSize(68, 54)
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {ACCENT}, stop:1 #5a52d5);
                color:white; border:none; border-radius:10px;
                font-size:11px; font-weight:bold;
            }}
            QPushButton:hover {{ background:{ACCENT2}; }}
            QPushButton:pressed {{ background:#4a42c0; }}
        """)
        send_btn.clicked.connect(self._send)
        input_row.addWidget(send_btn)
        input_lay.addLayout(input_row)
        root.addWidget(input_frame)

    def _toggle_style(self, active: bool, color=ACCENT) -> str:
        if active:
            return f"""
                QPushButton {{
                    background:rgba(108,99,255,40); color:{color};
                    border:1px solid {color}; border-radius:6px;
                    font-size:10px; padding:0 10px;
                }}
                QPushButton:hover {{ background:rgba(108,99,255,70); }}
            """
        return f"""
            QPushButton {{
                background:transparent; color:{DIMTEXT};
                border:1px solid {BORDER}; border-radius:6px;
                font-size:10px; padding:0 10px;
            }}
            QPushButton:hover {{ color:{TEXT}; border-color:{ACCENT}; }}
        """

    def _toggle_browse(self):
        self._browse_mode = self._browse_btn.isChecked()
        if self._browse_mode:
            self._browse_btn.setText("🌐 Browse: ON")
            self._browse_btn.setStyleSheet(self._toggle_style(True, TEAL))
            self._browse_bar.show()
            self._append_signal.emit(
                "Browse Mode is now ON. I'll search the web for your questions!", False, "")
        else:
            self._browse_btn.setText("🌐 Browse: OFF")
            self._browse_btn.setStyleSheet(self._toggle_style(False))
            self._browse_bar.hide()
            self._append_signal.emit(
                "Browse Mode is OFF. I'll answer only from your knowledge base and my training.", False, "")

    def _toggle_auto_speak(self):
        self._auto_speak = self._speak_btn.isChecked()
        if self._auto_speak:
            self._speak_btn.setText("🔊 Auto-Speak")
            self._speak_btn.setStyleSheet(self._toggle_style(True, AMBER))
            self._append_signal.emit("Auto-Speak enabled — I'll read every reply aloud!", False, "")
        else:
            self._speak_btn.setText("🔇 Auto-Speak")
            self._speak_btn.setStyleSheet(self._toggle_style(False))

    def _show_welcome(self):
        bubble = ChatBubble(
            "Hi! I'm V 👋\n\n"
            "• Browse OFF → I answer from your Knowledge Base + my training\n"
            "• Browse ON  → I search the web and give you live answers\n"
            "• Press 🔊 Listen on any reply to hear it, or enable Auto-Speak!\n\n"
            "Try: 'open YouTube', 'search Python tutorials', or ask me anything!",
            is_user=False, show_listen=True
        )
        bubble.listen_clicked.connect(self._speak_text)
        self._chat_lay.insertWidget(self._chat_lay.count() - 1, bubble)

    # ── event filter ──────────────────────────────────────
    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj is self._input and event.type() == QEvent.KeyPress:
            if (event.key() in (Qt.Key_Return, Qt.Key_Enter)
                    and not event.modifiers() & Qt.ShiftModifier):
                self._send()
                return True
        return super().eventFilter(obj, event)

    # ── send ──────────────────────────────────────────────
    def _send(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._append_signal.emit(text, True, "")
        self._history.append(("user", text))
        self._thinking_signal.emit(True, self._browse_mode)
        threading.Thread(
            target=self._process, args=(text, self._browse_mode), daemon=True
        ).start()

    # ── process (background thread) ───────────────────────
    def _process(self, text: str, browse_mode: bool):
        try:
            action_label = ""

            # ── 1. OS command fast-path (non-browse actions only) ──
            # In browse mode: still handle system commands (volume, scroll etc.)
            # but NEVER execute search_web or open_browser — those stay in chat
            if self._voice_ctrl:
                try:
                    fast = self._voice_ctrl._rule_based(text.lower(), text)
                    if fast:
                        action, params, fast_reply = fast
                        # Actions that ALWAYS execute (system control)
                        _system_actions = {
                            "volume_up", "volume_down", "mute", "scroll_up",
                            "scroll_down", "screenshot", "close_window",
                            "minimize_window", "maximize_window", "task_switch",
                            "ppt_next", "ppt_prev", "ppt_start", "ppt_end",
                            "copy", "paste", "save", "lock_screen",
                        }
                        if action in _system_actions:
                            action_label = self._ACTION_LABELS.get(action, f"▶ {action}")
                            try:
                                self._voice_ctrl._execute_action(action, params)
                            except Exception:
                                pass
                            self._finish(fast_reply, action_label)
                            return

                        # Non-system actions (open_browser, search_web, open sites):
                        # In browse OFF mode: execute them normally
                        # In browse ON mode: skip execution, let LLM/web answer in chat
                        if not browse_mode and action not in ("none", ""):
                            action_label = self._ACTION_LABELS.get(action, f"▶ {action}")
                            try:
                                self._voice_ctrl._execute_action(action, params)
                            except Exception:
                                pass
                            self._finish(fast_reply, action_label)
                            return
                except Exception:
                    pass

            # ── 2. Knowledge Base lookup ───────────────────
            knowledge_ctx = ""
            if self._knowledge:
                try:
                    knowledge_ctx = self._knowledge.build_context(text, top_k=5)
                except Exception:
                    pass

            # ── 3. BROWSE OFF: knowledge base ONLY — focused sentence extraction ──
            if not browse_mode:
                if knowledge_ctx and self._knowledge:
                    try:
                        results = self._knowledge.search(text, top_k=3)
                    except Exception:
                        results = []

                    if results:
                        answer = _extract_answer(text, results, max_sentences=4)
                        if not answer:
                            # Fallback: first 2 sentences of best-scoring entry
                            import re as _re2
                            best = results[0].get("text", "")
                            sents = _re2.split(r'(?<=[.!?])\s+', best)
                            answer = " ".join(s.strip() for s in sents[:2] if s.strip())
                        if answer:
                            self._history.append(("assistant", answer))
                            self._finish(answer, "📚 From knowledge base")
                            return

                # No KB match → strict "I don't know"
                reply = (
                    "I don't know — that topic isn't in my knowledge base.\n\n"
                    "💡 You can:\n"
                    "  • Enable **Browse Mode** (🌐 button above) to search the web\n"
                    "  • Add the information in the **Knowledge** tab so I can answer next time"
                )
                self._finish(reply, "📚 Not in knowledge base")
                return


            # ── 4. BROWSE ON: web search → LLM ────────────
            web_ctx = _web_search(text)
            web_label = ""

            if web_ctx:
                web_label = "🌐 Searched web"
                web_ctx_block = (
                    f"\n\n[WEB SEARCH RESULTS for: '{text}']\n"
                    f"{web_ctx}\n"
                    f"[END OF WEB RESULTS]\n"
                )
            else:
                web_ctx_block = ""
                web_label = "🌐 No web results"

            extra_context = knowledge_ctx + web_ctx_block

            if not self._ollama:
                if web_ctx:
                    reply = f"Here's what I found on the web:\n\n{web_ctx[:800]}"
                elif knowledge_ctx:
                    reply = f"Based on my knowledge base:\n\n{knowledge_ctx[:600]}"
                else:
                    reply = "Sorry, I couldn't find any information. Try rephrasing your question."
                self._finish(reply, web_label)
                return

            if web_ctx:
                llm_text = (
                    f"{text}\n\n"
                    f"[INSTRUCTION: Answer using the web search results provided. "
                    f"Summarize clearly in 2-4 sentences. "
                    f"Do NOT suggest opening a browser. Reply directly in the chat only.]"
                )
            else:
                llm_text = (
                    f"{text}\n\n"
                    f"[INSTRUCTION: Web search returned no results. "
                    f"If you have knowledge base context use it; otherwise say you couldn't find information.]"
                )

            result = self._ollama.ask(llm_text, extra_context=extra_context)
            reply  = result.get("reply", "") or "Sorry, I couldn't find a good answer."

            self._history.append(("assistant", reply))
            self._finish(reply, web_label)

        except Exception as e:
            log.error(f"Chat process error: {e}")
            self._thinking_signal.emit(False, False)
            self._append_signal.emit(f"Sorry, something went wrong: {e}", False, "")



    def _finish(self, reply: str, action_label: str):
        self._thinking_signal.emit(False, False)
        self._append_signal.emit(reply, False, action_label)
        if self._auto_speak:
            self._speak_text(reply)

    # ── TTS ───────────────────────────────────────────────
    def _speak_text(self, text: str):
        if not text:
            return
        # Try voice controller TTS first, then fallback
        tts = self._tts
        if not tts and self._voice_ctrl:
            tts = getattr(self._voice_ctrl, '_tts', None)
        if tts:
            threading.Thread(
                target=tts.speak, args=(text,), kwargs={"blocking": False},
                daemon=True
            ).start()
        else:
            # pyttsx3 fallback
            try:
                import pyttsx3
                def _say():
                    e = pyttsx3.init()
                    e.say(text[:500])
                    e.runAndWait()
                threading.Thread(target=_say, daemon=True).start()
            except Exception as ex:
                log.warning(f"TTS unavailable: {ex}")

    # ── Qt-thread slots ───────────────────────────────────
    @pyqtSlot(str, bool, str)
    def _append_bubble(self, text: str, is_user: bool, action_info: str):
        bubble = ChatBubble(
            text, is_user, action_info,
            show_listen=(not is_user)  # Listen button on every V reply
        )
        if not is_user:
            bubble.listen_clicked.connect(self._speak_text)
        self._chat_lay.insertWidget(self._chat_lay.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_bottom)

    @pyqtSlot(bool, bool)
    def _set_thinking(self, show: bool, browse_mode: bool):
        if show:
            if self._thinking_w is None:
                self._thinking_w = ThinkingBubble(browse_mode=browse_mode)
                self._chat_lay.insertWidget(
                    self._chat_lay.count() - 1, self._thinking_w)
            QTimer.singleShot(50, self._scroll_bottom)
        else:
            if self._thinking_w is not None:
                self._thinking_w.setParent(None)
                self._thinking_w.deleteLater()
                self._thinking_w = None

    def _scroll_bottom(self):
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        )

    def _clear_chat(self):
        while self._chat_lay.count() > 1:
            item = self._chat_lay.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._history.clear()
        self._show_welcome()
