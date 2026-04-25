"""
KnowledgeTab  –  Teach V anything, persist it forever.

Left panel:  paste text / upload file  ->  Save to Memory
Right panel: list all saved entries    ->  delete individual items or Clear All

Uses KnowledgeStore (injected via set_knowledge_store()).
"""

import logging
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QFrame, QScrollArea, QFileDialog,
    QMessageBox, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor

log = logging.getLogger("hgvcs.knowledge_tab")

# ── colour tokens ───────────────────────────────────────────
BG       = "#0d0f14"
CARD     = "#1a1f2e"
SURFACE2 = "#1e2435"
BORDER   = "#2a3245"
ACCENT   = "#6c63ff"
ACCENT2  = "#a78bfa"
GREEN    = "#22d3a5"
AMBER    = "#f59e0b"
RED      = "#ef4444"
TEXT     = "#e2e8f0"
DIMTEXT  = "#64748b"


def _font(size=12, weight=QFont.Normal):
    return QFont("Segoe UI", size, weight)


# ══════════════════════════════════════════════════════════
# KNOWLEDGE ENTRY CARD
# ══════════════════════════════════════════════════════════
class KnowledgeCard(QFrame):
    deleted = pyqtSignal(str)   # emits entry id

    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self._entry_id = entry["id"]
        self.setStyleSheet(f"""
            QFrame {{
                background:{SURFACE2};
                border:1px solid {BORDER};
                border-radius:10px;
            }}
            QFrame:hover {{
                border-color:{ACCENT};
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 10, 10)
        lay.setSpacing(8)

        # icon
        icon = QLabel("📄")
        icon.setFont(_font(18))
        icon.setFixedWidth(30)
        lay.addWidget(icon)

        # info
        info = QVBoxLayout()
        info.setSpacing(2)

        title_lbl = QLabel(entry.get("title", "Untitled"))
        title_lbl.setFont(_font(10, QFont.Bold))
        title_lbl.setStyleSheet(f"color:{TEXT};")
        info.addWidget(title_lbl)

        preview = entry.get("text", "")[:80].replace("\n", " ")
        if len(entry.get("text", "")) > 80:
            preview += "…"
        preview_lbl = QLabel(preview)
        preview_lbl.setFont(_font(9))
        preview_lbl.setStyleSheet(f"color:{DIMTEXT};")
        info.addWidget(preview_lbl)

        added = entry.get("added", "")
        meta_lbl = QLabel(
            f"{len(entry.get('text',''))} chars  •  added {added[:10]}"
        )
        meta_lbl.setFont(_font(8))
        meta_lbl.setStyleSheet(f"color:{DIMTEXT};")
        info.addWidget(meta_lbl)

        lay.addLayout(info, 1)

        # delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(26, 26)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setToolTip("Delete this entry")
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{DIMTEXT};
                border:1px solid transparent; border-radius:13px;
                font-size:10px; font-weight:bold;
            }}
            QPushButton:hover {{
                color:{RED}; border-color:{RED};
                background:rgba(239,68,68,15);
            }}
        """)
        del_btn.clicked.connect(lambda: self.deleted.emit(self._entry_id))
        lay.addWidget(del_btn, 0, Qt.AlignTop)


# ══════════════════════════════════════════════════════════
# KNOWLEDGE TAB
# ══════════════════════════════════════════════════════════
class KnowledgeTab(QWidget):
    """
    UI for teaching V new knowledge.
    Inject: tab.set_knowledge_store(ks)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ks = None
        self._cards = {}   # entry_id -> KnowledgeCard
        self._build()

    def set_knowledge_store(self, ks):
        self._ks = ks
        self._refresh_list()
        self._update_stats()

    # ── UI construction ────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ─ Header ─
        hdr = QFrame()
        hdr.setFixedHeight(52)
        hdr.setStyleSheet(f"background:{CARD}; border-bottom:1px solid {BORDER};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(20, 0, 20, 0)

        title = QLabel("📚  Knowledge Base")
        title.setFont(_font(13, QFont.Bold))
        title.setStyleSheet(f"color:{TEXT};")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()

        self._stats_lbl = QLabel("0 items stored")
        self._stats_lbl.setFont(_font(9))
        self._stats_lbl.setStyleSheet(f"color:{DIMTEXT};")
        hdr_lay.addWidget(self._stats_lbl)

        root.addWidget(hdr)

        # ─ Main split ─
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background:{BORDER}; }}")
        splitter.setChildrenCollapsible(False)

        # ── LEFT: Add knowledge ──
        left = QFrame()
        left.setStyleSheet(f"QFrame {{ background:{BG}; }}")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(20, 20, 16, 20)
        left_lay.setSpacing(12)

        add_title = QLabel("✏️  Add Knowledge")
        add_title.setFont(_font(12, QFont.Bold))
        add_title.setStyleSheet(f"color:{TEXT};")
        left_lay.addWidget(add_title)

        desc = QLabel(
            "Teach V anything — facts, docs, notes.\n"
            "V will use this context when answering by chat or voice."
        )
        desc.setFont(_font(9))
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{DIMTEXT};")
        left_lay.addWidget(desc)

        # Title field
        title_lbl = QLabel("Title  (short description)")
        title_lbl.setFont(_font(9))
        title_lbl.setStyleSheet(f"color:{DIMTEXT};")
        left_lay.addWidget(title_lbl)

        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("e.g. Company info, My schedule…")
        self._title_edit.setFont(_font(10))
        self._title_edit.setStyleSheet(f"""
            QLineEdit {{
                background:{SURFACE2}; color:{TEXT};
                border:1px solid {BORDER}; border-radius:8px;
                padding:6px 10px;
            }}
            QLineEdit:focus {{ border-color:{ACCENT}; }}
        """)
        left_lay.addWidget(self._title_edit)

        # Text area
        text_lbl = QLabel("Content  (paste text here)")
        text_lbl.setFont(_font(9))
        text_lbl.setStyleSheet(f"color:{DIMTEXT};")
        left_lay.addWidget(text_lbl)

        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText(
            "Paste any text here — facts, notes, articles, instructions…\n\n"
            "Example:\n  My meeting is every Monday at 10 AM.\n"
            "  The project deadline is May 30th."
        )
        self._text_edit.setFont(_font(10))
        self._text_edit.setStyleSheet(f"""
            QTextEdit {{
                background:{SURFACE2}; color:{TEXT};
                border:1px solid {BORDER}; border-radius:8px;
                padding:8px 12px;
            }}
            QTextEdit:focus {{ border-color:{ACCENT}; }}
        """)
        left_lay.addWidget(self._text_edit, 1)

        # Status label
        self._status_lbl = QLabel("")
        self._status_lbl.setFont(_font(9))
        self._status_lbl.setStyleSheet(f"color:{GREEN};")
        left_lay.addWidget(self._status_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        upload_btn = QPushButton("📁  Upload File")
        upload_btn.setFixedHeight(38)
        upload_btn.setCursor(Qt.PointingHandCursor)
        upload_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{ACCENT2};
                border:1px solid {ACCENT2}; border-radius:8px;
                font-size:10px; font-weight:bold; padding:0 14px;
            }}
            QPushButton:hover {{ background:rgba(167,139,250,20); }}
        """)
        upload_btn.clicked.connect(self._upload_file)
        btn_row.addWidget(upload_btn)

        save_btn = QPushButton("💾  Save to Memory")
        save_btn.setFixedHeight(38)
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {GREEN}, stop:1 #1ab894);
                color:#0d0f14;
                border:none; border-radius:8px;
                font-size:10px; font-weight:bold; padding:0 20px;
            }}
            QPushButton:hover {{ background:#1ab894; }}
            QPushButton:pressed {{ background:#15a07e; }}
        """)
        save_btn.clicked.connect(self._save_text)
        btn_row.addWidget(save_btn, 1)

        left_lay.addLayout(btn_row)

        # Supported formats note
        fmt_lbl = QLabel("Supported files: .txt  .md  .csv  .pdf  .docx")
        fmt_lbl.setFont(_font(8))
        fmt_lbl.setStyleSheet(f"color:{DIMTEXT};")
        left_lay.addWidget(fmt_lbl)

        splitter.addWidget(left)

        # ── RIGHT: Knowledge list ──
        right = QFrame()
        right.setStyleSheet(f"QFrame {{ background:{BG}; }}")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 20, 20, 20)
        right_lay.setSpacing(10)

        right_hdr = QHBoxLayout()
        list_title = QLabel("📖  Stored Knowledge")
        list_title.setFont(_font(12, QFont.Bold))
        list_title.setStyleSheet(f"color:{TEXT};")
        right_hdr.addWidget(list_title)
        right_hdr.addStretch()

        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedSize(75, 28)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{RED};
                border:1px solid {RED}; border-radius:6px;
                font-size:9px; font-weight:bold;
            }}
            QPushButton:hover {{ background:rgba(239,68,68,15); }}
        """)
        clear_btn.clicked.connect(self._clear_all)
        right_hdr.addWidget(clear_btn)
        right_lay.addLayout(right_hdr)

        search_lbl = QLabel("V searches these automatically when you chat or speak.")
        search_lbl.setFont(_font(9))
        search_lbl.setStyleSheet(f"color:{DIMTEXT};")
        right_lay.addWidget(search_lbl)

        # Scrollable list of cards
        self._list_scroll = QScrollArea()
        self._list_scroll.setWidgetResizable(True)
        self._list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list_scroll.setStyleSheet(f"""
            QScrollArea {{ border:none; background:{BG}; }}
            QScrollBar:vertical {{
                background:{CARD}; width:5px; border-radius:3px;
            }}
            QScrollBar::handle:vertical {{
                background:{BORDER}; border-radius:3px; min-height:20px;
            }}
        """)

        self._list_container = QWidget()
        self._list_container.setStyleSheet(f"background:{BG};")
        self._list_lay = QVBoxLayout(self._list_container)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(8)
        self._list_lay.addStretch()

        # Empty state label
        self._empty_lbl = QLabel(
            "No knowledge saved yet.\n\n"
            "Paste text or upload a file on the left to get started!"
        )
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setFont(_font(10))
        self._empty_lbl.setStyleSheet(f"color:{DIMTEXT};")
        self._empty_lbl.setWordWrap(True)
        self._list_lay.insertWidget(0, self._empty_lbl)

        self._list_scroll.setWidget(self._list_container)
        right_lay.addWidget(self._list_scroll, 1)

        splitter.addWidget(right)
        splitter.setSizes([480, 420])

        root.addWidget(splitter, 1)

    # ── actions ────────────────────────────────────────────
    def _save_text(self):
        if not self._ks:
            self._show_status("Knowledge store not ready.", error=True)
            return
        title = self._title_edit.text().strip() or "Untitled"
        text  = self._text_edit.toPlainText().strip()
        if not text:
            self._show_status("Please enter some text first.", error=True)
            return
        try:
            self._ks.add_text(title, text)
            self._title_edit.clear()
            self._text_edit.clear()
            self._show_status(f"✓ Saved '{title}' to memory!")
            self._refresh_list()
            self._update_stats()
        except Exception as e:
            self._show_status(f"Error: {e}", error=True)

    def _upload_file(self):
        if not self._ks:
            self._show_status("Knowledge store not ready.", error=True)
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload Knowledge File", "",
            "Documents (*.txt *.md *.csv *.pdf *.docx);;All Files (*)"
        )
        if not path:
            return
        try:
            self._show_status(f"Reading {os.path.basename(path)}…")
            entry_id = self._ks.add_file(path)
            self._show_status(f"✓ '{os.path.basename(path)}' saved to memory!")
            self._refresh_list()
            self._update_stats()
        except Exception as e:
            self._show_status(f"Error reading file: {e}", error=True)

    def _delete_entry(self, entry_id: str):
        if self._ks:
            self._ks.delete(entry_id)
            card = self._cards.pop(entry_id, None)
            if card:
                card.setParent(None)
                card.deleteLater()
            self._update_empty_state()
            self._update_stats()

    def _clear_all(self):
        if not self._ks:
            return
        reply = QMessageBox.question(
            self, "Clear All Knowledge",
            "Are you sure you want to delete ALL stored knowledge?\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._ks.clear()
            self._refresh_list()
            self._update_stats()

    # ── helpers ────────────────────────────────────────────
    def _refresh_list(self):
        """Rebuild the knowledge card list from scratch."""
        # Clear existing cards
        for card in list(self._cards.values()):
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        if not self._ks:
            return

        items = self._ks.all_items()
        for entry in items:
            card = KnowledgeCard(entry)
            card.deleted.connect(self._delete_entry)
            self._cards[entry["id"]] = card
            # Insert before the stretch
            self._list_lay.insertWidget(self._list_lay.count() - 1, card)

        self._update_empty_state()

    def _update_empty_state(self):
        has_items = len(self._cards) > 0
        self._empty_lbl.setVisible(not has_items)

    def _update_stats(self):
        if self._ks:
            n = self._ks.count()
            self._stats_lbl.setText(
                f"{n} item{'s' if n != 1 else ''} stored"
            )

    def _show_status(self, msg: str, error: bool = False):
        color = RED if error else GREEN
        self._status_lbl.setStyleSheet(f"color:{color};")
        self._status_lbl.setText(msg)
        QTimer.singleShot(4000, lambda: self._status_lbl.setText(""))
