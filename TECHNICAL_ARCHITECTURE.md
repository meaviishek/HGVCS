# HGVCS — Technical Architecture & AI Implementation Guide

> **Hand Gesture & Voice Control System** · Version 1.0.0  
> Python 3.11 · PyQt5 · MediaPipe · Whisper · Ollama · edge-tts

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [AI & Learning Model Architecture](#3-ai--learning-model-architecture)
4. [Knowledge Base — How Training Works](#4-knowledge-base--how-training-works)
5. [Voice Recognition Pipeline](#5-voice-recognition-pipeline)
6. [Command Routing — Rule Engine vs LLM](#6-command-routing--rule-engine-vs-llm)
7. [Chat System & Browse Mode](#7-chat-system--browse-mode)
8. [Gesture Recognition Engine](#8-gesture-recognition-engine)
9. [Text-to-Speech (TTS)](#9-text-to-speech-tts)
10. [Macro System](#10-macro-system)
11. [Data Flow Diagram](#11-data-flow-diagram)
12. [Dependencies & Stack](#12-dependencies--stack)

---

## 1. System Overview

HGVCS is a local-first AI assistant that controls the OS through **hand gestures** and **voice commands**. It is designed to work **entirely offline** by default — no cloud API keys required.

The system has three intelligence layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Perception** | MediaPipe Hands | Detect and classify hand gestures in real-time |
| **Language** | Whisper STT + Ollama LLM | Transcribe voice → understand intent |
| **Memory** | KnowledgeStore (custom) | Persistent, user-trained knowledge base |

---

## 2. Project Structure

```
hgvcs/
├── main.py                         # Entry point, dependency wiring
├── data/
│   └── v_knowledge.json            # Persistent knowledge base (auto-generated)
├── config/
│   └── macros.json                 # Saved gesture macro sequences
└── src/
    ├── gesture/
    │   ├── hand_engine.py          # MediaPipe hand landmark detection
    │   ├── gesture_classifier.py   # Rule-based gesture classifier
    │   ├── gesture_controller.py   # Gesture → OS action pipeline
    │   └── macro_engine.py         # Multi-gesture sequence macros
    ├── voice/
    │   ├── wake_word.py            # "Hey V" wake detector (Whisper tiny)
    │   ├── voice_controller.py     # Command routing (rules + LLM)
    │   ├── ollama_client.py        # Local LLM interface (Ollama)
    │   ├── knowledge_store.py      # Knowledge base engine (TF-IDF search)
    │   └── tts_engine.py           # Neural TTS (edge-tts / pyttsx3)
    ├── control/
    │   └── system_controller.py    # pyautogui OS actions
    └── ui/
        ├── main_window.py          # PyQt5 main application window
        ├── chat_tab.py             # AI chat interface
        └── knowledge_tab.py        # Knowledge base management UI
```

---

## 3. AI & Learning Model Architecture

### 3.1 The Hybrid AI Model

HGVCS does **not** use a single monolithic AI model. Instead, it uses a **three-layer hybrid architecture** for maximum speed and reliability:

```
User Input
    │
    ▼
┌──────────────────────────────────────────┐
│  LAYER 1: Rule Engine (< 1ms)            │
│  Regex-based fast-path for known cmds    │
│  search, scroll, volume, open app...     │
└──────────────────────────────────────────┘
    │ No match? ↓
┌──────────────────────────────────────────┐
│  LAYER 2: Knowledge Base (< 5ms)         │
│  TF-IDF search on user-trained content   │
│  Answers from v_knowledge.json only      │
└──────────────────────────────────────────┘
    │ Knowledge found? ↓
┌──────────────────────────────────────────┐
│  LAYER 3: Ollama LLM (1–10s)             │
│  llama3.2:3b runs locally via Ollama     │
│  Synthesises answer from KB context      │
└──────────────────────────────────────────┘
```

**Why this design?**
- Layer 1 handles 90% of daily OS commands instantly
- Layer 2 ensures V only answers what *you* taught it (Browse OFF mode)
- Layer 3 produces natural language — but is grounded to your knowledge, not hallucinating

---

### 3.2 The LLM — Ollama + llama3.2:3b

**File:** `src/voice/ollama_client.py`

HGVCS uses [Ollama](https://ollama.com/) to run a **local LLM** — no internet, no API key.

```python
# How the LLM is queried
result = ollama_client.ask(
    user_text,
    extra_context=knowledge_context  # injected from KnowledgeStore
)
# Returns: { "action": "open_browser", "params": {}, "reply": "Opening Chrome!" }
```

The LLM is given a **system prompt** that instructs it to:
1. Output valid JSON with `action`, `params`, and `reply` fields
2. Respond naturally as "V", the AI assistant
3. Only use knowledge from the provided context (when in Browse OFF mode)

**Model:** `llama3.2:3b` — small enough to run on CPU (4GB RAM), fast enough for real-time use.

```
POST http://localhost:11434/api/generate
{
  "model": "llama3.2:3b",
  "prompt": "<system_prompt>\n\nKNOWLEDGE:\n...\n\nUser: ...",
  "stream": false,
  "options": { "num_gpu": 0, "temperature": 0.3 }
}
```

---

## 4. Knowledge Base — How Training Works

**File:** `src/voice/knowledge_store.py`  
**UI:** `src/ui/knowledge_tab.py`  
**Storage:** `data/v_knowledge.json`

### 4.1 What it is

The `KnowledgeStore` is V's **long-term memory**. It stores anything you teach it — text snippets, uploaded documents, facts, procedures — and retrieves the most relevant pieces when you ask a question.

There is **no model fine-tuning** involved. Instead, a fast **TF-IDF-style keyword overlap algorithm** scores each stored entry against the query. This makes retrieval:
- ⚡ Instant (no GPU needed)
- 📖 Transparent (you can read the `v_knowledge.json` file directly)
- 🔒 Private (no data leaves your machine)

### 4.2 Data Schema

Each knowledge entry stored in `data/v_knowledge.json`:

```json
{
  "id": "a3f82c1b",
  "title": "How to reset the modem",
  "text": "Turn off the modem, wait 30 seconds, then press the reset button...",
  "added": "2026-04-22T10:30:00"
}
```

### 4.3 How to Add Knowledge

**Method 1: Knowledge Tab (UI)**
1. Open HGVCS → click the **Knowledge** tab
2. Type or paste any text in the left panel
3. Click **Save to Memory** → stored instantly

**Method 2: Upload a File**
- Supported: `.txt`, `.md`, `.csv`, `.pdf`, `.docx`
- Text is extracted automatically and added as a single entry

**Method 3: Via Voice**
- Say: *"Hey V, remember that [fact/text]"*

### 4.4 Retrieval Algorithm (TF-IDF-style Keyword Scoring)

When a query comes in, the engine:

**Step 1 — Tokenise**
```python
def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]
```
Stop words (`a`, `the`, `is`, `of`…) are stripped. Only meaningful keywords remain.

**Step 2 — Score each entry (Jaccard similarity)**
```python
def _score(query_tokens, entry_tokens) -> float:
    q = set(query_tokens)
    e = set(entry_tokens)
    intersection = q & e
    return len(intersection) / (len(q | e) + 1e-9)
```

This gives a score from 0.0 (no match) to 1.0 (identical). Higher = more relevant.

**Step 3 — Return top-k results**
```python
scored.sort(key=lambda x: x[0], reverse=True)
return scored[:top_k]
```

**Step 4 — Build context string for LLM**
```python
def build_context(query, top_k=3, max_chars=800) -> str:
    results = self.search(query, top_k)
    # Returns: "RELEVANT KNOWLEDGE:\n[Title] Text...\n"
```

This context is prepended to the LLM prompt so the model answers from *your* data.

### 4.5 Persistence

Every `add_text()` call immediately writes to disk:
```python
def _save(self):
    with open(self._path, "w", encoding="utf-8") as f:
        json.dump(self._items, f, indent=2, ensure_ascii=False)
```

Knowledge **persists forever** across sessions. Even if you restart HGVCS, all your taught content is available.

### 4.6 File Import Support

| Format | Library Used | Notes |
|--------|-------------|-------|
| `.txt`, `.md`, `.csv` | Built-in `open()` | Auto-detects encoding (UTF-8, Latin-1, etc.) |
| `.pdf` | `pypdf` or `pdfplumber` | Extracts text from all pages |
| `.docx` | `python-docx` | Extracts all paragraph text |

---

## 5. Voice Recognition Pipeline

**Files:** `src/voice/wake_word.py`, `src/voice/voice_controller.py`

### 5.1 Dual-Model Whisper Architecture

HGVCS uses **two separate Whisper models** for different roles:

```
Microphone Audio
       │
       ▼
┌─────────────────────────────────────────────┐
│  WHISPER TINY  (wake-word detection)         │
│  • ~39 MB model                              │
│  • beam_size=1 (fast)                        │
│  • Detects: "Hey V", "Hey B", "Wake up"...   │
│  • Runs every 0.5s on short audio chunks     │
└─────────────────────────────────────────────┘
       │ Wake word detected ↓
┌─────────────────────────────────────────────┐
│  WHISPER BASE  (command transcription)       │
│  • ~74 MB model (3× more accurate)           │
│  • beam_size=5, best_of=5                    │
│  • initial_prompt with command vocabulary    │
│  • Handles accented/non-native speech        │
└─────────────────────────────────────────────┘
       │ Text command ↓
  VoiceController
```

**Why dual model?**  
The `tiny` model is fast but imprecise. Using it only for wake detection (a 2-word phrase) is accurate enough. The `base` model is used for full command transcription where accuracy matters far more.

### 5.2 Wake Word Detection

```python
# Fuzzy wake word matching (handles mispronunciations)
WAKE_WORDS = ["hey v", "hey be", "hey vi", "heyv", "hey b", "hay v", "he v", ...]

def _is_wake(text: str) -> bool:
    tl = text.lower().strip()
    for w in WAKE_WORDS:
        if w in tl:
            return True
    # Also matches if text starts with "v" after "hey"
    return bool(re.search(r'\bhey\b.{0,5}\bv\b', tl))
```

### 5.3 Noise Filtering (VAD)

Energy-based Voice Activity Detection filters out keyboard noise, fan hum, and background speech:

```python
ENERGY_THRESH = 0.018   # RMS energy threshold (silence gate)
MIN_SPEECH_RMS = 0.030  # Minimum speech energy to process
```

Audio chunks below these thresholds are discarded before Whisper even sees them.

### 5.4 Vocabulary Priming (`initial_prompt`)

The `base` Whisper model is primed with a vocabulary hint to improve recognition of app names:

```python
COMMAND_PROMPT = (
    "open YouTube, open Chrome, open browser, search for, Google, "
    "screenshot, scroll up, scroll down, volume up, volume down, "
    "next slide, previous slide, close window, minimize, maximize, ..."
)
```

This dramatically improves recognition of words like "YouTube", "WhatsApp", "Chrome" that acoustic models often mangle.

---

## 6. Command Routing — Rule Engine vs LLM

**File:** `src/voice/voice_controller.py` → `_rule_based()`

### 6.1 Rule-Based Fast-Path

Before the LLM is ever called, a regex/keyword matcher handles common commands:

```python
def _rule_based(text_lower: str, text: str) -> Optional[tuple]:
    # Returns (action, params, reply) or None

    # 1. Search commands
    m = re.search(r'search\s+(?:for\s+)?(.+?)(?:\s+on\s+google)?$', tl)
    if m:
        return ("search_web", {"query": m.group(1)}, f"Searching for '{query}'!")

    # 2. PPT controls
    if "next slide" in tl: return ("ppt_next", {}, "Next slide!")

    # 3. Volume
    if "volume up" in tl: return ("volume_up", {}, "Volume up!")

    # 4. Open specific apps/sites (with STT misrecognition tolerance)
    _SITE_MAP = [
        (["youtube", "you do", "your door", "utube"], "https://youtube.com", "YouTube"),
        (["whatsapp", "whats app"], "https://web.whatsapp.com", "WhatsApp"),
        # ... 12 more sites
    ]
```

**Why rule-based before LLM?**
- Sub-millisecond response (no network, no inference)
- 100% reliable for known commands
- Handles STT misrecognitions explicitly (e.g. "your door" → YouTube)
- Reduces LLM load for common tasks

### 6.2 Browse Mode Decision Tree

```
User types/says something
         │
         ▼
  Is it a system command?  ──YES──► Execute immediately (both modes)
  (volume/scroll/screenshot)
         │ NO
         ▼
  Browse Mode OFF?
         │
    YES ─┼─ Knowledge found? ──YES──► LLM answers from KB only
         │         │
         │        NO ──────────────► "Sorry, I don't know that"
         │                           (suggest adding to KB or enabling Browse)
    NO   │
         ▼
  Browse Mode ON?
         │
         ▼
    Web search (DuckDuckGo)
         │
    Results found? ──YES──► LLM synthesises answer in chat (no browser opens)
         │
        NO ──────────────► "Couldn't find information"
```

---

## 7. Chat System & Browse Mode

**File:** `src/ui/chat_tab.py`

### 7.1 Architecture

The `ChatTab` is a PyQt5 widget that runs all LLM calls **off the main thread** to keep the UI responsive:

```python
def _send(self):
    self._thinking_signal.emit(True, self._browse_mode)   # Show spinner
    threading.Thread(
        target=self._process, args=(text, browse_mode), daemon=True
    ).start()

def _process(self, text, browse_mode):
    # Runs in background thread
    # Emits _append_signal when done (triggers Qt UI update)
```

### 7.2 Browse Mode OFF — Knowledge-Only

```
Browse OFF:
  query → KnowledgeStore.search() → build_context()
        → Ollama.ask(query, context=knowledge_ctx)
        → [INSTRUCTION: Answer ONLY from the knowledge base context]
        → Display reply in chat
```

If the knowledge store returns nothing (score = 0 for all entries):
```
Reply: "Sorry, I don't have information about that in my knowledge base.
        💡 Enable Browse Mode or add it to the Knowledge tab."
```

### 7.3 Browse Mode ON — Web Search

Web search uses two sources, no API key needed:

**Source 1: DuckDuckGo Instant Answer API**
```
GET https://api.duckduckgo.com/?q=<query>&format=json&no_html=1
→ AbstractText, Answer fields
```

**Source 2: DuckDuckGo HTML Search Snippets**
```
GET https://html.duckduckgo.com/html/?q=<query>
→ Regex extracts <a class="result__snippet"> elements
→ HTML tags stripped → plain text
```

Both results are passed as context to Ollama with the instruction:
```
"Answer using ONLY the web search results provided.
 Do NOT suggest opening a browser. Reply directly in chat."
```

### 7.4 TTS — Listen Button

Every V reply bubble has a `🔊 Listen` button. Clicking it calls:

```python
def _speak_text(self, text):
    tts.speak(text, blocking=False)   # edge-tts neural voice
```

Auto-Speak mode reads every reply automatically when enabled.

---

## 8. Gesture Recognition Engine

**File:** `src/gesture/hand_engine.py`

### 8.1 MediaPipe Pipeline

```
Camera Frame (BGR)
       │
       ▼
MediaPipe HandLandmarker
  → 21 3D landmarks per hand (x, y, z normalised to [0,1])
       │
       ▼
GestureClassifier
  → Rule-based geometry analysis
  → Returns: gesture_name (str) + confidence (float)
       │
       ▼
GestureController
  → Hold-guard (8-frame stability check)
  → Maps gesture → SystemController action
```

### 8.2 Gesture Classification (Rule-Based Geometry)

No ML model is used for gesture classification. Instead, pure geometric rules on landmark positions:

```python
# Example: thumbs_up detection
def _is_thumbs_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    index_tip = landmarks[8]
    # Thumb pointing up, other fingers curled
    return (thumb_tip.y < thumb_ip.y and
            index_tip.y > landmarks[6].y)  # index curled
```

Supported gestures:
`pointing`, `pinch`, `peace_sign`, `open_palm`, `closed_fist`, `thumbs_up`, `thumbs_down`, `three_fingers`, `four_fingers`, `ok_sign`, `wave`, `phone_sign`

### 8.3 Motion Gestures (Swipe/Scroll)

Swipe detection uses **wrist trajectory over time**:

```python
# Track wrist position history
self._wrist_trail.append((wrist.x, wrist.y, time.time()))

# Classify motion after 6+ frames
dx = current_x - start_x
dy = current_y - start_y

if abs(dy) > SCROLL_THRESHOLD and gesture == "peace_sign":
    return "swipe_up" if dy < 0 else "swipe_down"
```

**V-sign (peace_sign) is exclusively the scroll trigger** — it cannot be a macro sequence to avoid interference.

### 8.4 Two-Hand Gestures

The engine tracks both hands independently and detects combined postures:

| Two-Hand Combo | Action |
|---------------|--------|
| Both index+thumb pinch, moving apart | Zoom In |
| Both index+thumb pinch, moving together | Zoom Out |
| Both hands open, forming X | Minimize / Maximize toggle |

---

## 9. Text-to-Speech (TTS)

**File:** `src/voice/tts_engine.py`

```
TTSEngine.speak(text)
       │
       ▼
  edge-tts available?
  ┌───YES────────────────────────────────────────────┐
  │  edge_tts.Communicate(text, "en-US-AriaNeural")  │
  │  → Generate MP3 (async)                          │
  │  → Play via pygame.mixer                         │
  └──────────────────────────────────────────────────┘
       │ NO (offline/not installed)
       ▼
  pyttsx3 (system TTS fallback)
  → Selects first female voice (Zira/Hazel/Aria)
```

**Voice:** `en-US-AriaNeural` — Microsoft Edge neural voice (natural female, warm tone).

TTS runs in a daemon thread so it never blocks the UI or gesture processing.

---

## 10. Macro System

**File:** `src/gesture/macro_engine.py`

### 10.1 What it does

Macros let you bind **sequences of gestures** to complex OS actions.

```python
# Example: double closed_fist → Select All
{
  "name": "double_fist",
  "sequence": ["closed_fist", "closed_fist"],
  "max_gap": 0.9,           # Max seconds between gestures
  "description": "Double fist → Ctrl+A"
}
```

### 10.2 Spam Prevention (Per-Macro Cooldown)

Macros have a **2.5 second global cooldown** and a **3 second per-macro cooldown** to prevent rapid repeat-firing:

```python
self._macro_cooldown = 2.5   # No macro fires within 2.5s of any macro
self._per_macro_cd   = 3.0   # Each macro has its own 3s lock
```

### 10.3 Disabled Macros

Macros that conflict with motion gestures are explicitly disabled:

```python
{
  "name": "double_peace",
  "sequence": ["peace_sign", "peace_sign"],
  "disabled": True,   # peace_sign is reserved for scroll
}
```

---

## 11. Data Flow Diagram

```
┌──────────────┐    frames     ┌────────────────┐    gesture    ┌────────────────┐
│   Webcam     │──────────────►│  HandEngine    │──────────────►│GestureController│
└──────────────┘               │  (MediaPipe)   │               │ (hold-guard)   │
                               └────────────────┘               └───────┬────────┘
                                                                         │ action
                               ┌────────────────┐               ┌───────▼────────┐
│   Microphone │──────────────►│ WakeWordDetect │               │SystemController│
└──────────────┘  audio        │ (Whisper tiny) │               │ (pyautogui)    │
                               └───────┬────────┘               └────────────────┘
                                       │ "Hey V" detected
                               ┌───────▼────────┐
                               │ Whisper base   │ ◄── full command transcription
                               └───────┬────────┘
                                       │ text command
                               ┌───────▼────────┐
                               │VoiceController │
                               │ _rule_based()  │ ──► instant reply (90% of cmds)
                               └───────┬────────┘
                                       │ no match
                               ┌───────▼────────┐
                               │ KnowledgeStore │ ──► build_context() from v_knowledge.json
                               └───────┬────────┘
                                       │ context
                               ┌───────▼────────┐
                               │ OllamaClient   │ ──► llama3.2:3b (local, CPU)
                               └───────┬────────┘
                                       │ reply + action
                               ┌───────▼────────┐
                               │  TTSEngine     │ ──► edge-tts neural voice
                               └────────────────┘
```

---

## 12. Dependencies & Stack

### Core Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `Python` | 3.11 | Runtime |
| `PyQt5` | ≥5.15 | GUI framework |
| `mediapipe` | ≥0.10 | Hand landmark detection |
| `opencv-python` | ≥4.8 | Camera capture + frame processing |
| `pyautogui` | ≥0.9 | OS automation (mouse, keyboard, scroll) |

### AI & Voice

| Package | Version | Purpose |
|---------|---------|---------|
| `openai-whisper` | latest | STT (speech-to-text), tiny + base models |
| `faster-whisper` | latest | Optional: faster CPU inference via CTranslate2 |
| `torch` | ≥2.0 | Whisper backend |
| `edge-tts` | ≥6.0 | Neural TTS (Microsoft Aria voice) |
| `pyttsx3` | ≥2.9 | TTS fallback (offline system voice) |
| `pygame` | ≥2.5 | MP3 audio playback for edge-tts |
| `sounddevice` | ≥0.4 | Microphone capture |
| `numpy` | ≥1.24 | Audio array processing |

### LLM (must be installed separately)

| Tool | Installation | Purpose |
|------|-------------|---------|
| **Ollama** | `https://ollama.com` | Local LLM server |
| **llama3.2:3b** | `ollama pull llama3.2:3b` | The language model |

### File Import (optional)

| Package | Purpose |
|---------|---------|
| `pypdf` or `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |

---

## Quick Reference — Adding Your Own Knowledge

```python
# Programmatically (e.g. in a script)
from src.voice.knowledge_store import KnowledgeStore

ks = KnowledgeStore()

# Add raw text
ks.add_text("My Company Policy", "Employees must clock in by 9am...")

# Add from file
ks.add_file("my_notes.txt")
ks.add_file("manual.pdf")
ks.add_file("procedures.docx")

# Query it
results = ks.search("what time should I arrive?", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['title']}: {r['text'][:100]}")
```

The knowledge is immediately queryable — no retraining, no restart needed.

---

*Generated for HGVCS v1.0.0 · © 2026*
