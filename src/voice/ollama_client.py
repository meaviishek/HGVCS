"""
OllamaClient – Sends transcribed voice commands to a local Ollama LLM
and parses the response into structured HGVCS actions.

Fixes in this version
─────────────────────
• num_gpu=0 by default → CPU inference, no CUDA OOM crashes
• auto_probe_model() → picks the best available model automatically
• Friendlier error messages (CUDA crash vs server-not-running vs model-missing)
• Reduced num_predict (100) for faster CPU responses
"""

import json
import logging
import threading
from typing import Optional, Dict, Any, Callable, List

log = logging.getLogger("hgvcs.ollama")

# ── HTTP client ────────────────────────────────────────────
try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False
    log.warning("httpx not installed — trying urllib fallback")

OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_GEN_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAG_URL = f"{OLLAMA_BASE}/api/tags"
DEFAULT_MODEL  = "llama3.2:3b"

# Preference order when auto-probing models
_MODEL_PREFERENCE = [
    "llama3.2:3b",
    "llama3.2",
    "llama3:latest",
    "llama3",
    "mistral",
    "mistral:latest",
    "phi3",
    "phi3:mini",
    "gemma2:2b",
    "qwen2:1.5b",
]

# System prompt: bilingual Hindi+English, full system control
SYSTEM_PROMPT = """You are V, an AI voice assistant for HGVCS controlling a Windows computer.
You understand commands in BOTH Hindi and English (and Hindi-English mix / Hinglish).
Respond in the same language the user spoke. Keep responses under 2 sentences, friendly.

When the user gives a command, output a JSON object on the FIRST line, then a brief spoken reply.
JSON format: {"action": "<action_name>", "params": {}}

━━━ AVAILABLE ACTIONS ━━━
MEDIA:          volume_up, volume_down, mute, pause_media, next_track, prev_track
NAVIGATION:     scroll_up, scroll_down, swipe_left, swipe_right, zoom_in, zoom_out
BROWSER:        open_browser, search_web, close_tab, new_tab, reload, go_back, go_forward
WINDOW:         close_window, minimize_window, maximize_window, show_desktop, task_switch
SYSTEM:         screenshot, copy, paste, undo, redo, save, lock_screen, type_text
PRESENTATION:   ppt_next, ppt_prev, ppt_start, ppt_end, ppt_fullscreen
OTHER:          none  (for greetings / unknown / unclear)

━━━ HINDI COMMAND EXAMPLES ━━━
"volume badhao" → volume_up
"volume kam karo" → volume_down
"band karo" → close_window
"screenshot lo" → screenshot
"browser kholo" → open_browser
"Google pe search karo cats" → search_web  (params: {"query": "cats"})
"agli slide" → ppt_next
"pichli slide" → ppt_prev
"neeche scroll karo" → scroll_down
"upar scroll karo" → scroll_up
"copy karo" → copy
"paste karo" → paste
"undo karo" → undo

━━━ ENGLISH COMMAND EXAMPLES ━━━
"open browser" → open_browser
"search for cats on Google" → search_web  (params: {"query": "cats"})
"next slide" → ppt_next
"previous slide" → ppt_prev
"take a screenshot" → screenshot
"scroll down" → scroll_down
"volume up" → volume_up
"close this window" → close_window
"switch task" → task_switch

━━━ OUTPUT FORMAT ━━━
{"action": "volume_up", "params": {}}
Turning up the volume!

{"action": "search_web", "params": {"query": "cats"}}
Searching for cats on Google!

{"action": "none", "params": {}}
Hey! Kya help chahiye? (What help do you need?)
"""

# Chat/Q&A system prompt — NO JSON action format, pure conversational answers
CHAT_SYSTEM_PROMPT = """You are V, a helpful AI assistant.
Answer the user's question directly and conversationally in 2-5 sentences.
If a KNOWLEDGE BASE section is provided below, use it as your primary source of truth.
If the knowledge base does not cover the topic, answer from your general training knowledge.
Do NOT output any JSON. Do NOT mention actions or commands. Just answer naturally.
"""


class OllamaClient:
    """
    Sends user transcripts to local Ollama and returns (action, reply).
    Thread-safe — calls are blocking but can be wrapped in threads.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 url: str = OLLAMA_GEN_URL,
                 timeout: float = 120.0,
                 use_gpu: bool = False):
        self._url     = url
        self._timeout = timeout
        self._lock    = threading.Lock()
        # use_gpu=False → num_gpu=0 → pure CPU → no CUDA OOM
        self._num_gpu = -1 if use_gpu else 0

        # Auto-probe: find the best model that's actually installed
        self._model = self._probe_model(model)
        log.info(f"OllamaClient ready — model={self._model} gpu={'auto' if use_gpu else 'CPU-only'}")

    # ── public API ─────────────────────────────────────────

    def ask(self, user_text: str, extra_context: str = "") -> Dict[str, Any]:
        """
        Send *user_text* to Ollama using the VOICE COMMAND prompt.
        Returns dict: {"action": str, "params": dict, "reply": str, "raw": str}
        Use chat_ask() for Q&A / knowledge queries.
        """
        with self._lock:
            raw = self._request(user_text, extra_context)
        return self._parse(raw, user_text)

    def chat_ask(self, user_text: str, knowledge_ctx: str = "") -> str:
        """
        Send *user_text* to Ollama using the CHAT / Q&A prompt.
        knowledge_ctx: pre-built context string from KnowledgeStore.build_context().
        Returns a plain reply string (no JSON parsing needed).
        """
        with self._lock:
            return self._chat_request(user_text, knowledge_ctx)

    def ask_async(self, user_text: str,
                  callback: Callable[[Dict[str, Any]], None]):
        """Non-blocking version — calls *callback* with result dict."""
        def _run():
            result = self.ask(user_text)
            callback(result)
        threading.Thread(target=_run, daemon=True).start()

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            if _HTTPX_OK:
                r = httpx.get(f"{OLLAMA_BASE}/", timeout=2.0)
                return r.status_code == 200
            else:
                import urllib.request
                urllib.request.urlopen(f"{OLLAMA_BASE}/", timeout=2)
                return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of installed model names."""
        try:
            if _HTTPX_OK:
                r = httpx.get(OLLAMA_TAG_URL, timeout=5.0)
                r.raise_for_status()
                data = r.json()
            else:
                import urllib.request
                with urllib.request.urlopen(OLLAMA_TAG_URL, timeout=5) as resp:
                    data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            log.debug(f"list_models failed: {e}")
            return []

    # ── internal ──────────────────────────────────────────

    def _probe_model(self, requested: str) -> str:
        """
        Find the best available model.
        Priority: requested → preference list → any installed model.
        Falls back to DEFAULT_MODEL string if Ollama is offline (will fail at request time).
        """
        installed = self.list_models()
        if not installed:
            log.warning("Ollama not reachable — will retry at request time")
            return requested  # keep as-is, will fail gracefully later

        # Normalize: strip ":latest" suffix for comparison
        installed_norm = {m.split(":")[0]: m for m in installed}
        installed_set  = set(installed)

        # 1. Exact match of requested
        if requested in installed_set:
            return requested
        if requested.split(":")[0] in installed_norm:
            found = installed_norm[requested.split(":")[0]]
            log.info(f"Model '{requested}' → using '{found}'")
            return found

        # 2. Walk preference list
        for pref in _MODEL_PREFERENCE:
            if pref in installed_set:
                log.info(f"Preferred model '{requested}' not found → using '{pref}'")
                return pref
            base = pref.split(":")[0]
            if base in installed_norm:
                found = installed_norm[base]
                log.info(f"Preferred model '{pref}' → using '{found}'")
                return found

        # 3. Any installed model
        fallback = installed[0]
        log.warning(f"No preferred model found → using first available: '{fallback}'")
        return fallback

    def _request(self, user_text: str, extra_context: str = "") -> str:
        # Build prompt: system + optional knowledge context + user message
        if extra_context:
            prompt = f"{SYSTEM_PROMPT}\n\n{extra_context}\nUser: {user_text}"
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_text}"
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 100,          # keep short for faster CPU inference
                "num_gpu":     self._num_gpu,
                "num_thread":  6,
                "top_k":       10,
                "top_p":       0.9,
            }
        }
        try:
            if _HTTPX_OK:
                # Use split timeout: fast connect (5 s), long read for CPU inference
                _timeout = httpx.Timeout(
                    connect=5.0,
                    read=self._timeout,
                    write=10.0,
                    pool=5.0,
                )
                with httpx.Client(timeout=_timeout) as client:
                    resp = client.post(self._url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "")
            else:
                return self._urllib_request(payload)
        except Exception as e:
            err_str = str(e)
            if "500" in err_str or "CUDA" in err_str.upper():
                log.error(
                    f"Ollama model '{self._model}' crashed (likely out of VRAM). "
                    f"Try a smaller model or set ollama_use_gpu: false in config. "
                    f"Error: {e}"
                )
                # Try to re-probe and switch to a smaller model
                self._try_fallback()
            elif "Connection" in err_str or "refused" in err_str:
                log.error(f"Ollama server not running at {OLLAMA_BASE} — start with: ollama serve")
            else:
                log.error(f"Ollama request failed: {e}")
            return ""

    def _chat_request(self, user_text: str, knowledge_ctx: str = "") -> str:
        """
        Chat/Q&A request using CHAT_SYSTEM_PROMPT.
        knowledge_ctx is injected between the system prompt and the question
        so the model treats it as authoritative context.
        Returns raw text reply (no JSON parsing).
        """
        if knowledge_ctx:
            prompt = (
                f"{CHAT_SYSTEM_PROMPT}\n"
                f"KNOWLEDGE BASE:\n{knowledge_ctx}\n"
                f"END OF KNOWLEDGE BASE\n\n"
                f"User: {user_text}\nV:"
            )
        else:
            prompt = f"{CHAT_SYSTEM_PROMPT}\n\nUser: {user_text}\nV:"

        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,          # more natural for Q&A
                "num_predict": 300,          # allow longer knowledge answers
                "num_gpu":     self._num_gpu,
                "num_thread":  6,
                "top_k":       40,
                "top_p":       0.9,
            }
        }
        try:
            if _HTTPX_OK:
                _timeout = httpx.Timeout(
                    connect=5.0,
                    read=self._timeout,
                    write=10.0,
                    pool=5.0,
                )
                with httpx.Client(timeout=_timeout) as client:
                    resp = client.post(self._url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "").strip()
            else:
                return self._urllib_chat_request(payload)
        except Exception as e:
            log.error(f"Ollama chat request failed: {e}")
            return ""

    def _urllib_chat_request(self, payload: dict) -> str:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self._url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body).get("response", "").strip()
        except Exception as e:
            log.error(f"urllib Ollama chat request failed: {e}")
            return ""

    def _try_fallback(self):
        """After a crash, try switching to a smaller installed model."""
        installed = self.list_models()
        current_idx = None
        for i, pref in enumerate(_MODEL_PREFERENCE):
            if pref == self._model or pref.split(":")[0] == self._model.split(":")[0]:
                current_idx = i
                break

        if current_idx is None:
            current_idx = -1

        # Try models ranked after the current one in preference list
        for pref in _MODEL_PREFERENCE[current_idx + 1:]:
            if pref in installed:
                old = self._model
                self._model = pref
                log.warning(f"Switching model: '{old}' → '{self._model}' (CUDA fallback)")
                return
            base = pref.split(":")[0]
            for m in installed:
                if m.split(":")[0] == base:
                    old = self._model
                    self._model = m
                    log.warning(f"Switching model: '{old}' → '{self._model}' (CUDA fallback)")
                    return

    def _urllib_request(self, payload: dict) -> str:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self._url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body).get("response", "")
        except Exception as e:
            log.error(f"urllib Ollama request failed: {e}")
            return ""

    def _parse(self, raw: str, original: str) -> Dict[str, Any]:
        """Extract JSON action + spoken reply from Ollama response."""
        result = {
            "action": "none",
            "params": {},
            "reply":  "",
            "raw":    raw,
        }

        if not raw:
            result["reply"] = (
                "I'm having trouble connecting to my language model. "
                "Make sure Ollama is running and the model fits in your RAM."
            )
            return result

        lines = raw.strip().splitlines()

        # First JSON line = action
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    result["action"] = obj.get("action", "none")
                    result["params"] = obj.get("params", {})
                    # Remaining lines = spoken reply
                    reply_lines = [l.strip() for l in lines[i+1:] if l.strip()]
                    result["reply"] = " ".join(reply_lines)
                    return result
                except json.JSONDecodeError:
                    pass

        # No valid JSON found — treat entire response as conversational reply
        result["reply"] = raw.strip()
        log.warning(f"Could not parse action JSON from: {raw[:100]}")
        return result
