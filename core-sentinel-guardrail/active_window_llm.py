"""
Detect whether the current foreground window looks like an LLM target.

  - Desktop: window title and process name (ChatGPT, Claude, Gemini, Copilot, Perplexity, Poe,
    Comet, plus title substrings like claude.ai, chatgpt, gemini, copilot, perplexity, comet).
  - Chromium (Brave, Edge, Chrome, Comet): CDP tab URL/title when available; else title fallback.
    URLs: chat.openai.com, claude.ai, gemini.google.com, copilot.microsoft.com, perplexity.ai,
    comet.perplexity.ai, poe.com, any URL containing \"perplexity\", and any URL with \"/chat\"
    or \"assistant\" (labeled \"Web LLM\").
  - CDP: tries browser-specific port first, then CDP_PORTS (9222, 9223, 9224, 9229).
    Brave 9222 (GUARDRAIL_CDP_PORT), Edge 9223 (GUARDRAIL_EDGE_CDP_PORT),
    Chrome 9224 (GUARDRAIL_CHROME_CDP_PORT), Comet 9229 (GUARDRAIL_COMET_CDP_PORT).
  - Use get_active_llm_name() for a single string label or None.

  Non-browser windows use title matching only. cleaned_title from _clean_browser_title.

  Debug (terminal / stderr):
  - Set GUARDRAIL_DEBUG_WINDOW=1 or GUARDRAIL_DEBUG_LLM_DETECT=1 to log each Ctrl+V path
    (see log_guardrail_active_window_banner). Uses stderr so lines show in `python main.py` consoles.
"""

import json
import os
import re
import sys
import ctypes
from ctypes import wintypes
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


# --- Consolidated LLM hints (titles, processes, URL fragments; includes Comet / Perplexity) ---
KNOWN_LLM_SUBSTRINGS = [
    "chatgpt",
    "claude",
    "gemini",
    "copilot",
    "perplexity",
    "comet",
    "poe",
    "mistral",
    "grok",
    "llama",
    "ollama",
    "openai",
    "anthropic",
    "hugging",
    "together",
]

LLM_TITLES = KNOWN_LLM_SUBSTRINGS

LLM_PROCESSES = [
    "chatgpt.exe",
    "perplexity.exe",
    "claude.exe",
    "gemini.exe",
    "poe.exe",
    "copilot.exe",
    "windowscopilot",
    "brave.exe",
    "chrome.exe",
    "msedge.exe",
    "comet.exe",
    "comet browser",
]

LLM_URLS = [
    "gemini.google.com",
    "chat.openai.com",
    "chatgpt.com",
    "claude.ai",
    "copilot.microsoft.com",
    "comet.perplexity.ai",
    "perplexity.ai",
    "poe.com",
    "bing.com/chat",
    "anthropic.com",
]

# Chrome DevTools Protocol: try browser default first, then these in order (Comet may use 9229).
CDP_PORTS: tuple[int, ...] = (9222, 9223, 9224, 9229)

_CDP_HTTP_TIMEOUT_SEC = 0.5

# Browser suffixes to strip from window/tab titles (order matters: longer first)
_BROWSER_SUFFIXES = [
    " - Personal - Microsoft Edge",
    " - Personal - Microsoft\u200b Edge",
    " - Microsoft\u200b Edge",
    " - Microsoft Edge",
    " - Brave",
    " - Google Chrome",
    " - Chromium",
    " - Comet",
    " - Firefox",
]


def _clean_browser_title(raw_title: str) -> str:
    """
    Remove known browser suffixes and trim whitespace/zero-width characters.
    Example: "ChatGPT and 5 more pages - Personal - Microsoft Edge" -> "ChatGPT and 5 more pages"
    """
    if not raw_title:
        return ""
    # Remove zero-width and other invisible chars (e.g. U+200B) first.
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", raw_title).strip()

    s_lower = s.lower()
    for suffix in _BROWSER_SUFFIXES:
        suf_clean = re.sub(r"[\u200b\ufeff]", "", suffix).strip().lower()
        if suf_clean and s_lower.endswith(suf_clean):
            s = s[: -len(suf_clean)].strip()
            s_lower = s.lower()
            break

    # Also handle common variations where the browser name is the suffix.
    for browser_suffix in [
        " - microsoft edge",
        " - brave",
        " - google chrome",
        " - chrome",
        " - edge",
        " - comet",
        " - firefox",
    ]:
        if s_lower.endswith(browser_suffix):
            s = s[: -len(browser_suffix)].strip()
            s_lower = s.lower()
            break

    # Edge/Chrome often include: "ChatGPT and 5 more pages"
    m = re.match(r"^(.*?)\s+and\s+\d+\s+more\s+pages?$", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    return s.strip()


def _title_matches_known_llm_substrings(title_raw: str) -> bool:
    """True if window or tab title contains any KNOWN_LLM_SUBSTRING (case-insensitive)."""
    t = (title_raw or "").lower()
    if not t:
        return False
    return any(s in t for s in KNOWN_LLM_SUBSTRINGS)


def _agent_from_title(title_raw: str) -> str:
    """Derive an agent label from the foreground/tab title text (desktop + browser tab titles)."""
    title_lower = (title_raw or "").lower()
    if not title_lower:
        return ""

    if "comet" in title_lower:
        return "Comet"
    if "perplexity" in title_lower:
        return "Perplexity"

    # Longer URL fragments first (subset of browser/CDP URL hints that appear in titles).
    for needle, label in [
        ("claude.ai", "Claude"),
        ("chat.openai.com", "ChatGPT"),
        ("gemini.google.com", "Gemini"),
        ("copilot.microsoft.com", "Copilot"),
        ("comet.perplexity.ai", "Comet"),
        ("perplexity.ai", "Perplexity"),
        ("poe.com", "Poe"),
        ("chatgpt.com", "ChatGPT"),
        ("anthropic.com", "Claude"),
    ]:
        if needle in title_lower:
            return label

    # Required substrings (any window title containing these counts as LLM-related).
    for needle, label in [
        ("chatgpt", "ChatGPT"),
        ("gemini", "Gemini"),
        ("copilot", "Copilot"),
        ("perplexity", "Perplexity"),
        ("comet", "Comet"),
    ]:
        if needle in title_lower:
            return label

    # KNOWN_LLM_SUBSTRINGS fallback labels (title heuristic).
    for sub in ("mistral", "grok", "llama", "ollama", "openai", "anthropic", "hugging", "together"):
        if sub in title_lower:
            return sub.title()

    # Desktop / tab names: explicit product names (word-boundary where ambiguous).
    if re.search(r"\bchatgpt\b", title_lower):
        return "ChatGPT"
    if re.search(r"\bclaude\b", title_lower):
        return "Claude"
    if re.search(r"\bgemini\b", title_lower):
        return "Gemini"
    if re.search(r"\bcopilot\b", title_lower):
        return "Copilot"
    if re.search(r"\bperplexity\b", title_lower):
        return "Perplexity"
    if re.search(r"\bpoe\b", title_lower):
        return "Poe"

    # Extra keywords from env (e.g. GUARDRAIL_LLM_TITLE_KEYWORDS=chatgpt.com,my chat)
    extra = os.environ.get("GUARDRAIL_LLM_TITLE_KEYWORDS", "").strip()
    if extra:
        for part in extra.split(","):
            kw = part.strip().lower()
            if kw and kw in title_lower:
                return part.strip()

    return ""


def _get_foreground_window() -> int:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    return int(user32.GetForegroundWindow())


def _get_window_title(hwnd: int) -> str:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(wintypes.HWND(hwnd), buf, len(buf))
    return buf.value or ""


def _get_process_name_from_pid(pid: int) -> str:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        psapi.GetModuleBaseNameW(handle, None, buf, len(buf))
        return buf.value or ""
    finally:
        kernel32.CloseHandle(handle)


def _get_pid_from_hwnd(hwnd: int) -> int:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value)


# URL host/path fragments -> agent label (longer / more specific first). Kept in sync with LLM_URLS.
_URL_AGENT_PATTERNS = [
    ("gemini.google.com", "Gemini"),
    ("chat.openai.com", "ChatGPT"),
    ("chatgpt.com", "ChatGPT"),
    ("claude.ai", "Claude"),
    ("copilot.microsoft.com", "Copilot"),
    ("comet.perplexity.ai", "Comet"),
    ("perplexity.ai", "Perplexity"),
    ("poe.com", "Poe"),
    ("bing.com/chat", "Copilot"),
    ("anthropic.com", "Claude"),
]

# CDP port per browser (env overrides)
_CDP_PORTS = {
    "brave": ("GUARDRAIL_CDP_PORT", 9222),
    "chrome": ("GUARDRAIL_CHROME_CDP_PORT", 9224),
    "msedge": ("GUARDRAIL_EDGE_CDP_PORT", 9223),
    "comet": ("GUARDRAIL_COMET_CDP_PORT", 9229),
}


def _get_cdp_port_for_process(process_name: str) -> int:
    process_lower = process_name.lower()
    for key, (env_key, default) in _CDP_PORTS.items():
        if key in process_lower:
            try:
                return int(os.environ.get(env_key, str(default)))
            except ValueError:
                return default
    return int(os.environ.get("GUARDRAIL_CDP_PORT", "9222"))


def _get_active_tab_from_cdp_with_fallback(window_title: str, process_name: str) -> Optional[dict]:
    """Try CDP /json on the browser's default port, then CDP_PORTS in order (Comet may differ)."""
    primary = _get_cdp_port_for_process(process_name)
    seen: set[int] = set()
    order: list[int] = []
    for p in (primary, *CDP_PORTS):
        if p not in seen:
            seen.add(p)
            order.append(p)
    for port in order:
        tab = _get_active_tab_from_cdp(window_title, port)
        if tab:
            return tab
    return None


def _get_active_tab_from_cdp(window_title: str, port: int) -> Optional[dict]:
    """
    Query CDP /json/list; find the tab whose title best matches window_title or whose URL
    matches a known LLM host. Return {"url": str, "title": str} or None.
    """
    url = f"http://127.0.0.1:{port}/json/list"
    try:
        req = Request(url)
        with urlopen(req, timeout=_CDP_HTTP_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, ValueError, KeyError):
        return None
    if not isinstance(data, list):
        return None
    window_clean = _clean_browser_title(window_title or "").lower()
    candidates = []
    for target in data:
        if target.get("type") != "page":
            continue
        tab_title = (target.get("title") or "").strip()
        tab_url = (target.get("url") or "").strip()
        if not tab_url:
            continue
        tab_title_lower = tab_title.lower()
        # Score: exact title match > title contains > URL host match
        if window_clean and tab_title_lower:
            if window_clean == tab_title_lower:
                return {"url": tab_url, "title": tab_title}
            if window_clean in tab_title_lower or tab_title_lower in window_clean:
                candidates.append((2, {"url": tab_url, "title": tab_title}))
            elif window_clean.startswith(tab_title_lower) or tab_title_lower.startswith(window_clean):
                candidates.append((1, {"url": tab_url, "title": tab_title}))
        agent = _agent_from_url(tab_url)
        if agent:
            candidates.append((0, {"url": tab_url, "title": tab_title}))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    return None


def _agent_from_url(url: str) -> str:
    """Return agent label if url matches a known LLM host/path rule, else ''."""
    url_lower = (url or "").lower()
    if not url_lower:
        return ""
    for pattern, label in _URL_AGENT_PATTERNS:
        if pattern in url_lower:
            return label
    if "perplexity" in url_lower:
        return "Perplexity"
    # Browser tabs: any URL with /chat or assistant-style paths (after known hosts).
    if "/chat" in url_lower or "assistant" in url_lower:
        return "Web LLM"
    return ""


def _llm_fallback_from_title(
    title_raw: str, *, debug: bool = False
) -> Optional[tuple[bool, str, Optional[str], Optional[str]]]:
    """If window or cleaned title matches KNOWN_LLM_SUBSTRINGS, return an LLM tuple."""
    for text in (title_raw, _clean_browser_title(title_raw)):
        if not text:
            continue
        if not _title_matches_known_llm_substrings(text):
            continue
        agent = _agent_from_title(text)
        if not agent:
            continue
        cleaned = _clean_browser_title(title_raw)
        if debug:
            print(
                f"[LLM detect] Agent={agent!r} Title={cleaned!r} (substring fallback)",
                flush=True,
            )
        return (True, agent, None, cleaned or None)
    return None


def is_active_window_llm(*, debug: bool = False) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    Return (is_llm, agent_name, url_or_none, cleaned_title_or_none).

    For Chromium browsers we try CDP first to get tab URL and title; then derive
    agent_name from URL and cleaned_title by stripping browser suffixes.
    When CDP is unavailable, fall back to title keywords / browser label and
    cleaned_title from the foreground window title only.
    Non-LLM windows return (False, "", None, None).
    """
    hwnd = _get_foreground_window()
    if not hwnd:
        if debug:
            print("[LLM detect] No foreground window", flush=True)
        return (False, "", None, None)

    title_raw = _get_window_title(hwnd)
    pid = _get_pid_from_hwnd(hwnd)
    process_name = _get_process_name_from_pid(pid).lower()

    # Desktop app process names (non-browser) => use agent label mapping.
    known_processes = [
        ("chatgpt.exe", "ChatGPT"),
        ("perplexity.exe", "Perplexity"),
        ("claude.exe", "Claude"),
        ("gemini.exe", "Gemini"),
        ("poe.exe", "Poe"),
        ("copilot.exe", "Copilot"),
        ("windowscopilot", "Copilot"),
        ("comet.exe", "Comet"),
    ]
    for p, agent_label in known_processes:
        if p in process_name:
            cleaned = _clean_browser_title(title_raw)
            if debug:
                print(f"[LLM detect] Agent={agent_label!r} Title={cleaned!r}", flush=True)
            return (True, agent_label, None, cleaned or None)

    # Chromium browsers (incl. Comet): prefer CDP to get tab title/url; try multiple CDP ports.
    is_chromium = any(k in process_name for k in ["brave", "chrome", "msedge", "comet"])
    if is_chromium:
        tab = _get_active_tab_from_cdp_with_fallback(title_raw, process_name)
        if tab:
            url = tab.get("url") or ""
            tab_title = tab.get("title") or ""

            agent = _agent_from_url(url) or _agent_from_title(tab_title)
            if not agent and (
                _title_matches_known_llm_substrings(tab_title)
                or _title_matches_known_llm_substrings(title_raw)
            ):
                agent = _agent_from_title(tab_title) or _agent_from_title(title_raw)
            cleaned = _clean_browser_title(tab_title)
            # If tab titles are like "<page> - ChatGPT", remove redundant suffix.
            if agent:
                cleaned_lower = cleaned.lower()
                agent_lower = agent.lower()
                for sep in [" - ", " — "]:
                    suffix = f"{sep}{agent_lower}"
                    if cleaned_lower.endswith(suffix):
                        cleaned = cleaned[: -len(suffix)].strip()
                        break

            if agent:
                if debug:
                    url_short = url[:80] + "..." if len(url) > 80 else url
                    print(
                        f"[LLM detect] Agent={agent!r} URL={url_short!r} Title={cleaned!r}",
                        flush=True,
                    )
                return (True, agent, url or None, cleaned or None)

        # CDP failed/unhelpful => fallback to title-only detection.
        agent = _agent_from_title(title_raw)
        cleaned = _clean_browser_title(title_raw)
        if agent:
            if debug:
                print(f"[LLM detect] Agent={agent!r} Title={cleaned!r}", flush=True)
            return (True, agent, None, cleaned or None)
        fb = _llm_fallback_from_title(title_raw, debug=debug)
        if fb is not None:
            return fb
        return (False, "", None, None)

    # Non-browser: title match.
    agent = _agent_from_title(title_raw)
    cleaned = _clean_browser_title(title_raw)
    if agent:
        if debug:
            print(f"[LLM detect] Agent={agent!r} Title={cleaned!r}", flush=True)
        return (True, agent, None, cleaned or None)

    fb = _llm_fallback_from_title(title_raw, debug=debug)
    if fb is not None:
        return fb
    return (False, "", None, None)


# When CDP / URL heuristics miss (e.g. no --remote-debugging-port), match common IDE & chat titles.
LLM_TITLE_KEYWORDS: tuple[str, ...] = (
    "claude",
    "chatgpt",
    "gemini",
    "copilot",
    "perplexity",
    "grok",
    "openai",
    "anthropic",
    "mistral",
    "llama",
    "poe",
    "huggingface",
    "cursor",
    "vscode",
    "visual studio code",
    "windsurf",
    "zed",
    "comet",
    "deepseek",
    "ollama",
)


def detect_llm_window(*, debug: bool = False) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    Primary is_active_window_llm() plus raw title keyword fallback (IDEs, browsers without CDP).
    Use this for hook gating, tray UI, and get_active_llm_name() so behavior stays consistent.
    """
    is_llm, agent_name, url, cleaned_title = is_active_window_llm(debug=debug)
    if not is_llm:
        raw_title = get_active_window_title().lower()
        for kw in LLM_TITLE_KEYWORDS:
            if kw in raw_title:
                is_llm = True
                agent_name = agent_name or kw.title()
                break
    return is_llm, agent_name, url, cleaned_title


def get_active_llm_name() -> Optional[str]:
    """
    Return the detected LLM product name (e.g. \"ChatGPT\", \"Claude\") for the
    active foreground window, or None if it is not recognized as an LLM target.
    """
    is_llm, agent, _, _ = detect_llm_window()
    if is_llm and agent:
        return agent
    return None


def get_foreground_app_label() -> str:
    """
    Short human-readable label for the foreground window (for debug), e.g. \"Notepad\"
    from \"Untitled - Notepad\".
    """
    hwnd = _get_foreground_window()
    if not hwnd:
        return "none"
    title_raw = _get_window_title(hwnd)
    if title_raw.strip():
        cleaned = (_clean_browser_title(title_raw) or title_raw).strip()
        for sep in (" — ", " - "):
            if sep in cleaned:
                tail = cleaned.rsplit(sep, 1)[-1].strip()
                if tail:
                    return tail[:60]
        return cleaned[:60] or "unknown"
    pid = _get_pid_from_hwnd(hwnd)
    proc = _get_process_name_from_pid(pid)
    if proc:
        base = proc.rsplit("\\", 1)[-1]
        if base.lower().endswith(".exe"):
            return base[:-4][:60]
        return base[:60]
    return "unknown"


def get_active_window_title() -> str:
    """Raw title string of the foreground window (for debugging / scripts)."""
    hwnd = _get_foreground_window()
    if not hwnd:
        return ""
    return _get_window_title(hwnd) or ""


def is_llm_window() -> bool:
    """True if the foreground window is treated as an LLM paste target."""
    return bool(detect_llm_window()[0])


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def llm_debug_enabled() -> bool:
    """
    True when terminal should show verbose LLM detection (agent, URL, title).
    Same flags as log_guardrail_active_window_banner:
    GUARDRAIL_DEBUG_WINDOW=1 or GUARDRAIL_DEBUG_LLM_DETECT=1
    """
    return _env_truthy("GUARDRAIL_DEBUG_WINDOW") or _env_truthy("GUARDRAIL_DEBUG_LLM_DETECT")


def log_guardrail_active_window_banner(is_llm: bool, label: str) -> None:
    """
    Optional debug line before clipboard scoring (stderr, flushed).

    Enable with either:
      GUARDRAIL_DEBUG_WINDOW=1
      GUARDRAIL_DEBUG_LLM_DETECT=1
    """
    if not llm_debug_enabled():
        return
    title_snip = (get_active_window_title() or "")[:120]
    suffix = "scoring paste" if is_llm else "skipping (LLM-only mode)"
    print(
        f"[guardrail] active window: {label} — {suffix} | title={title_snip!r}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    # Quick check (no env vars): python active_window_llm.py
    is_llm, agent, url, ct = detect_llm_window(debug=True)
    print("detect_llm_window():")
    print(f"  is_llm={is_llm!r}")
    print(f"  agent_name={agent!r}")
    print(f"  url={url!r}")
    print(f"  cleaned_title={ct!r}")
    print(f"  raw_title={get_active_window_title()!r}")
