"""
Detect whether the current foreground window looks like an LLM target.

  - Desktop: window title and process name (ChatGPT, Claude, Gemini, Copilot, Perplexity, Poe,
    plus title substrings like claude.ai, chatgpt, gemini, copilot, perplexity).
  - Chromium (Brave, Edge, Chrome): CDP tab URL/title when available; else title fallback.
    URLs: chat.openai.com, claude.ai, gemini.google.com, copilot.microsoft.com, perplexity.ai,
    poe.com, and any URL containing \"/chat\" or \"assistant\" (labeled \"Web LLM\").
  - Ports: Brave 9222 (GUARDRAIL_CDP_PORT), Edge 9223 (GUARDRAIL_EDGE_CDP_PORT),
    Chrome 9224 (GUARDRAIL_CHROME_CDP_PORT).
  - Use get_active_llm_name() for a single string label or None.

  Non-browser windows use title matching only. cleaned_title from _clean_browser_title.
"""

import json
import os
import re
import ctypes
from ctypes import wintypes
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


# Browser suffixes to strip from window/tab titles (order matters: longer first)
_BROWSER_SUFFIXES = [
    " - Personal - Microsoft Edge",
    " - Personal - Microsoft\u200b Edge",
    " - Microsoft\u200b Edge",
    " - Microsoft Edge",
    " - Brave",
    " - Google Chrome",
    " - Chromium",
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


def _agent_from_title(title_raw: str) -> str:
    """Derive an agent label from the foreground/tab title text (desktop + browser tab titles)."""
    title_lower = (title_raw or "").lower()
    if not title_lower:
        return ""

    # Longer URL fragments first (subset of browser/CDP URL hints that appear in titles).
    for needle, label in [
        ("claude.ai", "Claude"),
        ("chat.openai.com", "ChatGPT"),
        ("gemini.google.com", "Gemini"),
        ("copilot.microsoft.com", "Copilot"),
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
    ]:
        if needle in title_lower:
            return label

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


# URL host/path fragments -> agent label (longer / more specific first).
_URL_AGENT_PATTERNS = [
    ("gemini.google.com", "Gemini"),
    ("chat.openai.com", "ChatGPT"),
    ("chatgpt.com", "ChatGPT"),
    ("claude.ai", "Claude"),
    ("copilot.microsoft.com", "Copilot"),
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


def _get_active_tab_from_cdp(window_title: str, port: int) -> Optional[dict]:
    """
    Query CDP /json/list; find the tab whose title best matches window_title or whose URL
    matches a known LLM host. Return {"url": str, "title": str} or None.
    """
    url = f"http://127.0.0.1:{port}/json/list"
    try:
        req = Request(url)
        with urlopen(req, timeout=1.5) as resp:
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
    # Browser tabs: any URL with /chat or assistant-style paths (after known hosts).
    if "/chat" in url_lower or "assistant" in url_lower:
        return "Web LLM"
    return ""


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
    ]
    for p, agent_label in known_processes:
        if p in process_name:
            cleaned = _clean_browser_title(title_raw)
            if debug:
                print(f"[LLM detect] Agent={agent_label!r} Title={cleaned!r}", flush=True)
            return (True, agent_label, None, cleaned or None)

    # Chromium browsers: prefer CDP to get tab title/url.
    is_chromium = any(k in process_name for k in ["brave", "chrome", "msedge"])
    if is_chromium:
        port = _get_cdp_port_for_process(process_name)
        tab = _get_active_tab_from_cdp(title_raw, port)
        if tab:
            url = tab.get("url") or ""
            tab_title = tab.get("title") or ""

            agent = _agent_from_url(url) or _agent_from_title(tab_title)
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
        return (False, "", None, None)

    # Non-browser: title match.
    agent = _agent_from_title(title_raw)
    cleaned = _clean_browser_title(title_raw)
    if agent:
        if debug:
            print(f"[LLM detect] Agent={agent!r} Title={cleaned!r}", flush=True)
        return (True, agent, None, cleaned or None)

    return (False, "", None, None)


def get_active_llm_name() -> Optional[str]:
    """
    Return the detected LLM product name (e.g. \"ChatGPT\", \"Claude\") for the
    active foreground window, or None if it is not recognized as an LLM target.
    """
    is_llm, agent, _, _ = is_active_window_llm()
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


def is_llm_window() -> bool:
    """True if the foreground window is treated as an LLM paste target."""
    return bool(is_active_window_llm()[0])


def log_guardrail_active_window_banner(is_llm: bool, label: str) -> None:
    """
    Optional debug line before clipboard scoring. Disabled when GUARDRAIL_DEBUG_WINDOW
    is 0, false, or no.
    """
    v = os.environ.get("GUARDRAIL_DEBUG_WINDOW", "1").strip().lower()
    if v in ("0", "false", "no", ""):
        return
    suffix = "scoring paste" if is_llm else "skipping"
    print(f"[guardrail] active window: {label} — {suffix}", flush=True)
