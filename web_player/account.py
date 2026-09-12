"""What the signed-in account holds right now, read between test cases.

The planner otherwise only knows the account state it was configured with, which
goes stale the moment a test creates something — and then every later test is
told the account is empty and rebuilds the same scaffolding.
"""

from __future__ import annotations

from collections import Counter

_NAME_KEYS = ("title", "name", "label")

_FETCH_JS = """
async ({paths, key}) => {
  const headers = {};
  if (key) {
    const token = localStorage.getItem(key);
    if (token) headers['Authorization'] = 'Bearer ' + token;
  }
  const out = [];
  for (const path of paths) {
    try {
      const r = await fetch(path, {credentials: 'same-origin', headers});
      let body = null;
      try { body = await r.json(); } catch (e) {}
      out.push({path, status: r.status, body});
    } catch (e) {
      out.push({path, status: 0, body: null});
    }
  }
  return out;
}
"""


def safe_paths(paths) -> list[str]:
    """Same-site paths only: a probe carries the session's bearer token."""
    return [p for p in paths or ()
            if p.startswith("/") and not p.startswith("//") and "\\" not in p]


async def read(session, cfg) -> str:
    """One-line summary of the account's current contents, or "" if unavailable."""
    paths = safe_paths(getattr(cfg, "WEB_ACCOUNT_PROBES", ()))
    if not paths:
        return ""
    try:
        await session.reset_to_base()
        results = await session.page.evaluate(
            _FETCH_JS, {"paths": paths, "key": getattr(cfg, "WEB_ACCOUNT_PROBE_BEARER_KEY", "") or ""})
    except Exception:
        return ""
    return summarize(results)


def summarize(results, max_names: int = 6) -> str:
    """[{path, status, body}, ...] -> 'projects: 5 — Alpha ×4, Beta'."""
    parts: list[str] = []
    for r in results or []:
        path = r.get("path", "?")
        if r.get("status") != 200:
            parts.append(f"{path} unreadable (HTTP {r.get('status')})")
            continue
        for name, items in _collections(r.get("body"), path):
            if not items:
                parts.append(f"{name}: none")
                continue
            counts = Counter(_label(i) for i in items)
            shown = [f"{t} ×{n}" if n > 1 else t for t, n in counts.most_common(max_names)]
            more = f", +{len(counts) - max_names} more" if len(counts) > max_names else ""
            parts.append(f"{name}: {len(items)} — {', '.join(shown)}{more}")
    return "; ".join(parts)


def _collections(body, path: str) -> list[tuple[str, list]]:
    if isinstance(body, list):
        return [(path.rstrip("/").rsplit("/", 1)[-1] or path, body)]
    if not isinstance(body, dict):
        return []
    found = [(k, v) for k, v in body.items() if isinstance(v, list)]
    if not found:
        for v in body.values():
            if isinstance(v, dict):
                found += [(k, vv) for k, vv in v.items() if isinstance(vv, list)]
    return found


def _label(item) -> str:
    if isinstance(item, dict):
        for key in _NAME_KEYS:
            if item.get(key):
                return str(item[key])[:60]
        return "untitled"
    return str(item)[:60]
