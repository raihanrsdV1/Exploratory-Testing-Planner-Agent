"""Observe a page as a compact, ref-addressable element list.

This is the web counterpart of Android's accessibility-tree dump, and it exists
because neither raw HTML nor a screenshot is a workable observation for an LLM
agent: HTML is enormous and mostly layout, and a screenshot gives no way to name
a control precisely enough to click it again.

Two decisions worth stating:

* **Identity comes from a stamped attribute, not a selector.** Every observed
  element gets ``data-etp-ref="e<N>"`` written onto it, and actions address it by
  that attribute. Generated CSS class names (``css-1x7f2b``) and nth-child paths
  both break on the next render; a stamp survives anything short of the element
  being replaced — and when it *is* replaced, the locator misses and we report
  STALE_ELEMENT honestly instead of clicking the wrong thing.

* **The accessible name is the label, not the text content.** ``aria-label``,
  then a bound ``<label>``, then placeholder, then trimmed text. This is the same
  notion of identity the Android side gets from ``content_description``, which is
  what keeps the planner's prompt vocabulary consistent across both platforms.
"""

from __future__ import annotations

# Collected in the page, not in Python: one round-trip instead of one per element.
# Returns {url, title, elements: [...], headings: [...], messages: [...]}.
_COLLECT_JS = r"""
(maxElements) => {
  const INTERACTIVE_SEL = [
    'a[href]', 'button', 'input', 'select', 'textarea', 'summary',
    '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]',
    '[role=tab]', '[role=menuitem]', '[role=switch]', '[role=combobox]',
    '[role=option]', '[role=searchbox]', '[role=textbox]',
    '[contenteditable=""]', '[contenteditable=true]', '[onclick]',
  ].join(',');

  const isVisible = (el) => {
    if (!el || !el.getClientRects || el.getClientRects().length === 0) return false;
    const s = window.getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none' || s.opacity === '0') return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };

  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim().slice(0, 100);

  const accessibleName = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return clean(aria);
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map((n) => n.innerText || n.textContent);
      if (parts.length) return clean(parts.join(' '));
    }
    if (el.id) {
      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lab) return clean(lab.innerText || lab.textContent);
    }
    const wrapping = el.closest('label');
    if (wrapping) {
      const t = clean(wrapping.innerText || wrapping.textContent);
      if (t) return t;
    }
    const ph = el.getAttribute('placeholder');
    if (ph) return clean(ph);
    const text = clean(el.innerText || el.textContent);
    if (text) return text;
    for (const attr of ['title', 'alt', 'name', 'value']) {
      const v = el.getAttribute(attr);
      if (v) return clean(v);
    }
    const img = el.querySelector('img[alt]');
    if (img) return clean(img.getAttribute('alt'));
    return '';
  };

  const roleOf = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'select') return 'select';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t === 'checkbox' || t === 'radio' || t === 'submit' || t === 'button') return t;
      if (t === 'password') return 'password';
      return 'textbox';
    }
    return 'control';
  };

  // Clear stamps from the previous observation so refs never survive a render
  // and silently address a stale element.
  document.querySelectorAll('[data-etp-ref]').forEach((el) => el.removeAttribute('data-etp-ref'));

  const out = [];
  let i = 0;
  for (const el of document.querySelectorAll(INTERACTIVE_SEL)) {
    if (out.length >= maxElements) break;
    if (!isVisible(el)) continue;
    const ref = 'e' + (++i);
    el.setAttribute('data-etp-ref', ref);
    const entry = { ref, role: roleOf(el), name: accessibleName(el) };
    if (el.disabled) entry.disabled = true;
    if (el.required || el.getAttribute('aria-required') === 'true') entry.required = true;
    if (typeof el.checked === 'boolean' && ['checkbox', 'radio'].includes(entry.role)) {
      entry.checked = el.checked;
    }
    // A checkbox's value is always "on"; `checked` above already says the useful
    // half, so reporting both spends prompt tokens on nothing.
    const valueIsNoise = ['checkbox', 'radio'].includes(entry.role);
    if (!valueIsNoise && 'value' in el && el.value !== undefined && el.value !== null) {
      const raw = String(el.value);
      // Never echo a secret back into the prompt or the logs.
      entry.value = entry.role === 'password' ? ('*'.repeat(raw.length)) : clean(raw);
      // Report the TRUE length whenever we shorten the value for display.
      // Without this the agent reads a 300-character entry back as ~100 and
      // concludes the field truncated its input — a defect report about our own
      // rendering. It happened on the first real run.
      if (raw.length > entry.value.length) entry.value_length = raw.length;
    }
    if (el.tagName.toLowerCase() === 'a' && el.getAttribute('href')) {
      entry.href = clean(el.getAttribute('href'));
    }
    if (el.tagName.toLowerCase() === 'select') {
      entry.options = Array.from(el.options).slice(0, 12).map((o) => clean(o.textContent || o.value));
    }
    out.push(entry);
  }

  const headings = Array.from(document.querySelectorAll('h1,h2,h3,[role=heading]'))
    .filter(isVisible).slice(0, 12).map((el) => clean(el.innerText || el.textContent))
    .filter(Boolean);

  // Validation / status text is the whole point of most assertions, and it is
  // almost never on an interactive element — so it needs collecting separately.
  const messages = Array.from(document.querySelectorAll(
    '[role=alert],[role=status],[aria-live],.error,.alert,.invalid-feedback,' +
    '.help-block,.form-error,[class*="error"],[class*="Error"]'
  )).filter(isVisible).slice(0, 12).map((el) => clean(el.innerText || el.textContent))
    .filter(Boolean);

  // Ordinary page content. Headings and alerts are not enough: most assertions
  // are about plain text ("Items: 0", "Total: $42", "No results"), which belongs
  // to no interactive element and would otherwise be invisible to the agent.
  const claimed = new Set([...headings, ...messages, ...out.map((e) => e.name)]);
  const texts = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node && texts.length < 30; node = walker.nextNode()) {
    const parent = node.parentElement;
    if (!parent) continue;
    if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(parent.tagName)) continue;
    if (parent.closest('[data-etp-ref]')) continue;   // already reported as a control
    const t = clean(node.textContent);
    if (t.length < 2 || claimed.has(t)) continue;
    if (!isVisible(parent)) continue;
    claimed.add(t);
    texts.push(t);
  }

  const modal = document.querySelector('[role=dialog],[role=alertdialog],dialog[open],[aria-modal=true]');

  return {
    url: location.href,
    title: document.title || '',
    elements: out,
    headings: [...new Set(headings)],
    messages: [...new Set(messages)],
    texts,
    dialog_open: !!(modal && isVisible(modal)),
  };
}
"""


async def observe(page, max_elements: int) -> dict:
    """Snapshot the page. Never raises — a failed observation is still a turn."""
    try:
        data = await page.evaluate(_COLLECT_JS, max_elements)
    except Exception as exc:  # navigation mid-evaluate, closed page, CSP oddity
        return {
            "url": _safe_url(page),
            "title": "",
            "elements": [],
            "headings": [],
            "messages": [],
            "texts": [],
            "dialog_open": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return data


def _safe_url(page) -> str:
    try:
        return page.url
    except Exception:
        return ""


def render(snap: dict) -> str:
    """Format an observation as the compact text the agent actually reads."""
    lines = [f"URL: {snap.get('url', '')}"]
    if snap.get("title"):
        lines.append(f"TITLE: {snap['title']}")
    if snap.get("dialog_open"):
        lines.append("A MODAL DIALOG IS OPEN — deal with it before anything else.")
    if snap.get("error"):
        lines.append(f"OBSERVATION ERROR: {snap['error']}")

    if snap.get("headings"):
        lines.append("HEADINGS: " + " | ".join(snap["headings"]))
    if snap.get("messages"):
        lines.append("MESSAGES ON PAGE: " + " | ".join(snap["messages"]))

    if snap.get("texts"):
        lines.append("PAGE TEXT: " + " | ".join(snap["texts"]))

    elements = snap.get("elements") or []
    if not elements:
        lines.append("INTERACTIVE ELEMENTS: none found (the page may still be loading).")
        return "\n".join(lines)

    lines.append(f"INTERACTIVE ELEMENTS ({len(elements)}):")
    for el in elements:
        lines.append("  " + _render_element(el))
    return "\n".join(lines)


def _render_element(el: dict) -> str:
    parts = [f"[{el['ref']}] {el.get('role', 'control')}"]
    name = el.get("name") or ""
    parts.append(f'"{name}"' if name else '""')
    if el.get("value"):
        parts.append(f'value="{el["value"]}"')
        if el.get("value_length"):
            parts.append(f"({el['value_length']} chars total — shown shortened by the "
                         f"observer, the field is NOT truncated)")
    if el.get("checked") is not None and el.get("role") in ("checkbox", "radio"):
        parts.append("checked" if el["checked"] else "unchecked")
    if el.get("required"):
        parts.append("required")
    if el.get("disabled"):
        parts.append("DISABLED")
    if el.get("options"):
        parts.append("options=[" + ", ".join(el["options"]) + "]")
    if el.get("href"):
        parts.append(f'href={el["href"]}')
    return " ".join(parts)


def find(snap: dict, ref: str) -> dict | None:
    """Look up an element in an observation by its ref."""
    for el in snap.get("elements") or []:
        if el.get("ref") == ref:
            return el
    return None
