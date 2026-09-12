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

  // A modal is whatever sits on top, marked up as a dialog or not: the outermost
  // fixed layer over most of the viewport at the screen centre hides everything outside it.
  const vw = window.innerWidth, vh = window.innerHeight;
  let overlay = null;
  for (let n = document.elementFromPoint(vw / 2, vh / 2);
       n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
    const st = getComputedStyle(n);
    if (st.position !== 'fixed' || st.pointerEvents === 'none') continue;
    const r = n.getBoundingClientRect();
    if (r.width >= vw * 0.6 && r.height >= vh * 0.6) overlay = n;
  }
  const isCovered = (el) => !!overlay && !overlay.contains(el);

  const out = [];
  let i = 0;
  for (const el of document.querySelectorAll(INTERACTIVE_SEL)) {
    if (out.length >= maxElements) break;
    if (!isVisible(el)) continue;
    const ref = 'e' + (++i);
    el.setAttribute('data-etp-ref', ref);
    const entry = { ref, role: roleOf(el), name: accessibleName(el) };
    if (isCovered(el)) entry.covered = true;
    if (el.disabled) entry.disabled = true;
    if (el.required || el.getAttribute('aria-required') === 'true') entry.required = true;
    // Native HTML5 validation renders as a browser tooltip that is in no node,
    // so a form refusing to submit looked to the agent like a page that simply
    // ignored the click. It probed six different ways and tripped the wandering
    // guard — while testing validation, which is the thing it could not see.
    if (typeof el.checkValidity === 'function' && !el.disabled) {
      try {
        if (!el.checkValidity()) {
          entry.invalid = true;
          const vm = (el.validationMessage || '').trim();
          if (vm) entry.validation = clean(vm);
        }
      } catch (e) { /* not a form control */ }
    }
    if (typeof el.checked === 'boolean' && ['checkbox', 'radio'].includes(entry.role)) {
      entry.checked = el.checked;
    }
    // A checkbox's value is always "on"; `checked` above already says the useful
    // half, so reporting both spends prompt tokens on nothing.
    const valueIsNoise = ['checkbox', 'radio'].includes(entry.role);
    if (!valueIsNoise && 'value' in el && el.value !== undefined && el.value !== null) {
      const raw = String(el.value);
      // Never echo a secret into the prompt or the logs — but do not render it as
      // a string of asterisks either. A model read '********' back out of its own
      // observation and typed it as the new password on a real account; only the
      // site's password policy stopped the change. Describe the field instead of
      // showing something that looks like its contents.
      if (entry.role === 'password') {
        if (raw.length) entry.filled_chars = raw.length;
      } else {
        entry.value = clean(raw);
        // clean() trims, so a whitespace-only value becomes '' and the renderer
        // then omits it entirely. The agent reads its own typing back as gone and
        // reports that the field "discards input silently" — a defect report about
        // our own rendering, and exactly the kind of false finding that costs a
        // whole investigation. Whitespace is content: say so.
        if (!entry.value && raw.length) entry.whitespace_chars = raw.length;
      }
      // Report the TRUE length whenever we shorten the value for display.
      // Without this the agent reads a 300-character entry back as ~100 and
      // concludes the field truncated its input — a defect report about our own
      // rendering. It happened on the first real run.
      if (entry.value && raw.length > entry.value.length) entry.value_length = raw.length;
    }
    if (el.tagName.toLowerCase() === 'a' && el.getAttribute('href')) {
      entry.href = clean(el.getAttribute('href'));
    }
    if (el.tagName.toLowerCase() === 'select') {
      entry.options = Array.from(el.options).slice(0, 12).map((o) => clean(o.textContent || o.value));
    }
    out.push(entry);
  }

  // Second pass: elements that are clickable but say so only in CSS.
  // React attaches handlers with addEventListener, so there is no `onclick`
  // attribute and often no role or tabindex either — a plain
  // <div class="sec-card-header"> that opens an accordion is invisible to every
  // selector above. `cursor: pointer` is the one signal such controls reliably
  // carry, because the page has to look clickable to a human.
  // Only the OUTERMOST element of each pointer cluster is taken, and never one
  // that wraps a control already reported, so a button inside a clickable card
  // does not appear twice.
  for (const el of document.querySelectorAll('*')) {
    if (out.length >= maxElements) break;
    if (el.hasAttribute('data-etp-ref')) continue;
    if (el.closest('[data-etp-ref]')) continue;          // inside something reported
    if (el.querySelector('[data-etp-ref]')) continue;     // wraps something reported
    if (getComputedStyle(el).cursor !== 'pointer') continue;
    if (!isVisible(el)) continue;
    const name = accessibleName(el);
    if (!name) continue;
    // A parent already taken in this pass covers its children.
    if (out.some((o) => o.inferred && document.querySelector(
        '[data-etp-ref="' + o.ref + '"]')?.contains(el))) continue;
    const ref = 'e' + (++i);
    el.setAttribute('data-etp-ref', ref);
    const entry = { ref, role: 'clickable', name, inferred: true };
    if (isCovered(el)) entry.covered = true;
    const expanded = el.getAttribute('aria-expanded');
    if (expanded !== null) entry.expanded = expanded === 'true';
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

  // Human-verification widgets render inside a cross-origin iframe, so nothing
  // above can see them. The agent was therefore submitting a form that silently
  // refused, with no clue why, and wandering off to look for a different survey.
  // It cannot solve one and must never try; it only needs to KNOW one is there.
  const captchaFrame = Array.from(document.querySelectorAll('iframe')).find((f) => {
    const src = (f.getAttribute('src') || '').toLowerCase();
    return src.includes('recaptcha') || src.includes('hcaptcha')
        || src.includes('turnstile') || src.includes('captcha');
  });
  const captchaHost = document.querySelector(
    '.g-recaptcha, .h-captcha, .cf-turnstile, [data-sitekey]');
  // A SOLVED challenge does not go away - reCAPTCHA just shows a green tick, and
  // the iframe and .g-recaptcha host both remain. Testing for presence alone kept
  // reporting "CAPTCHA on the page" after a person had already solved it, so the
  // run waited out its whole 300s and then failed the test as unsolved.
  // Every provider exposes a response token that is empty until solved; that, not
  // the widget, is what says whether the way is clear.
  const token = document.querySelector(
    'textarea[name="g-recaptcha-response"], textarea#g-recaptcha-response, ' +
    'textarea[name="h-captcha-response"], input[name="cf-turnstile-response"]');
  const solved = !!(token && String(token.value || '').trim().length > 0);
  const captcha = !!(captchaFrame || captchaHost) && !solved;

  const modal = document.querySelector('[role=dialog],[role=alertdialog],dialog[open],[aria-modal=true]');
  const coveredCount = out.filter((e) => e.covered).length;
  // An overlay that hides nothing is layout (a fixed full-screen app shell), not a modal.
  const blocking = overlay && coveredCount > 0 ? overlay : (modal && isVisible(modal) ? modal : null);
  const heading = blocking && blocking.querySelector('h1,h2,h3,h4,[role=heading]');

  // The browser's own constraint bubble ("Please fill out this field.") is chrome,
  // not DOM: a form it refuses to submit looks exactly like a dead click, and the
  // agent then repeats the click until the livelock guard ends the test.
  // :user-invalid matches only fields the person actually tried to submit, so an
  // untouched form reports nothing.
  let invalid = [];
  try {
    document.querySelectorAll(':user-invalid').forEach((el) => {
      const msg = el.validationMessage || '';
      if (!msg) return;
      const ref = el.getAttribute('data-etp-ref');
      invalid.push((ref ? '[' + ref + '] ' : '') + (accessibleName(el) || roleOf(el)) + ': ' + msg);
    });
  } catch (e) { invalid = []; }   // older engines reject the selector outright

  return {
    url: location.href,
    title: document.title || '',
    elements: out,
    headings: [...new Set(headings)],
    messages: [...new Set(messages)],
    texts,
    validation_messages: invalid,
    dialog_open: !!blocking,
    dialog_title: heading ? clean(heading.innerText || heading.textContent) : '',
    covered_count: coveredCount,
    captcha,
  };
}
"""


# A single-page app renders after `domcontentloaded`, so the first look at a new
# route frequently finds nothing at all. 16% of observations in a real DataGhurhi
# run reported zero controls, and the agent had to spend a whole turn on `wait`
# to recover — roughly a sixth of its step budget. Re-reading here costs a second
# of wall clock and saves a step, and it also covers in-page route changes that
# fire no navigation event for the driver to wait on.
_SETTLE_RETRIES = 3
_SETTLE_DELAY_MS = 700


async def observe(page, max_elements: int) -> dict:
    """Snapshot the page, re-reading briefly if it looks unrendered.

    Never raises — a failed observation is still a turn.
    """
    data = await _observe_once(page, max_elements)
    for _ in range(_SETTLE_RETRIES):
        if data.get("elements") or data.get("error"):
            break
        try:
            await page.wait_for_timeout(_SETTLE_DELAY_MS)
        except Exception:
            break
        data = await _observe_once(page, max_elements)
    return data


async def _observe_once(page, max_elements: int) -> dict:
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
    if snap.get("captcha"):
        lines.append(
            "A HUMAN-VERIFICATION CHALLENGE (CAPTCHA) IS ON THIS PAGE. You cannot "
            "solve it and must not try. Anything behind it - submitting this form, "
            "for example - is unreachable. If your test needs that, call finish "
            "with success=false and say the test is BLOCKED BY CAPTCHA. Anything "
            "in front of it (field validation, layout, navigation) is still testable."
        )
    if snap.get("dialog_open"):
        title = snap.get("dialog_title")
        # "Deal with it" was read as "close it": the agent cancelled the very dialog
        # its test needed, reopened it, cancelled again, and burned the run.
        line = ("A MODAL DIALOG IS OPEN" + (f' ("{title}")' if title else "")
                + ". If it is where your test happens, work INSIDE it and do not close "
                  "it; only cancel it when it is genuinely in your way.")
        if snap.get("covered_count"):
            line += (" Controls marked COVERED are behind it and cannot be used until it "
                     "is closed with its own buttons.")
        lines.append(line)

    if snap.get("validation_messages"):
        lines.append("THE BROWSER REFUSED TO SUBMIT THE FORM — "
                     + " | ".join(snap["validation_messages"])
                     + ". The form was never sent; fix these fields, then submit again. "
                       "Clicking submit again unchanged will do nothing.")
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
    if el.get("filled_chars"):
        # Deliberately not the characters: the agent must never be able to copy a
        # secret out of the observation and type it somewhere else.
        parts.append(f"(contains {el['filled_chars']} hidden characters)")
    elif el.get("role") == "password":
        parts.append("(empty)")
    if el.get("value"):
        parts.append(f'value="{el["value"]}"')
        if el.get("value_length"):
            parts.append(f"({el['value_length']} chars total — shown shortened by the "
                         f"observer, the field is NOT truncated)")
    elif el.get("whitespace_chars"):
        # "Empty" and "holds spaces" are different answers to a blank-input test.
        parts.append(f"value=whitespace only ({el['whitespace_chars']} space characters "
                     f"— your typing WAS accepted; the field is not empty)")
    if el.get("checked") is not None and el.get("role") in ("checkbox", "radio"):
        parts.append("checked" if el["checked"] else "unchecked")
    if el.get("required"):
        parts.append("required")
    if el.get("validation"):
        parts.append(f'REJECTED BY THE BROWSER: "{el["validation"]}"')
    elif el.get("invalid"):
        parts.append("INVALID (the browser will refuse to submit this)")
    if el.get("disabled"):
        parts.append("DISABLED")
    if el.get("covered"):
        parts.append("COVERED (behind the open dialog)")
    if el.get("inert"):
        parts.append("ALREADY TRIED — it did nothing; choose something else")
    if el.get("expanded") is not None:
        parts.append("expanded" if el["expanded"] else "collapsed")
    if el.get("inferred"):
        parts.append("(clickable by style — not a standard control)")
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
