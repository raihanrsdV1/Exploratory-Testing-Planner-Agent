"""Render every Mermaid block in docs/DIAGRAMS.md to a PNG in docs/diagrams/.

    ./venv/bin/python scripts/render_diagrams.py

Uses headless Chrome plus a vendored mermaid.min.js — no Node toolchain needed.
Each diagram is measured first, then screenshotted at 2x device scale so the PNG
is sharp enough to drop into slides or a report.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "DIAGRAMS.md"
OUT = ROOT / "docs" / "diagrams"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;background:#fff}}
  #wrap{{display:inline-block;padding:24px;background:#fff}}
  .mermaid{{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}}
</style><script>{js}</script></head><body>
<div id="wrap"><pre class="mermaid">{src}</pre></div>
<script>
  // useMaxWidth:true (the default) scales every SVG down to the viewport, which
  // produced 348px-wide PNGs regardless of the diagram's real size.
  const noMax={{useMaxWidth:false}};
  mermaid.initialize({{startOnLoad:false,theme:"base",securityLevel:"loose",
    flowchart:noMax, sequence:noMax, er:noMax, state:noMax, gantt:noMax,
    themeVariables:{{fontSize:"15px",fontFamily:'-apple-system,"Segoe UI",Helvetica,Arial,sans-serif'}}}});
  mermaid.run().then(()=>{{
    const w=document.getElementById("wrap").getBoundingClientRect();
    document.title="SIZE:"+Math.ceil(w.width)+"x"+Math.ceil(w.height);
  }}).catch(e=>{{document.title="ERR:"+e.message;}});
</script></body></html>"""


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-+", "-", s)


def chrome(args: list[str]) -> str:
    r = subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                        "--allow-file-access-from-files", *args],
                       capture_output=True, text=True, timeout=180)
    return r.stdout


def main() -> int:
    if not Path(CHROME).exists():
        sys.exit("Google Chrome not found — it is what renders the diagrams.")
    js_path = ROOT / ".mermaid-cache" / "mermaid.min.js"
    if not js_path.exists():
        js_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetching mermaid → {js_path}")
        subprocess.run(["curl", "-sL", CDN, "-o", str(js_path)], check=True)
    js = js_path.read_text(encoding="utf-8")

    md = SRC.read_text(encoding="utf-8")
    blocks = re.findall(r"^## (\d+)\. (.+?)\n(.*?)```mermaid\n(.*?)```",
                        md, re.S | re.M)
    if not blocks:
        sys.exit("No '## N. Title' + mermaid blocks found in docs/DIAGRAMS.md")
    OUT.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp())
    ok = fail = 0
    index = []
    for num, title, _intro, src in blocks:
        name = f"{int(num):02d}-{slug(title)}"
        html = tmp / f"{name}.html"
        html.write_text(PAGE.format(js=js, src=src.strip()), encoding="utf-8")
        url = html.as_uri()

        # pass 1 — render and read the measured size back out of <title>
        dom = chrome(["--window-size=3000,2400", "--virtual-time-budget=9000",
                      "--dump-dom", url])
        m = re.search(r"<title>SIZE:(\d+)x(\d+)</title>", dom)
        if not m:
            err = re.search(r"<title>ERR:(.*?)</title>", dom)
            print(f"  ✗ {name}: {err.group(1)[:90] if err else 'did not render'}")
            fail += 1
            continue
        w, h = int(m.group(1)), int(m.group(2))

        # pass 2 — screenshot at exactly that size, 2x for a crisp image
        png = OUT / f"{name}.png"
        chrome([f"--window-size={w},{h}", "--force-device-scale-factor=2",
                "--virtual-time-budget=9000", f"--screenshot={png}", url])
        if png.exists() and png.stat().st_size > 1000:
            print(f"  ✓ {png.name}  ({w}x{h} @2x, {png.stat().st_size//1024} KB)")
            index.append({"n": int(num), "title": title.strip(), "file": png.name})
            ok += 1
        else:
            print(f"  ✗ {name}: screenshot empty")
            fail += 1

    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  {ok} rendered, {fail} failed → {OUT.relative_to(ROOT)}/")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
