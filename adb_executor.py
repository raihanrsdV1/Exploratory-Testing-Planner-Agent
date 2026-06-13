import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
import requests

# Load .env file manually at startup if it exists
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Config
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9100").rstrip("/")
PROJECT = os.getenv("PROJECT_NAME", "contacts-app")
APP_NAME = os.getenv("APP_NAME", "contacts app")
DEVICE_ID = "emulator-5554"  # Your running Pixel 9 emulator
DUMP_FILE_ON_DEVICE = "/sdcard/window_dump.xml"
DUMP_FILE_LOCAL = "./window_dump.xml"


def run_adb(args):
    """Run an adb command on the target device."""
    cmd = ["adb", "-s", DEVICE_ID] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ADB Error: {' '.join(cmd)}")
        print(result.stderr)
    return result.stdout


def dump_ui():
    """Dump the current UI hierarchy XML and parse it."""
    run_adb(["shell", "uiautomator", "dump", DUMP_FILE_ON_DEVICE])
    run_adb(["pull", DUMP_FILE_ON_DEVICE, DUMP_FILE_LOCAL])
    if not os.path.exists(DUMP_FILE_LOCAL):
        return None
    try:
        return ET.parse(DUMP_FILE_LOCAL)
    except Exception as e:
        print("XML Parse Error:", e)
        return None


def parse_bounds(bounds_str):
    """Convert boundary string '[x1,y1][x2,y2]' to center (x, y) coordinates."""
    # Matches [x1,y1][x2,y2]
    match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2) // 2, (y1 + y2) // 2
    return None


def find_node(tree, label):
    """
    Search UI XML tree for a node matching the label.
    Checks: text, content-desc, and resource-id (case-insensitive substring match).
    """
    if tree is None:
        return None
    root = tree.getroot()
    label_lower = label.lower()

    # Heuristic 1: Exact/substring match on text or content-desc
    for node in root.iter("node"):
        text = node.get("text", "").lower()
        desc = node.get("content-desc", "").lower()
        res_id = node.get("resource-id", "").lower()

        if label_lower in text or label_lower in desc or label_lower in res_id:
            return node

    return None


def find_input_field(tree, label):
    """
    Search for an input field (EditText) associated with a label.
    Usually the label text is in a TextView, and the EditText is next to it.
    """
    if tree is None:
        return None
    root = tree.getroot()
    nodes = list(root.iter("node"))

    for idx, node in enumerate(nodes):
        text = node.get("text", "").lower()
        desc = node.get("content-desc", "").lower()

        if label.lower() in text or label.lower() in desc:
            for search_idx in range(idx + 1, min(idx + 5, len(nodes))):
                candidate = nodes[search_idx]
                c_class = candidate.get("class", "")
                if "EditText" in c_class or candidate.get("focusable") == "true":
                    return candidate
            return node
    return None


def execute_tap(x, y):
    print(f"  Action: Tapping center coordinates ({x}, {y})")
    run_adb(["shell", "input", "tap", str(x), str(y)])
    time.sleep(1.5)


def execute_input(x, y, text):
    # Tap to focus
    execute_tap(x, y)
    
    # Clear existing text by sending backspaces
    print("  Action: Clearing text field")
    run_adb(["shell", "input", "keyevent", "123"]) # KEYCODE_MOVE_END
    for _ in range(35):
         run_adb(["shell", "input", "keyevent", "67"]) # KEYCODE_DEL
        
    # Send text (replace spaces with %s since adb input requires it on some shells)
    safe_text = text.replace(" ", "%s")
    print(f"  Action: Typing text '{text}'")
    run_adb(["shell", "input", "text", safe_text])
    time.sleep(1.5)


def get_interactive_elements(tree):
    """Extract interactive and readable elements from tree to make a compact context."""
    if tree is None:
        return []
    root = tree.getroot()
    elements = []
    for node in root.iter("node"):
        bounds = node.get("bounds", "")
        if not bounds or bounds == "[0,0][0,0]":
            continue
        text = node.get("text", "").strip()
        desc = node.get("content-desc", "").strip()
        res_id = node.get("resource-id", "").strip()
        cls = node.get("class", "").strip()
        clickable = node.get("clickable", "") == "true"
        focusable = node.get("focusable", "") == "true"
        
        if text or desc or clickable or focusable:
            elements.append({
                "text": text,
                "desc": desc,
                "res_id": res_id.split("/")[-1] if "/" in res_id else res_id,
                "class": cls.split(".")[-1] if "." in cls else cls,
                "bounds": bounds,
                "clickable": clickable
            })
    return elements


def resolve_action_with_llm(step, elements):
    """Ask OpenRouter LLM to select the target bounds and action for a step."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
        
    elements_summary = ""
    for idx, el in enumerate(elements):
        elements_summary += f"- Idx {idx}: text='{el['text']}', desc='{el['desc']}', id='{el['res_id']}', class='{el['class']}', bounds='{el['bounds']}', clickable={el['clickable']}\n"
        
    prompt = f"""You are a Mobile GUI Execution Agent.
You are running a test case step on an Android device emulator.

Current Step to execute: "{step}"

Here is a list of interactive UI elements currently visible on the screen:
{elements_summary}

Based on the step description and the visible elements, decide the next action to perform.
If the step says to click, tap, select, or open an element, select "tap".
If the step says to enter, input, type, or write text, select "input".
If the step is a wait or check, select "wait".
If no matching element is visible or no action can be taken, select "none".

Return STRICT JSON only with this schema:
{{
  "action": "tap" | "input" | "wait" | "none",
  "text_to_type": "text to type if action is input, otherwise empty",
  "target_bounds": "[x1,y1][x2,y2] (copy the bounds of the element to interact with)",
  "reason": "short explanation"
}}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Exploratory Testing Planner Agent",
    }
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "qwen/qwen3-32b"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                import json
                return json.loads(match.group(0))
    except Exception as e:
        print("  LLM Action Resolution failed:", e)
    return None


def run_testcase(tc):
    print(f"\n--- Executing Test: {tc.get('test_case_id')} - {tc.get('title')} ---")
    steps = tc.get("steps", [])
    expected = tc.get("expected_result", "")

    for step in steps:
        print(f"\nExecuting step: {step}")
        tree = dump_ui()
        if tree is None:
            print("Failed to get UI hierarchy. Skipping step.")
            continue

        # Try LLM agent resolver first
        elements = get_interactive_elements(tree)
        llm_decision = resolve_action_with_llm(step, elements)
        
        if llm_decision and llm_decision.get("action") in {"tap", "input"}:
            action = llm_decision["action"]
            bounds = llm_decision.get("target_bounds", "")
            reason = llm_decision.get("reason", "")
            print(f"  [LLM Decision] Action: {action.upper()} | Target Bounds: {bounds} | Reason: {reason}")
            
            coords = parse_bounds(bounds)
            if coords:
                if action == "tap":
                    execute_tap(coords[0], coords[1])
                elif action == "input":
                    val = llm_decision.get("text_to_type", "Test Input")
                    execute_input(coords[0], coords[1], val)
                continue
            else:
                print("  [LLM Decision] Failed to parse target bounds coordinates.")

        # Fallback to regex matcher if LLM is unavailable or signals "none"
        print("  [Fallback] Running local regex matcher...")
        click_match = re.search(r"(?:click|tap|press)\s+['\"]([^'\"]+)['\"]", step, re.IGNORECASE)
        input_match = re.search(r"(?:enter|type|input)\s+['\"]([^'\"]+)['\"]\s+(?:into|in)\s+['\"]([^'\"]+)['\"]", step, re.IGNORECASE)

        if input_match:
            val, label = input_match.groups()
            node = find_input_field(tree, label)
            if node is not None:
                coords = parse_bounds(node.get("bounds", ""))
                if coords:
                    execute_input(coords[0], coords[1], val)
                    continue
            print(f"  Warning: Could not locate input field matching '{label}'")

        elif click_match:
            label = click_match.group(1)
            node = find_node(tree, label)
            if node is not None:
                coords = parse_bounds(node.get("bounds", ""))
                if coords:
                    execute_tap(coords[0], coords[1])
                    continue
            print(f"  Warning: Could not locate clickable element matching '{label}'")
        
        else:
            # General fallback: check if quotes contain labels
            quotes = re.findall(r"['\"]([^'\"]+)['\"]", step)
            if quotes:
                label = quotes[-1]
                node = find_node(tree, label)
                if node is not None:
                    coords = parse_bounds(node.get("bounds", ""))
                    if coords:
                        execute_tap(coords[0], coords[1])
                        continue
            print(f"  Action Skipped: Step '{step}' could not be parsed.")

    # Post-execution verification
    time.sleep(2)
    tree_after = dump_ui()
    
    # Save a screenshot of the final state
    screenshot_path = f"./logs/{tc.get('test_case_id')}_result.png"
    os.makedirs("./logs", exist_ok=True)
    run_adb(["shell", "screencap", "-p", f"/sdcard/{tc.get('test_case_id')}.png"])
    run_adb(["pull", f"/sdcard/{tc.get('test_case_id')}.png", screenshot_path])
    print(f"Saved execution screenshot to: {screenshot_path}")

    # Check assertions in XML
    verdict = "failed"
    notes = "Expected UI condition not met."
    
    if tree_after:
        root = tree_after.getroot()
        expected_words = re.findall(r"[a-zA-Z0-9]+", expected.lower())
        stop_words = {"user", "system", "app", "error", "warning", "message", "displays", "shows"}
        keywords = [w for w in expected_words if len(w) > 3 and w not in stop_words]
        
        matches = 0
        for node in root.iter("node"):
            text = node.get("text", "").lower()
            desc = node.get("content-desc", "").lower()
            for kw in keywords:
                if kw in text or kw in desc:
                    matches += 1
                    
        if keywords and (matches / len(keywords)) >= 0.5:
            verdict = "pass"
            notes = f"UI matched expected result keywords: {keywords}"
        else:
            notes = f"Failed. Expected text: '{expected}' not found on screen."
            
    print(f"Execution Result: {verdict.upper()} ({notes})")
    return verdict, notes


def main(rounds=3):
    print("==================================================")
    print(f"Starting Fully Automated QA Loop on device '{DEVICE_ID}'")
    print("==================================================")
    
    # Check gateway status
    try:
        requests.get(f"{GATEWAY_URL}/health")
    except Exception:
        print(f"Error: Gateway not running at {GATEWAY_URL}. Please start start.sh first.")
        return

    # Ingest SRS
    print("Ingesting SRS...")
    ingest_payload = {"project": PROJECT, "source_path": "./SRS1.txt"}
    requests.post(f"{GATEWAY_URL}/srs/ingest", json=ingest_payload)

    # Get first test case
    tc_payload = {
        "project": PROJECT,
        "app_name": APP_NAME,
        "objective": "generate next high-value non-duplicate test case",
        "top_k": 8,
        "max_new_tokens": 700,
        "enable_thinking": False,
    }
    
    print("Planning first test case...")
    r = requests.post(f"{GATEWAY_URL}/agent/next-testcase", json=tc_payload)
    data = r.json()
    tc = data.get("next_testcase", {})

    for i in range(1, rounds + 1):
        if not tc:
            print("No test case generated. Stopping.")
            break
            
        verdict, notes = run_testcase(tc)
        
        # Log verdict and request next test case
        print("Logging verdict and requesting next test case...")
        log_payload = {
            "project": PROJECT,
            "app_name": APP_NAME,
            "test_case_id": tc.get("test_case_id"),
            "title": tc.get("title"),
            "verdict": verdict,
            "notes": notes,
            "area": tc.get("area", "general"),
            "top_k": 8,
            "max_new_tokens": 700,
            "enable_thinking": False,
        }
        
        r = requests.post(f"{GATEWAY_URL}/agent/log-verdict-and-next", json=log_payload)
        out = r.json()
        tc = out.get("next", {}).get("next_testcase", {})

    print("\n==================================================")
    print("Autonomous test session completed!")
    print("==================================================")

if __name__ == "__main__":
    main(rounds=3)
