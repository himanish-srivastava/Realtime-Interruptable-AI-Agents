import sys


def render(snap: dict, stream=sys.stderr) -> None:
    r = snap.get("active_route")
    lines = [f"+-- NAV STATE -- {snap.get('last_event', '')}"]
    if r:
        stops = ", ".join(r["stops"]) or "none"
        lines.append(f"| ACTIVE  {r['route_id']} -> {r['destination']}  ETA {r['eta_min']} min  stops: {stops}")
    else:
        lines.append("| ACTIVE  (no route)")
    for old in snap.get("superseded", []):
        lines.append(f"| replaced {old['route_id']} -> {old['destination']}")
    for m in snap.get("messages", []):
        lines.append(f"| sent to {m['contact']} [{m['kind']}] {m['text']}")
    lines.append("+" + "-" * 60)
    print("\n".join(lines), file=stream, flush=True)

