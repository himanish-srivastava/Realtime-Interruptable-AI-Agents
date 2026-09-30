"""Deterministic mock navigation backend for the in-car extension.

Places are fictional. Route computation and place search are *async and slow*
on purpose (configurable delays) so the demo shows background work: the
conversation continues while a route is being computed.

State rules that make the demo meaningful:
* exactly one active route; a new destination supersedes the old one,
* an ETA message is sent at most once per (contact, route); after a reroute the
  next message to the same contact is an explicit "update", never a duplicate,
* unknown places change nothing.
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable, Optional

PLACES = {
    "city airport": {"aliases": ["airport"], "drive_min": 34},
    "central station": {"aliases": ["station", "train station"], "drive_min": 18},
    "harbor market": {"aliases": ["market", "harbor", "harbour market"], "drive_min": 22},
    "tech park": {"aliases": ["office park", "the tech park"], "drive_min": 27},
    "city hospital": {"aliases": ["hospital"], "drive_min": 12},
    "home": {"aliases": ["my house", "house"], "drive_min": 25},
}
POIS = [
    {"id": "CHG-1", "category": "charging", "name": "VoltHub Fast Charging", "detour_min": 4},
    {"id": "CHG-2", "category": "charging", "name": "GreenPlug Station", "detour_min": 9},
    {"id": "CAF-1", "category": "coffee", "name": "Roadside Roasters", "detour_min": 3},
    {"id": "CAF-2", "category": "coffee", "name": "Bean There", "detour_min": 6},
    {"id": "FUEL-1", "category": "fuel", "name": "Metro Fuel", "detour_min": 2},
    {"id": "FOOD-1", "category": "food", "name": "Lane Seven Diner", "detour_min": 7},
    {"id": "PARK-1", "category": "parking", "name": "Central Garage", "detour_min": 1},
]


def _title(key: str) -> str:
    return " ".join(w.capitalize() for w in key.split())


def resolve_place(text: str) -> Optional[str]:
    t = " ".join(text.lower().replace("the ", " ").split())
    for key, info in PLACES.items():
        if t == key or t in info["aliases"]:
            return key
    for key, info in PLACES.items():
        if key in t or any(a in t for a in info["aliases"]):
            return key
    return None


class NavBackend:
    def __init__(self, route_delay_s: float = 2.0, search_delay_s: float = 1.0,
                 on_change: Optional[Callable[[dict], None]] = None) -> None:
        self.route_delay_s = route_delay_s
        self.search_delay_s = search_delay_s
        self.on_change = on_change
        self.active_route: Optional[dict] = None
        self.superseded: list[dict] = []
        self.cancelled: list[dict] = []
        self.messages: list[dict] = []
        self.route_count = 0
        self.last_event = ""

    @property
    def active_route_id(self) -> Optional[str]:
        return self.active_route["route_id"] if self.active_route else None

    def snapshot(self) -> dict:
        return copy.deepcopy({
            "active_route": self.active_route, "superseded": self.superseded, "cancelled": self.cancelled,
            "messages": self.messages, "last_event": self.last_event,
        })

    def _event(self, text: str) -> None:
        self.last_event = text
        if self.on_change:
            self.on_change(self.snapshot())

    async def set_destination(self, destination: str) -> dict:
        key = resolve_place(destination)
        if key is None:
            return {"status": "error", "message": f"unknown destination {destination!r}",
                    "known_places": [_title(k) for k in PLACES]}
        name = _title(key)
        if self.active_route and self.active_route["destination"] == name:
            return {"status": "success", "unchanged": True, **self._route_view()}
        await asyncio.sleep(self.route_delay_s)  # route computation runs in the background
        replaced = None
        if self.active_route:
            replaced = self.active_route["destination"]
            self.superseded.append({**self.active_route, "superseded_at": time.time()})
        self.route_count += 1
        self.active_route = {"route_id": f"R{self.route_count}", "destination": name,
                             "eta_min": PLACES[key]["drive_min"], "stops": []}
        self._event(f"route: {name} ({self.active_route['route_id']}, {self.active_route['eta_min']} min)"
                    + (f", replaces {replaced}" if replaced else ""))
        return {"status": "success", "replaced": replaced, **self._route_view()}

    def _route_view(self) -> dict:
        r = self.active_route or {}
        return {"route_id": r.get("route_id"), "destination": r.get("destination"),
                "eta_min": r.get("eta_min"), "stops": list(r.get("stops", []))}

    async def get_eta(self) -> dict:
        if not self.active_route:
            return {"status": "error", "message": "no active route"}
        return {"status": "success", **self._route_view()}

    async def find_nearby(self, category: str) -> dict:
        await asyncio.sleep(self.search_delay_s)
        results = [dict(p) for p in POIS if p["category"] == category.lower().strip()]
        return {"status": "success", "category": category, "results": results,
                "current_eta_min": self.active_route["eta_min"] if self.active_route else None}

    async def add_stop(self, place_id: str) -> dict:
        poi = next((p for p in POIS if p["id"].lower() == place_id.lower().strip()), None)
        if poi is None:
            return {"status": "error", "message": f"unknown place id {place_id!r}"}
        if not self.active_route:
            return {"status": "error", "message": "no active route; set a destination first"}
        if poi["id"] in self.active_route["stops"]:
            return {"status": "success", "already_added": True, **self._route_view()}
        self.active_route["stops"].append(poi["id"])
        self.active_route["eta_min"] += poi["detour_min"]
        self._event(f"stop: {poi['name']} added (+{poi['detour_min']} min)")
        return {"status": "success", "stop": poi["name"], **self._route_view()}

    async def send_eta(self, contact: str) -> dict:
        if not self.active_route:
            return {"status": "error", "message": "no active route"}
        who = contact.strip().lower()
        rid = self.active_route["route_id"]
        if any(m["contact_key"] == who and m["route_id"] == rid for m in self.messages):
            return {"status": "already_sent", "contact": contact, "route_id": rid}
        kind = "update" if any(m["contact_key"] == who for m in self.messages) else "initial"
        r = self.active_route
        text = (f"{'Update: ' if kind == 'update' else ''}Heading to {r['destination']}, "
                f"arriving in about {r['eta_min']} minutes.")
        self.messages.append({"contact": contact, "contact_key": who, "route_id": rid, "kind": kind, "text": text})
        self._event(f"message: {kind} ETA to {contact}")
        return {"status": "sent", "contact": contact, "kind": kind, "text": text}

    async def cancel_navigation(self) -> dict:
        if not self.active_route:
            return {"status": "success", "message": "no active route"}
        self.cancelled.append(self.active_route)
        dest = self.active_route["destination"]
        self.active_route = None
        self._event(f"cancel: navigation to {dest}")
        return {"status": "success", "cancelled": dest}
