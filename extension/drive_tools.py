"""In-car tools for the extension, built on the same Holdfast ToolSpec layer
(same gate, same ledger, same watchdog) as the benchmark agent."""

from __future__ import annotations

from holdfast.toolkit import ToolSpec

from .nav_backend import NavBackend


def drive_tool_specs(nav: NavBackend) -> list[ToolSpec]:
    route_ctx = lambda _args: nav.active_route_id  # noqa: E731  (state-dependent actions)

    def spec(name, desc, props, required, mutating, fn, ctx=None):
        async def handler(args):
            return await fn(**args)

        return ToolSpec(name, desc, {"type": "object", "properties": props, "required": required},
                        mutating, handler, dedupe_context=ctx)

    s = lambda d: {"type": "string", "description": d}  # noqa: E731
    return [
        spec("set_destination", "Start or change navigation to a destination. Replaces any current route.",
             {"destination": s("Destination as the user finally stated it.")}, ["destination"], True,
             nav.set_destination, route_ctx),
        spec("get_eta", "Current destination, stops and ETA.", {}, [], False, nav.get_eta, route_ctx),
        spec("find_nearby", "Find places along the route by category.",
             {"category": {"type": "string", "enum": ["charging", "coffee", "fuel", "food", "parking"],
                           "description": "Kind of place."}}, ["category"], False, nav.find_nearby, route_ctx),
        spec("add_stop", "Add a place returned by find_nearby as a stop on the route.",
             {"place_id": s("The id from a find_nearby result.")}, ["place_id"], True, nav.add_stop, route_ctx),
        spec("send_eta", "Text a contact the current destination and ETA.",
             {"contact": s("Contact name.")}, ["contact"], True, nav.send_eta, route_ctx),
        spec("cancel_navigation", "Stop navigating.", {}, [], True, nav.cancel_navigation, route_ctx),
    ]
