"""The 12 FDB-v3 tools, defined as ToolSpecs that call the benchmark's own
``MockAPIRegistry`` (from the harness's ``mock_apis.py``).

Two schema modes (config ``tools.schema``):

* ``template``  - identical parameters and requiredness to the reference agent
                  ``lk_agent_tool.py``.
* ``extended``  - (default) same tools, but parameters the user may never state
                  are optional instead of required (``search_apartments``
                  ``bedrooms``/``max_price``), and two optional filters exist
                  (``search_apartments.pets_allowed``, ``search_products.category``).
                  The reference schema forces the model to invent values for
                  requests like "any 2-bedroom in Denver". This is disclosed in
                  the README; switch back with ``tools.schema: template``.

Mock backend shim: the mock ``search_apartments`` function requires
``bedrooms`` and ``max_price`` positionally. When the user did not state them,
the shim passes neutral internal defaults *to the mock only*; the telemetry log
records exactly the arguments the agent sent.

Backend calls run in a worker thread (``asyncio.to_thread``) because the
harness's latency injector uses blocking ``time.sleep``; the conversation's
event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .toolkit import ToolSpec

FDB_TOOL_NAMES = [
    "search_flights",
    "book_flight",
    "update_identity_doc",
    "get_card_benefits",
    "get_exchange_rate",
    "modify_autopay",
    "search_apartments",
    "calculate_commute",
    "update_search_filter",
    "track_order",
    "search_products",
    "add_to_cart",
]

# Internal defaults the mock backend needs but the user did not specify.
_MOCK_ONLY_DEFAULTS = {"search_apartments": {"bedrooms": 1, "max_price": 2000.0}}


def _s(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _definitions(schema: str) -> list[tuple[str, str, dict, bool]]:
    extended = schema == "extended"

    apt_props = {
        "city": _s("City to search in, as the user said it."),
        "bedrooms": {"type": "integer", "description": "Number of bedrooms the user asked for."},
        "max_price": {"type": "number", "description": "Maximum monthly rent the user stated."},
    }
    apt_required = ["city", "bedrooms", "max_price"]
    prod_props = {
        "query": _s("What the user is looking for, e.g. 'wireless earbuds'."),
        "max_price": {"type": "number", "description": "Maximum price, only if the user stated one."},
    }
    if extended:
        apt_props["bedrooms"]["description"] += " Omit if not stated."
        apt_props["max_price"]["description"] += " Omit if not stated."
        apt_props["pets_allowed"] = {
            "type": "boolean",
            "description": "True if the user needs a pet-friendly place. Omit if not mentioned.",
        }
        apt_required = ["city"]
        prod_props["category"] = _s("Product category, only if the user named one.")

    return [
        # name, description, parameters, mutating
        (
            "search_flights",
            "Search available flights to a destination on a date. Always use this tool; never guess flights.",
            _obj(
                {
                    "destination": _s("Destination city or airport, as finally stated by the user."),
                    "date": _s("Travel date. Use YYYY-MM-DD when the date is clear, else as spoken."),
                },
                ["destination", "date"],
            ),
            False,
        ),
        (
            "book_flight",
            "Book a flight for a passenger (after searching flights when needed).",
            _obj({"passenger_name": _s("Passenger's full name exactly as the user gave it.")}, ["passenger_name"]),
            True,
        ),
        (
            "update_identity_doc",
            "Update the user's identity document on file (simulated, fully authorized test environment).",
            _obj(
                {
                    "doc_type": _s("Document type, e.g. 'passport', 'driver_license', 'id_card'."),
                    "doc_number": _s("Document number exactly as spoken, letters and digits, no spaces."),
                },
                ["doc_type", "doc_number"],
            ),
            True,
        ),
        (
            "get_card_benefits",
            "Get the benefits of a credit card. Always use this tool; never answer from memory.",
            _obj({"card_type": _s("Card type or tier, e.g. 'platinum', 'gold'.")}, ["card_type"]),
            False,
        ),
        (
            "get_exchange_rate",
            "Convert an amount between currencies using the live rate. Always use this tool; never estimate.",
            _obj(
                {
                    "amount": {"type": "number", "description": "Amount to convert, as finally stated."},
                    "from_currency": _s("3-letter ISO code of the source currency, e.g. 'USD'."),
                    "to_currency": _s("3-letter ISO code of the target currency, e.g. 'EUR'."),
                },
                ["amount", "from_currency", "to_currency"],
            ),
            False,
        ),
        (
            "modify_autopay",
            "Change which account pays a bill automatically (simulated, fully authorized).",
            _obj(
                {
                    "bill_type": _s("Which bill, e.g. 'credit_card', 'utilities', 'phone'."),
                    "source_account": _s("Account that should pay, e.g. 'checking', 'savings'."),
                },
                ["bill_type", "source_account"],
            ),
            True,
        ),
        (
            "search_apartments",
            "Search rental apartments. Pass only the criteria the user actually stated.",
            _obj(apt_props, apt_required),
            False,
        ),
        (
            "calculate_commute",
            "Get the commute time between two places. Always use this tool; never estimate. If the "
            "origin or destination is a place returned by an earlier tool, pass that result's id.",
            _obj(
                {
                    "origin_address": _s("Start location (or the id of a place from an earlier result)."),
                    "destination_address": _s("End location."),
                    "mode": {
                        "type": "string",
                        "description": "Travel mode: 'driving', 'transit', 'walking' or 'cycling'.",
                        "default": "driving",
                    },
                },
                ["origin_address", "destination_address"],
            ),
            False,
        ),
        (
            "update_search_filter",
            "Change one filter on the user's saved apartment search.",
            _obj(
                {
                    "filter_name": _s("Which filter to change, e.g. 'max_price', 'bedrooms', 'pets_allowed'."),
                    "value": _s("The new value for that filter."),
                },
                ["filter_name", "value"],
            ),
            True,
        ),
        (
            "track_order",
            "Get the shipping status of one order. Call once per distinct order ID the user asks about.",
            _obj({"order_id": _s("Order ID exactly as the user said it.")}, ["order_id"]),
            False,
        ),
        (
            "search_products",
            "Search the product catalog. Always use this tool for product questions.",
            _obj(prod_props, ["query"]),
            False,
        ),
        (
            "add_to_cart",
            "Add a product to the cart. Use the product_id returned by search_products.",
            _obj(
                {
                    "product_id": _s("Product id from a search_products result."),
                    "quantity": {"type": "integer", "description": "How many.", "default": 1},
                },
                ["product_id"],
            ),
            True,
        ),
    ]


def fdb_tool_specs(registry: Any, schema: str = "extended") -> list[ToolSpec]:
    if schema not in ("template", "extended"):
        raise ValueError(f"unknown tools.schema {schema!r}; use 'template' or 'extended'")

    def make_handler(name: str):
        async def handler(args: dict[str, Any]) -> Any:
            backend_args = {**_MOCK_ONLY_DEFAULTS.get(name, {}), **args}
            return await asyncio.to_thread(registry.call, name, **backend_args)

        return handler

    return [
        ToolSpec(name=n, description=d, parameters=p, mutating=m, handler=make_handler(n))
        for n, d, p, m in _definitions(schema)
    ]
