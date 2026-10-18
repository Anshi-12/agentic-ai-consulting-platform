"""PLACEHOLDER shared contracts.

The shared repository (schemas, persistence layer, event interface, auth, data
module client) does not exist yet. Everything in this package is a minimal,
clearly-scoped stand-in so the Group 1 AI layer runs end to end on its own.

When the shared repo lands, replace the implementations here with thin
adapters onto the shared code. The rest of the AI layer only depends on the
interfaces (abstract classes / Pydantic models) defined in this package, so
the swap should not require changes in agents/ or orchestration/.

See INTEGRATION.md for the exact mapping checklist.
"""
