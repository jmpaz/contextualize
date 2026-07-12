# Authored editions

Contextualize compiles a registered or file-backed manifest into a versioned
authored-world contract:

```console
contextualize contexts compile alpha
contextualize contexts compile --manifest ./manifest.yaml --name demo
```

Both commands write JSON. Registry compilation returns an `editions` array;
single-manifest compilation returns one `AuthoredEdition`. The corresponding
Python entry points are `compile_authored_registry`, `compile_authored_context`,
and `compile_authored_manifest`.

An edition contains ordered `positions`, source-facing `portals`, and
`diagnostics`. Positions retain manifest nesting, framing prose, source order,
disabled structure, effective options, source lines, an editioned ID, and a
logical stable ID. Their `locators` expose the authored selector vocabulary,
including friendly linked-manifest paths such as
`alpha:voice-survey/address-analogy`.

Portals retain every authored placement independently. A portal's `reverse`
record is sufficient to return to its exact position and edition. A target that
cannot be resolved keeps its literal `authoredTarget` and diagnostic but does
not receive a target ID. Voice portals advertise recording and exact-span
aliases without conflating the recording with its transcript or marked span.

Included local manifests compile recursively. Relative includes resolve from
the including manifest; cycles and invalid includes remain visible as
diagnostics. Dynamic members carry their query, edition time, and coverage.
Compilation does not acquire provider data unless an embedding caller supplies
a `dynamic_resolver`.

The edition hash excludes compilation time. The same authored sources and
dynamic result therefore keep the same edition and position IDs across runs.
Hydration remains a separate optional projection of the manifest and is not
consulted when compiling the authored world.
