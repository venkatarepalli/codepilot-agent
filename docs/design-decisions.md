# Design Decisions

This document is the index for significant CodePilot engineering decisions.

## Initial decisions

### DD-001 — Documentation-first development

**Decision:** Maintain README and supporting documentation throughout implementation rather than creating documentation after the code is complete.

**Reason:** Minimize rework and ensure setup, operational behavior, testing, troubleshooting, and design rationale remain accurate.

### DD-002 — Tests alongside features

**Decision:** Add tests as each major feature is implemented.

**Reason:** Prevent late-stage test retrofitting and make requirement verification continuous.

### DD-003 — Explicit tool-substitution log

**Decision:** Record every assignment-specified tool substitution and its rationale.

**Reason:** Preserve assignment traceability and engineering rationale.

### DD-004 — Explicit engineering issue log

**Decision:** Record significant implementation problems, root causes, resolutions, and verification.

**Reason:** Preserve the engineering history and reduce repeated investigation.

### DD-005 — Classification/Skills ambiguity

**Decision status:** Open.

The assignment defines five task classification categories but explicitly specifies four Skills. This will be resolved before the Skills implementation and recorded with the final rationale.
