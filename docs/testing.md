# Testing Strategy

## Principles

- Tests are implemented with the feature, not after the feature.
- Unit tests must remain fast and isolated.
- Integration tests cover component/service boundaries.
- End-to-end tests cover the complete issue-to-PR workflow.
- Tests that require external services are clearly identified.
- Security/guardrail behavior is tested explicitly.

## Test layers

```text
Unit
  -> component behavior

Integration
  -> GitHub / memory / sandbox / agent boundaries

E2E
  -> issue -> explore -> implement -> test -> approval -> PR
```

Exact commands and environment requirements will be added as the test infrastructure is implemented.
