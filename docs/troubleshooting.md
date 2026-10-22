# Troubleshooting & Recovery

The common problems and fixes are listed in the README's [Troubleshooting](../README.md#troubleshooting) table.

Additional recovery steps:

- **Reset episodic memory** (e.g. to retry an issue that failed recently): delete `.codepilot/memory/sessions.jsonl`.
- **Reset semantic lessons:** delete `.codepilot/memory/semantic/`.
- **Reset the Repo Map cache:** delete `.codepilot/memory/repo_map_cache/`.
- **Stale sandbox workspaces:** delete `../.<checkout-name>-codepilot-workspaces/`. They are retained after failures for inspection.
- **NeMo Guardrails warnings at startup** (`nim_url` deprecation) are harmless.

For engineering investigations, see `docs/engineering-issues/`.
