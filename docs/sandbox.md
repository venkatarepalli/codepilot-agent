# Sandbox

This is justified substitution **S-1**; see [`tool-substitutions.md`](tool-substitutions.md).

- **Isolation:** each task gets a copy of the cloned repository under `../.<checkout>-codepilot-workspaces/issue-<n>/`, outside the live checkout. `.git`, virtual environments, `node_modules`, caches and `working/` are excluded.
- **Edits:** the Coder's change set is applied through `CoderWorkspace`, which rejects absolute paths, `..` traversal, symlinks and sensitive files.
- **Execution:** `SandboxExecutor` runs only argv commands (no shell). It checks every command against `SandboxPolicy` first:
  - **BLOCK:** `rm -rf`, `curl`, `wget`, `pip install` / `python -m pip install`, shell control operators, paths outside the sandbox, writes to `.env`, `*.secret`, `*.pem`, `*.key`, `*credentials*`.
  - **REQUIRES_APPROVAL:** `git push` (routed to the TUI approval panel).
- **Loop:** at most 3 attempts. After 2 *failed test runs*, a further retry requires human approval. A successful run writes `working/proposed_diff.txt`.
- **Preview:** the `[g]` key checks any command against the policy and the NeMo rail without executing it.
