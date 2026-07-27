# Troubleshooting

## GitHub Desktop shows no changes after Save

Most likely the wrong repository is open or the file was saved outside the repo folder.

Check:

- Current Repository in GitHub Desktop;
- actual file path in Explorer or editor;
- whether the editor tab was saved.

## There is no Push origin button

Either there is no new local commit or the repository has not been published yet.

Check History first. If the repository is local-only, GitHub Desktop may show `Publish repository`.

## GitHub.com did not change after commit

This is expected if you committed locally but did not push.

Press `Push origin`, then refresh GitHub.com.

## GitHub Desktop asks to Fetch before Push

The remote has commits that are not local yet.

For a solo classroom demo this should rarely happen. If it does, fetch/sync first and avoid explaining merge theory during C2.

## AGENTS.md appears in Changes

Check that `.gitignore` contains the exact line:

```text
AGENTS.md
```

For live demo, the cleanest path is to use a fresh demo repo where `AGENTS.md` was never tracked.

## Codex sees the wrong files

The wrong project root is open.

Close the project and open the root folder of the Git repository.

## Agent changed too much

Do not commit.

Ask the agent to revert unrelated changes or restore the demo repo, then repeat the prompt with explicit scope:

```text
Do not change unrelated files.
Only change PROJECT_STATUS.md.
Pause before commit for classroom review.
```

## Agent committed too early

Use GitHub Desktop History to show the commit diff.

For the next block, add:

```text
Pause before commit for classroom review.
```
