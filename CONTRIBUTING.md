# Contributing to Core Sentinel

## Workflow

1. **Clone** the repository (do not rely on ZIP-only copies for ongoing work).  
2. Create a **branch** for your change (`feature/…`, `fix/…`).  
3. Keep commits **focused** with clear messages.  
4. Before opening a PR or merging, follow **[docs/PRODUCTION_AND_RELEASE.md](docs/PRODUCTION_AND_RELEASE.md)** — at minimum run `adversarial_tests.py` from `core-sentinel-guardrail` if you touched inference or guardrail logic.  
5. **Never commit** `.env`, API keys, or large artifacts listed in `.gitignore`.

## Documentation

- User-facing setup: **[README.md](README.md)** and **[docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)**.  
- Repository map: **[REPO_LAYOUT.md](REPO_LAYOUT.md)**.  
- Operations and releases: **[docs/PRODUCTION_AND_RELEASE.md](docs/PRODUCTION_AND_RELEASE.md)**.  
- Backup and recovery: **[docs/BACKUP_AND_RECOVERY.md](docs/BACKUP_AND_RECOVERY.md)**.

## Code style

Match existing patterns in the files you edit; prefer small diffs over wide refactors unless agreed.
