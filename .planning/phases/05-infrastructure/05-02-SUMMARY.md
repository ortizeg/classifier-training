---
phase: 05-infrastructure
plan: 02
subsystem: infra
tags: [github-actions, ci-cd, pixi, semantic-release, codecov, gitflow]

# Dependency graph
requires:
  - phase: 05-infrastructure
    provides: "pyproject.toml with semantic_release config, pixi.toml with format-check/lint/typecheck/test-cov tasks"
provides:
  - "lint.yml: format-check, lint, typecheck on all pushes"
  - "test.yml: pytest with coverage on main/develop branches"
  - "release.yml: semantic release with PyPI publish on main/develop"
  - "Bug report and feature request issue templates"
  - "PR template with pixi run test/lint/format checklist"
  - "CODEOWNERS assigning @ortizeg to all paths"
  - "develop branch created locally (no remote configured)"
affects: [future PRs, release automation, branch protection]

# Tech tracking
tech-stack:
  added:
    - "prefix-dev/setup-pixi@v0.8.1 (GitHub Actions pixi integration)"
    - "codecov/codecov-action@v4 (coverage upload)"
    - "python-semantic-release/python-semantic-release@v10.5.3 (semantic release)"
    - "pypa/gh-action-pypi-publish@release/v1 (PyPI trusted publishing)"
    - "python-semantic-release/publish-action@v10.5.3 (GitHub Releases)"
  patterns:
    - "CONDA_OVERRIDE_CUDA=12.1 env var on all CI jobs for GPU-optional builds"
    - "pixi cache=true for fast workflow runs"
    - "release workflow uses check job as prerequisite before semantic release"
    - "fetch-depth: 0 + git reset --hard SHA for semantic release commit analysis"
    - "gitflow branching: main (production) + develop (prerelease)"

key-files:
  created:
    - ".github/workflows/lint.yml"
    - ".github/workflows/test.yml"
    - ".github/workflows/release.yml"
    - ".github/ISSUE_TEMPLATE/bug_report.yml"
    - ".github/ISSUE_TEMPLATE/feature_request.yml"
    - ".github/PULL_REQUEST_TEMPLATE.md"
    - ".github/CODEOWNERS"
  modified: []

key-decisions:
  - "Workflows ported identically from object-detection-training sibling repo — same action versions, same CUDA override, same pixi integration"
  - "CODEOWNERS references classifier_training (not object_detection_training) for conf/ path"
  - "develop branch created locally only — no remote configured, push deferred until GitHub repo is set up"
  - "Task 4 (branch protection) requires human action via GitHub Settings UI — not automatable"

patterns-established:
  - "CI pattern: all three workflows share CONDA_OVERRIDE_CUDA=12.1 + setup-pixi@v0.8.1 + cache:true"
  - "Release pattern: check job gates release job, semantic release on main+develop"

# Metrics
duration: 2min
completed: 2026-02-19
---

# Phase 5 Plan 02: GitHub CI/CD and Repository Templates Summary

**3 GitHub Actions workflows (lint/test/release), issue/PR templates, CODEOWNERS, and develop branch — ported from object-detection-training sibling repo with classifier_training path substitution**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-19T06:32:26Z
- **Completed:** 2026-02-19T06:34:03Z
- **Tasks:** 3 automated + 1 human-action documented
- **Files modified:** 7 created

## Accomplishments

- Three CI/CD workflows: lint on all pushes, test with coverage on main/develop, semantic release on main/develop
- Repository templates: bug report (structured YAML form), feature request, PR checklist with pixi commands
- CODEOWNERS with `classifier_training` conf path (correctly substituted from sibling repo's `object_detection_training`)
- develop branch created from main (local only — no remote configured yet)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create GitHub Actions CI/CD workflows** - `8e75f0e` (feat)
2. **Task 2: Create repository templates and CODEOWNERS** - `0762ccb` (feat)
3. **Task 3: Create develop branch** - no file commit (git branch operation only, develop points to `0762ccb`)
4. **Task 4: Configure branch protection** - REQUIRES HUMAN ACTION (see below)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `.github/workflows/lint.yml` - Lint & Format workflow: format-check, lint, typecheck on all pushes
- `.github/workflows/test.yml` - Test Suite workflow: pytest with coverage on main/develop, codecov upload
- `.github/workflows/release.yml` - Release workflow: check job (lint+typecheck+test) + semantic release job with PyPI publish
- `.github/ISSUE_TEMPLATE/bug_report.yml` - Bug report template (description, reproduction steps, expected behavior, version, OS dropdown)
- `.github/ISSUE_TEMPLATE/feature_request.yml` - Feature request template (problem statement, proposed solution, alternatives)
- `.github/PULL_REQUEST_TEMPLATE.md` - PR template with Summary/Changes/Test Plan/Checklist sections
- `.github/CODEOWNERS` - @ortizeg owns all paths, with `/src/classifier_training/conf/` specific entry

## Decisions Made

- Workflows ported identically from object-detection-training sibling repo — action versions are frozen (setup-pixi@v0.8.1, python-semantic-release@v10.5.3, codecov-action@v4)
- CODEOWNERS uses `classifier_training` (not `object_detection_training`) for the conf/ path
- develop branch created locally only — no remote configured in this repo yet; push to origin when GitHub remote is added

## User Setup Required

### Task 4: Configure Branch Protection (Requires Human Action)

Branch protection must be configured via GitHub Settings UI after the remote is set up.

**For `main` branch:**
1. Go to GitHub repo Settings > Branches > Add branch protection rule
2. Branch name pattern: `main`
3. Enable "Require a pull request before merging"
4. Enable "Require status checks to pass before merging"
5. Add required status checks: `lint`, `test` (from the CI workflows)
6. Enable "Require branches to be up to date before merging"
7. Under "Allow specified actors to bypass required pull requests", add `github-actions[bot]` — this allows semantic release to push version bump commits and tags directly to main without a PR

**For `develop` branch:**
1. Branch name pattern: `develop`
2. Enable "Require status checks to pass before merging"
3. Add required status checks: `lint`, `test`
4. Under "Allow specified actors to bypass required pull requests", add `github-actions[bot]`

**Verification:** After configuring, try pushing directly to main — it should be rejected. PRs should show required status checks.

## Deviations from Plan

None - plan executed exactly as written. Task 4 was a checkpoint:human-action as planned; documented above without blocking.

## Issues Encountered

- No remote configured for this repo: `git remote -v` returned empty. Task 3 (develop branch) was completed locally only. The `git push -u origin develop` step was skipped since no remote exists. Note in STATE.md: push develop to origin when GitHub remote is configured.

## Next Phase Readiness

- All 7 CI/CD files are in place and committed to main
- develop branch exists locally pointing to same commit as main
- Once a GitHub remote is added: `git remote add origin <url> && git push -u origin main && git push -u origin develop`
- Branch protection (Task 4) can be configured after GitHub remote setup
- Phase 5 is now complete pending Task 4 (branch protection)

---
*Phase: 05-infrastructure*
*Completed: 2026-02-19*

## Self-Check: PASSED

All 8 files found, both commits verified (8e75f0e, 0762ccb), all content assertions pass.
