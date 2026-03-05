# COMP34812 NLU Coursework

## CSF3 usage tips IMPORTANT
if below:
```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```
then run
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## Setup

1. Download the dataset zips and place them in the `downloads/` folder.
2. Extract `training_data.zip` and `trial_data.zip` to the project root so the structure matches:

**Do not push large files (datasets, model weights, etc.) to the repo.** The `downloads/`, `training_data/`, and `trial_data/` folders are gitignored — only the empty directories are tracked.

## Workflow

- **Create a branch for everything.** Do not commit directly to `main`.
- **Try not to merge to `main` unless everyone is around.** Coordinate merges so the whole team can review together.

## Code

- **Use Python scripts (`.py`) rather than notebooks for shared code.** Scripts are easier to review, merge, and version-control.
- Notebooks are fine for personal experimentation and exploration.
