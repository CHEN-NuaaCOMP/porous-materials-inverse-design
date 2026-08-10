# GitHub upload checklist

1. Create an empty private or public repository without generated starter files.
2. Copy the contents of this directory to the repository root.
3. From the repository root, run `python tests/test_smoke.py`.
4. Confirm that `git status` does not include `outputs/`, `results/`, caches, or
   local environment files.
5. Initialize and push:

```bash
git init
git add .
git commit -m "Add reviewer reproducibility code"
git branch -M main
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

All tracked files are below GitHub's 100 MB per-file limit; Git LFS is not
required for the bundled checkpoints.

