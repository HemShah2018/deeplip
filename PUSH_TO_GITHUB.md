# Push to GitHub - Authentication Guide

## Option 1: Personal Access Token (Quickest)

1. **Create a GitHub Personal Access Token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Name it: "deeplip-push"
   - Select scope: `repo` (full control of private repositories)
   - Click "Generate token"
   - **COPY THE TOKEN** (you won't see it again!)

2. **Push using the token:**
   ```bash
   git push -u origin main
   ```
   When prompted:
   - Username: `zaydabash`
   - Password: **paste your token** (not your GitHub password)

## Option 2: SSH Key (More Secure, One-Time Setup)

1. **Generate SSH key:**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default location
   # Optionally set a passphrase
   ```

2. **Add SSH key to GitHub:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   # Copy the output
   ```
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste the key and save

3. **Change remote to SSH:**
   ```bash
   git remote set-url origin git@github.com:zaydabash/deeplip.git
   git push -u origin main
   ```

## Current Status

- Git repository: Initialized
- Files committed: 33 files, 3210+ lines
- Remote configured: https://github.com/zaydabash/deeplip.git
- Ready to push: Yes (just need authentication)

## What's Included

The repository includes:
- All source code (`src/`)
- Documentation (README.md, guides)
- Configuration files
- Helper scripts

**Excluded** (via .gitignore):
- Model weights (`models/*.h5`)
- Dataset files (`data/`)
- Training logs
- Python cache files

