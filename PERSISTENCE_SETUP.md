# GitHub Persistence Setup Guide

## Problem
Streamlit Cloud free tier doesn't have persistent storage. When the app goes to sleep, all data (photos, attendance records) is lost.

## Solution
The app now automatically syncs all data to your GitHub repository. Data persists across app restarts!

## Setup Instructions

### Step 1: Create a GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "Streamlit Attendance App"
4. Set expiration to "No expiration" (or your preference)
5. Check the box for **`repo`** (full control of private repositories)
6. Click "Generate token" at the bottom
7. **COPY THE TOKEN** - you won't see it again!

### Step 2: Add Token to Streamlit Cloud

1. Go to your Streamlit Cloud dashboard: https://share.streamlit.io/
2. Click on your "Attendance-System" app
3. Click the hamburger menu (⋮) → "Settings"
4. Go to "Secrets" section
5. Add this line (replace with your actual token):
   ```
   GITHUB_TOKEN = "ghp_your_token_here"
   ```
6. Click "Save"
7. App will restart automatically

### Step 3: Verify It's Working

1. Add a new person or mark attendance
2. Wait 1-2 minutes
3. Check your GitHub repo - you should see new commits from "Streamlit App"
4. Let the app sleep (wait 7 days or force sleep)
5. When it wakes up, your data will still be there!

## What Gets Synced

- ✅ `dataset/` folder (all person photos)
- ✅ `attendance_*.csv` files (all attendance records)
- ✅ `face_encodings_cache.pkl` (face recognition cache)

## How It Works

- On app startup: Pulls latest data from GitHub
- After adding a person: Commits and pushes to GitHub
- After marking attendance: Commits and pushes to GitHub
- After removing a person: Commits and pushes to GitHub

## Important Notes

⚠️ The GitHub token has full access to your repositories. Keep it secret!
⚠️ Don't share your secrets.toml file
⚠️ Syncing adds 1-2 seconds delay after each operation
✅ Works only on Streamlit Cloud (not needed locally)
✅ If token is not configured, app works normally but data won't persist

## Troubleshooting

**Data still gets lost after sleep:**
- Check that token is correctly added in Streamlit Cloud secrets
- Verify token has `repo` permission
- Check GitHub repo for new commits from "Streamlit App"

**Slow performance:**
- This is normal - each operation now syncs to GitHub
- Trade-off for data persistence

**Token expired:**
- Create a new token and update secrets on Streamlit Cloud
