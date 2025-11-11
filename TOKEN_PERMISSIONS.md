# GitHub Personal Access Token Permissions

## Required Permissions for Pushing Code

When creating your Personal Access Token, select these scopes:

### **Required: `repo` scope**
- **Full name:** "repo"
- **Description:** "Full control of private repositories"
- **What it includes:**
  - Read and write access to code
  - Read and write access to commit statuses
  - Read and write access to deployments
  - Read and write access to issues and pull requests
  - Read and write access to metadata
  - Read and write access to repository projects
  - Read and write access to repository hooks
  - Read and write access to repository variables
  - Read and write access to secrets

### **Minimum Required:**
- ✅ **`repo`** - This is the only scope you need for pushing code

### **Optional (not needed for basic push):**
- `workflow` - If you want to use GitHub Actions
- `write:packages` - If you want to publish packages
- `delete_repo` - If you want to delete repositories

## Step-by-Step Token Creation

1. Go to: https://github.com/settings/tokens
2. Click: **"Generate new token"** → **"Generate new token (classic)"**
3. **Name:** `deeplip-push` (or any name you prefer)
4. **Expiration:** Choose your preference (30 days, 90 days, or no expiration)
5. **Select scopes:** Check the box for **`repo`**
   - This will automatically select all sub-permissions under `repo`
6. Click: **"Generate token"**
7. **Copy the token immediately** - you won't see it again!

## Security Note

- The `repo` scope gives full access to your repositories
- For public repositories, you can use a more limited scope, but `repo` works for both public and private
- Never share your token or commit it to git
- If you lose the token, you can always generate a new one and revoke the old one

