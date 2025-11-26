# GitHub Setup Instructions

To push this repository to GitHub:

## Option 1: Create a new repository on GitHub first

1. Go to https://github.com and create a new repository (e.g., `cmsc740-review-materials`)
2. **Do NOT** initialize with README, .gitignore, or license
3. Then run:

```bash
cd /fs/nexus-projects/Generative_Detection/workspace/courses/Fall_2025/cmsc740/review_material
git remote add origin https://github.com/YOUR_USERNAME/cmsc740-review-materials.git
git branch -M main
git push -u origin main
```

## Option 2: Use GitHub CLI (if installed)

```bash
cd /fs/nexus-projects/Generative_Detection/workspace/courses/Fall_2025/cmsc740/review_material
gh repo create cmsc740-review-materials --public --source=. --remote=origin --push
```

## Option 3: Manual push after creating repo

After creating the repository on GitHub:

```bash
cd /fs/nexus-projects/Generative_Detection/workspace/courses/Fall_2025/cmsc740/review_material
git remote add origin https://github.com/YOUR_USERNAME/cmsc740-review-materials.git
git push -u origin master
```

**Note:** Replace `YOUR_USERNAME` with your actual GitHub username.

