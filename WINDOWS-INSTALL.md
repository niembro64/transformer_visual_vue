# Windows Installation Fix

If you're getting "'vite' is not recognized" errors, it's because the dependencies were installed from WSL but you're running from Windows CMD.

## Solution 1: Use npx (Already configured)

The package.json has been updated to use `npx`. Just run:

```cmd
npm start
```

This should work now!

## Solution 2: Reinstall from Windows CMD (if Solution 1 doesn't work)

Open Windows Command Prompt (CMD) or PowerShell:

```cmd
cd C:\Users\nieme\Code\transformer_visual_vue
rmdir /s /q node_modules
del package-lock.json
npm install
npm start
```

## Solution 3: Use the batch file

Double-click `start.bat` to launch the dev server.

## Why this happens

WSL creates symlinks that Windows CMD can't read properly. Installing from Windows CMD creates Windows-compatible shortcuts.
