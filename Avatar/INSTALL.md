# INSTALL — agent runbook

Instructions for a **Claude Code agent** (or any operator) to install and run
this project on a fresh **macOS, Windows, or Linux** machine. Follow top to
bottom. All commands are run **from the project folder** (the directory that
contains this `INSTALL.md`). Do not hardcode absolute paths — use the cloned/
copied folder's own location.

---

## 0. Assumptions

- You have the project folder on disk (cloned, unzipped, or copied).
- You have an **OpenAI API key** with access to a Realtime model
  (`gpt-realtime` or `gpt-realtime-2`) and `gpt-4o-mini`.
- The end user will open the app in **Google Chrome or Microsoft Edge**
  (the mic uses WebRTC + Web Speech, which are Chromium features).

---

## 1. Ensure Node.js 18+ is installed

Check first:

```bash
node --version
```

If it prints `v18` or higher, skip to step 2. Otherwise install per OS:

**macOS**
```bash
# Homebrew (preferred)
brew install node
# …or nvm if you don't have Homebrew:
#   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
#   nvm install --lts
```

**Windows** (PowerShell)
```powershell
winget install OpenJS.NodeJS.LTS
# …or download the LTS installer from https://nodejs.org and run it.
# Open a NEW terminal afterward so PATH updates.
```

**Linux** (Debian/Ubuntu)
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
# Fedora/RHEL: sudo dnf install -y nodejs
# Arch:        sudo pacman -S nodejs npm
# Any distro via nvm (no root):
#   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && nvm install --lts
```

Re-check `node --version` before continuing.

---

## 2. Install dependencies

From the project folder:

```bash
npm install
```

This installs `express` and `dotenv` only (no native builds, no AWS SDK).

---

## 3. Configure environment (`.env`)

Copy the template and add the OpenAI key. The `.env` file stays on the server
and is git-ignored; the browser never sees the real key.

**macOS / Linux**
```bash
cp .env.example .env
```

**Windows** (PowerShell)
```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set at minimum:

```
OPENAI_API_KEY=sk-...your key...
REALTIME_MODEL=gpt-realtime-2     # if your account rejects it, use: gpt-realtime
REALTIME_VOICE=marin
AVATAR_MANAGER_MODEL=gpt-4o-mini
PORT=3000                         # change if 3000 is taken
```

Set the key without opening an editor, if you prefer:

- macOS/Linux: `printf 'OPENAI_API_KEY=%s\n' "sk-..." >> .env`
- Windows PS: `Add-Content .env 'OPENAI_API_KEY=sk-...'`

---

## 4. Verify the bundled avatar exists

The 3D avatar is served locally from `public/avatars/brunette.glb` (~4.7 MB).
Confirm it's present:

```bash
# macOS/Linux
ls -l public/avatars/brunette.glb
# Windows PS
Get-Item public/avatars/brunette.glb
```

If it is **missing** (e.g. it was stripped from a zip), re-download it:

```bash
# macOS/Linux
mkdir -p public/avatars
curl -L -o public/avatars/brunette.glb \
  "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@main/avatars/brunette.glb"
```
```powershell
# Windows PS
New-Item -ItemType Directory -Force public/avatars | Out-Null
Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@main/avatars/brunette.glb" `
  -OutFile "public/avatars/brunette.glb"
```

A valid file starts with the ASCII bytes `glTF`.

---

## 5. Start the server

```bash
npm start
```

Expected output (port may differ):

```
  Realtime avatar → http://localhost:3000
  Voice model: gpt-realtime-2 (marin)
  Avatar manager: openai / gpt-4o-mini
```

Leave this process running. Stop it with `Ctrl+C`.

---

## 6. Verify it works (no browser needed)

In a second terminal, from the project folder. Replace `3000` if you changed `PORT`.

```bash
# Page serves
curl -s -o /dev/null -w "index: %{http_code}\n" http://localhost:3000/

# Avatar serves as binary glTF
curl -s -o /dev/null -w "avatar: %{http_code} %{content_type}\n" http://localhost:3000/avatars/brunette.glb

# Avatar-manager (gpt-4o-mini) returns mood/gesture JSON
curl -s -X POST http://localhost:3000/api/avatar/cues \
  -H "Content-Type: application/json" \
  -d '{"context":"The avatar just said: \"Congratulations, that is wonderful!\""}'

# Realtime ephemeral token mints (needs a valid OPENAI_API_KEY + model access)
curl -s -X POST http://localhost:3000/api/realtime/session
```

Healthy results: `index: 200`, `avatar: 200 model/gltf-binary`, a JSON object
like `{"mood":"happy",...}`, and a session JSON containing a `"value"` token.

> Windows: if `curl` is unavailable, use `Invoke-RestMethod` / `Invoke-WebRequest`,
> or just open the URL in a browser.

---

## 7. Open the app

Open **http://localhost:3000** in **Chrome or Edge**, allow the microphone when
prompted, then:

- **🎙️ Start talking** — live voice conversation via GPT-Realtime.
- **🎬 Test all** — auto-cycle every mood, gesture, pose, emoji, head move, gaze.
- **Picker + ▶ Play** — trigger any single action on demand.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `npm start` → `EADDRINUSE` | Port in use. Set a different `PORT` in `.env`, or stop the other process. |
| `index: 200` but avatar blank | Check the browser console; ensure `public/avatars/brunette.glb` exists (step 4). |
| `/api/realtime/session` has no `value` | Bad/absent `OPENAI_API_KEY`, or your account lacks the model. Try `REALTIME_MODEL=gpt-realtime`. |
| Cues always `neutral` | Server log shows `[avatarManager]` errors — usually the `gpt-4o-mini` call failing (key/quota). |
| No mic / no voice | Use Chrome or Edge, allow mic permission, and click a button first (browsers gate audio behind a user gesture). |
| `node: command not found` | Node not installed or PATH not refreshed — redo step 1 in a new terminal. |

---

## Notes for the agent

- **Never** write absolute paths into the repo. The server resolves its own
  directory via `fileURLToPath(import.meta.url)`; the client uses relative URLs
  (`/avatars/...`, `/api/...`). Keep it that way.
- Do **not** commit `.env` (it's in `.gitignore`).
- The avatar manager is intentionally swappable to **AWS Bedrock** later; see
  `README.md` → "Switching the avatar manager to Bedrock". No AWS keys are
  required to run the project as-is.
