# GPT-Realtime Talking Avatar

A 3D avatar ([TalkingHead](https://github.com/met4citizen/TalkingHead)) whose
**voice** is OpenAI **GPT-Realtime** (speech-to-speech over WebRTC), and whose
**body language** (mood, gestures, gaze) is driven by a separate, swappable
**avatar-manager LLM** — `gpt-4o-mini` today, AWS **Bedrock** later.

```
        ┌─────────────── browser ───────────────┐
  🎤 mic ─┤ WebRTC ⇄ GPT-Realtime (voice in/out) │
        │        │                               │
        │        ├─ audio track ─► energy lip-sync (jaw)
        │        └─ transcript ──► POST /api/avatar/cues
        └────────────────────────────────────────┘
                                   │  (server)
                          avatar-manager LLM  →  { mood, gesture, gaze }
                          gpt-4o-mini  ⟶  Bedrock (later)
                                   │
                      head.setMood / playGesture / gaze
```

## The two brains

| Role | What it does | Backend |
|------|--------------|---------|
| **Talking model** | Runs the actual voice conversation: hears you, thinks, speaks. The avatar lip-syncs to its audio. | **GPT-Realtime** (`gpt-realtime-2`) over WebRTC |
| **Avatar manager** | Watches what the avatar says and decides how it should *emote* — mood, hand gesture, gaze. | **`gpt-4o-mini`** now → **Bedrock** when you have AWS keys |

The seam between them is one HTTP endpoint (`/api/avatar/cues`) and one module
(`avatarManager.js`). Moving the manager to Bedrock changes only that file.

## Prerequisites

- **Node 18+**.
- An **OpenAI API key** with access to `gpt-realtime` (or `gpt-realtime-2`) and
  `gpt-4o-mini`.
- A **Chromium browser** (Chrome/Edge) — WebRTC + mic.

## Setup

```bash
cd path/to/Avatar      # the folder containing this README
npm install
cp .env.example .env      # add OPENAI_API_KEY (adjust REALTIME_MODEL if needed)
npm start
```

Open http://localhost:3000, click **🎙️ Start talking**, allow the mic, and
just speak. The avatar replies in GPT-Realtime's voice, lip-syncs, and emotes
based on the avatar-manager's cues (shown under the button).

## Lip-sync note

GPT-Realtime streams a *voice audio track* but no viseme timing, so the mouth is
driven from live audio **loudness** (jaw opens with volume). It's robust and
believable. If you later want phoneme-accurate visemes, run the assistant
transcript through TalkingHead's `streamStart({lipsyncType:"words"})` path — the
audio tap stays the same.

## Switching the avatar manager to Bedrock (later)

When you have AWS credentials:

1. `npm i @aws-sdk/client-bedrock-runtime`
2. In `avatarManager.js`, uncomment the `getCuesBedrock` function and its import.
3. In `.env`, set `AVATAR_MANAGER_BACKEND=bedrock`, `AWS_REGION`,
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `BEDROCK_MODEL_ID`
   (e.g. `openai.gpt-oss-20b-1:0`, with model access enabled in the Bedrock console).

Nothing else changes — the voice frontend and the browser are untouched.

## Picking your own avatar

Default is a Ready Player Me placeholder. Make a **half-body** avatar at
https://readyplayer.me, copy its `.glb` URL into `AVATAR_URL` in `public/app.js`,
and keep the `?morphTargets=ARKit,Oculus Visemes…` query string.

## Files

| File | Role |
|------|------|
| `server.js` | Mints ephemeral Realtime tokens (`/api/realtime/session`); proxies avatar cues (`/api/avatar/cues`). OpenAI key stays server-side. |
| `avatarManager.js` | The avatar-manager brain. OpenAI now, Bedrock branch ready. |
| `public/app.js` | TalkingHead, WebRTC to GPT-Realtime, energy lip-sync, applies cues. |
| `public/index.html` / `style.css` | Page + importmap (three 0.180.0, TalkingHead 1.7). |

## Troubleshooting

- *"Could not mint Realtime token"* → bad/absent `OPENAI_API_KEY`, or your account
  doesn't expose `REALTIME_MODEL`. Try `REALTIME_MODEL=gpt-realtime`.
- *No voice / no mouth movement* → allow the mic; check the browser console. Audio
  is unlocked by the click on **Start talking**.
- *Avatar won't load* → check `AVATAR_URL` in the console; try the default first.
- *Manager always "neutral"* → check the server log for `[avatarManager]` errors
  (usually the `gpt-4o-mini` call failing on the key/quota).
