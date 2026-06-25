import "dotenv/config";
import express from "express";
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { getAvatarCues } from "./avatarManager.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || "0.0.0.0";
const REALTIME_MODEL = process.env.REALTIME_MODEL || "gpt-realtime-2";
const REALTIME_VOICE = process.env.REALTIME_VOICE || "marin";

const PERSONA =
  "You are a warm, concise voice avatar. Speak naturally and briefly, like a " +
  "real conversation. Keep answers to a couple of sentences unless asked for more.";

const app = express();
app.use(express.json({ limit: "256kb" }));
app.use(express.static(path.join(__dirname, "public")));

// --- Mint a short-lived ephemeral token for the browser's WebRTC session ---
// The real OPENAI_API_KEY never leaves the server; the browser gets only this.
app.post("/api/realtime/session", async (_req, res) => {
  try {
    const r = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session: {
          type: "realtime",
          model: REALTIME_MODEL,
          instructions: PERSONA,
          audio: { output: { voice: REALTIME_VOICE } },
        },
      }),
    });
    const data = await r.json();
    if (!r.ok) {
      console.error("[realtime/session]", data);
      return res.status(r.status).json({ error: data });
    }
    // Response shape has varied; support both flat and nested forms.
    const value = data.value || data.client_secret?.value;
    res.json({ value, model: REALTIME_MODEL, voice: REALTIME_VOICE });
  } catch (err) {
    console.error("[realtime/session]", err);
    res.status(500).json({ error: err.message });
  }
});

// --- Avatar manager: conversation snippet -> {mood, gesture, gaze} ----------
app.post("/api/avatar/cues", async (req, res) => {
  const context = String(req.body?.context || "").trim();
  if (!context) return res.status(400).json({ error: "No context provided." });
  res.json(await getAvatarCues(context));
});

// --- Serve over HTTPS when certs are present (required for mic access on -----
// remote devices: getUserMedia needs a secure context, i.e. HTTPS or localhost).
const KEY_PATH = process.env.TLS_KEY || path.join(__dirname, "certs", "key.pem");
const CERT_PATH = process.env.TLS_CERT || path.join(__dirname, "certs", "cert.pem");
const hasTLS = fs.existsSync(KEY_PATH) && fs.existsSync(CERT_PATH);

const server = hasTLS
  ? https.createServer(
      { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) },
      app
    )
  : http.createServer(app);

server.listen(PORT, HOST, () => {
  const scheme = hasTLS ? "https" : "http";
  console.log(`\n  Realtime avatar → ${scheme}://localhost:${PORT}  (bound ${HOST}:${PORT})`);
  if (!hasTLS) {
    console.log("  ⚠  HTTP only — microphone won't work on remote devices (no HTTPS).");
  }
  console.log(`  Voice model: ${REALTIME_MODEL} (${REALTIME_VOICE})`);
  console.log(
    `  Avatar manager: ${process.env.AVATAR_MANAGER_BACKEND || "openai"} / ${
      process.env.AVATAR_MANAGER_MODEL || "gpt-4o-mini"
    }\n`
  );
});
