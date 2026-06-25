import { TalkingHead } from "talkinghead";

// Avatar served locally from /public/avatars (no external dependency). This is
// TalkingHead's bundled demo avatar, already rigged with ARKit + Oculus viseme
// morph targets. To use your own Ready Player Me avatar, make a half-body one at
// https://readyplayer.me and set AVATAR_URL to its .glb with the query string
// "?morphTargets=ARKit,Oculus Visemes&textureAtlas=1024&lod=1".
const AVATAR_URL = "/avatars/brunette.glb";

const el = (id) => document.getElementById(id);
const statusEl = el("status");
const subtitleEl = el("subtitle");
const cueEl = el("cue");
const connectBtn = el("connect");
const audioEl = el("oaiAudio");
const setStatus = (s) => (statusEl.textContent = s);

// --- Boot the avatar -----------------------------------------------------
const head = new TalkingHead(el("avatar"), {
  lipsyncModules: ["en"],
  cameraView: "upper",
  mixerGainSpeech: 3,
});
window.head = head; // exposed for manual testing from the browser console

let avatarReady = false;
try {
  setStatus("Loading avatar…");
  await head.showAvatar({
    url: AVATAR_URL,
    body: "F",
    avatarMood: "neutral",
    lipsyncLang: "en",
  });
  avatarReady = true;
  setStatus("Ready. Click to start a voice conversation.");
} catch (e) {
  console.error(e);
  setStatus("Failed to load avatar — check AVATAR_URL. " + e.message);
}

// --- Energy-driven lip-sync ----------------------------------------------
// GPT-Realtime sends us a voice audio track but no viseme timing, so we open
// the mouth in proportion to the live audio loudness. Believable and robust.
function setJaw(value) {
  // Prefer TalkingHead's realtime blendshape channel; fall back to setFixedValue.
  if (head.mtAvatar && head.mtAvatar.jawOpen) {
    Object.assign(head.mtAvatar.jawOpen, { realtime: value, needsUpdate: true });
  } else if (typeof head.setFixedValue === "function") {
    head.setFixedValue("jawOpen", value);
  }
}
function releaseJaw() {
  if (head.mtAvatar && head.mtAvatar.jawOpen) {
    Object.assign(head.mtAvatar.jawOpen, { realtime: null, needsUpdate: true });
  } else if (typeof head.setFixedValue === "function") {
    head.setFixedValue("jawOpen", null);
  }
}

// Lip-sync tuning. The mouth is driven from the SAME live stream the <audio>
// element plays, so it tracks loudness as produced and never *drifts*. The only
// offset is constant: the analyser "hears" samples slightly before your speaker
// emits them (output-device latency, large on Bluetooth). If the mouth looks
// like it LEADS the voice, raise LIPSYNC_DELAY_MS to hold the mouth back.
const LIPSYNC_DELAY_MS = 0; // try 120–250 on Bluetooth if the mouth runs ahead
const JAW_ATTACK = 0.6; // how fast the mouth OPENS  (higher = snappier onset)
const JAW_RELEASE = 0.25; // how fast the mouth CLOSES (lower = softer tail)

let audioCtx, analyser, lipRAF, jaw = 0, jawDelayLine = [];

function startLipSync(stream) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  audioCtx.resume?.(); // mobile browsers can start the context suspended
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  src.connect(analyser); // analysis only — the <audio> element does playback
  const buf = new Uint8Array(analyser.fftSize);
  jawDelayLine = [];

  const tick = () => {
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / buf.length); // 0..~0.5
    const target = Math.min(0.7, rms * 5.5); // scale to a sane jaw range
    // Faster attack than release: the mouth pops open on a sound, eases shut —
    // a tight onset reads as better sync than a symmetric lerp.
    jaw += (target - jaw) * (target > jaw ? JAW_ATTACK : JAW_RELEASE);

    // Optional latency compensation: hold the jaw back by N ms so it lines up
    // with the (slightly later) speaker output.
    let out = jaw;
    if (LIPSYNC_DELAY_MS > 0) {
      jawDelayLine.push(jaw);
      const frames = Math.max(1, Math.round(LIPSYNC_DELAY_MS / 16.7));
      out = jawDelayLine.length > frames ? jawDelayLine.shift() : 0;
    }
    setJaw(out < 0.02 ? 0 : out);
    lipRAF = requestAnimationFrame(tick);
  };
  tick();
}

function stopLipSync() {
  if (lipRAF) cancelAnimationFrame(lipRAF);
  lipRAF = null;
  releaseJaw();
  if (audioCtx) audioCtx.close();
  audioCtx = null;
}

// --- Apply avatar-manager cues to the avatar ------------------------------
function applyCues({ mood, intensity, emoji, gesture, gaze }) {
  const i = typeof intensity === "number" ? intensity : 0.6;
  cueEl.textContent =
    `mood:${mood}${emoji ? " " + emoji : ""}` +
    `${gesture ? " · " + gesture : ""}${gaze ? " · gaze:" + gaze : ""} (${i.toFixed(1)})`;
  if (!avatarReady) return; // avatar still loading/failed — don't touch it
  try {
    if (mood) head.setMood(mood);
    // The emoji is the moment-to-moment facial expression. Scale how long it
    // holds (and thus how strongly it reads) by the model's intensity.
    if (emoji) head.playGesture(emoji, 1.5 + i * 2.5); // ~1.5–4s
    if (gesture) head.playGesture(gesture, 2.5);
    if (gaze === "camera") head.makeEyeContact?.(2500);
    else if (gaze === "away") head.lookAhead?.(2000);
  } catch (e) {
    console.warn("applyCues", e);
  }
}

// Ask the avatar-manager LLM how to emote for a chunk the avatar is speaking.
async function requestCues(assistantText) {
  if (!assistantText.trim()) return;
  try {
    const res = await fetch("/api/avatar/cues", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context: `The avatar is saying: "${assistantText}"` }),
    });
    if (res.ok) applyCues(await res.json());
  } catch (e) {
    console.warn("requestCues", e);
  }
}

// --- Stream expressions DURING a turn, not just once at the end -----------
// The transcript arrives as a stream of deltas. We re-evaluate the avatar's
// expression at each sentence boundary (throttled) so the face shifts as the
// content shifts, instead of holding one mood for the whole reply.
const CUE_MIN_INTERVAL_MS = 2200; // don't hammer the manager LLM
const CUE_MIN_CHARS = 18; // need enough new text to be worth a cue
let cueCursor = 0; // chars of assistantBuf already sent for cueing
let lastCueTime = 0;

function resetCueStream() {
  cueCursor = 0;
}

function maybeStreamCue(force = false) {
  const pending = assistantBuf.slice(cueCursor);
  if (!pending.trim()) return;
  const now = Date.now();
  if (!force && now - lastCueTime < CUE_MIN_INTERVAL_MS) return;

  // Prefer to cue on a completed sentence; otherwise (on force) take what's left.
  const sentence = pending.match(/^[\s\S]*?[.!?…](?:\s|$)/);
  if (!force && !sentence && pending.trim().length < CUE_MIN_CHARS) return;

  const chunk = (sentence ? sentence[0] : pending).trim();
  if (!chunk) return;
  cueCursor += sentence ? sentence[0].length : pending.length;
  lastCueTime = now;
  requestCues(chunk);
}

// --- GPT-Realtime over WebRTC --------------------------------------------
let pc = null, live = false, assistantBuf = "";

async function connect() {
  setStatus("Connecting to GPT-Realtime…");

  // 1) Get an ephemeral token from our server (never the real API key).
  const sess = await (await fetch("/api/realtime/session", { method: "POST" })).json();
  if (!sess.value) {
    setStatus("Could not mint Realtime token — check OPENAI_API_KEY / model id.");
    return;
  }

  // 2) WebRTC: send our mic, receive the model's voice.
  pc = new RTCPeerConnection();
  const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
  pc.addTrack(mic.getTracks()[0]);

  pc.ontrack = (e) => {
    audioEl.srcObject = e.streams[0]; // play the model's voice
    startLipSync(e.streams[0]); // and animate the mouth from it
  };

  // 3) Data channel carries text events (transcripts, etc.).
  const dc = pc.createDataChannel("oai-events");
  dc.addEventListener("message", (e) => onServerEvent(JSON.parse(e.data)));

  // 4) SDP offer/answer handshake with OpenAI, authed by the ephemeral token.
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const answer = await fetch(
    `https://api.openai.com/v1/realtime/calls?model=${encodeURIComponent(sess.model)}`,
    {
      method: "POST",
      body: offer.sdp,
      headers: {
        Authorization: `Bearer ${sess.value}`,
        "Content-Type": "application/sdp",
      },
    }
  );
  await pc.setRemoteDescription({ type: "answer", sdp: await answer.text() });

  live = true;
  connectBtn.classList.add("live");
  connectBtn.textContent = "■ End conversation";
  setStatus("Live — just talk. The avatar listens and replies.");
}

function onServerEvent(evt) {
  switch (evt.type) {
    case "response.audio_transcript.delta":
    case "response.output_audio_transcript.delta": // GA alias
      assistantBuf += evt.delta || "";
      subtitleEl.textContent = assistantBuf;
      maybeStreamCue(); // re-emote as each sentence lands
      break;
    case "response.audio_transcript.done":
    case "response.output_audio_transcript.done": {
      maybeStreamCue(true); // emote on whatever sentence remains
      assistantBuf = "";
      resetCueStream();
      break;
    }
    case "response.created":
      assistantBuf = "";
      resetCueStream();
      break;
    case "error":
      console.error("Realtime error:", evt.error || evt);
      setStatus("Realtime error: " + (evt.error?.message || "see console"));
      break;
  }
}

function disconnect() {
  live = false;
  stopLipSync();
  if (pc) pc.close();
  pc = null;
  audioEl.srcObject = null;
  subtitleEl.textContent = "";
  connectBtn.classList.remove("live");
  connectBtn.textContent = "🎙️ Start talking";
  setStatus("Ended. Click to start again.");
  if (avatarReady) head.setMood("neutral");
}

connectBtn.addEventListener("click", () => {
  if (live) disconnect();
  else connect().catch((e) => { console.error(e); setStatus("Connect failed: " + e.message); });
});

// ==========================================================================
// FULL ACTION TEST RIG — every mood, gesture (+mirror), pose, emoji, head
// move, gaze, and custom semantic hand gestures. Drives the avatar two ways:
//   • "🎬 Test all" button → auto-cycles through everything with a delay
//   • picker + "▶ Play"    → trigger any single action on demand
// ==========================================================================
const testBtn = el("test");
const picker = el("picker");
const playBtn = el("play");
const mirrorBox = el("mirror");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- Custom semantic hand gestures (authored from the built-in templates) --
// Registered onto the live instance so playGesture("measure") etc. just works.
// Bone values are Euler rotations; arrays are [from,to,...] keyframes.
function registerCustomGestures() {
  const G = head.gestureTemplates;
  // "this big" — both arms spread wide to show size.
  G.measure = {
    "LeftShoulder.rotation": { x: 1.6, y: -0.2, z: -1.45 },
    "LeftArm.rotation": { x: 1.2, y: -0.9, z: 0.9 },
    "LeftForeArm.rotation": { x: 0, y: 0, z: 0.7 },
    "LeftHand.rotation": { x: -0.3, y: -1.2, z: -0.1 },
    "RightShoulder.rotation": { x: 1.6, y: 0.2, z: 1.45 },
    "RightArm.rotation": { x: 1.2, y: 0.9, z: -0.9 },
    "RightForeArm.rotation": { x: 0, y: 0, z: -0.7 },
    "RightHand.rotation": { x: -0.3, y: 1.2, z: 0.1 },
  };
  // "welcome / open arms" — both arms raised and open.
  G.welcome = {
    "LeftShoulder.rotation": { x: 1.7, y: 0.3, z: -1.4 },
    "LeftArm.rotation": { x: 1.55, y: -0.5, z: 1.1 },
    "LeftForeArm.rotation": { x: -0.6, y: 0, z: 1.4 },
    "LeftHand.rotation": { x: -0.5, y: -0.2, z: 0.02 },
    "RightShoulder.rotation": { x: 1.7, y: -0.3, z: 1.4 },
    "RightArm.rotation": { x: 1.55, y: 0.5, z: -1.1 },
    "RightForeArm.rotation": { x: -0.6, y: 0, z: -1.4 },
    "RightHand.rotation": { x: -0.5, y: 0.2, z: -0.02 },
  };
  // "stop / halt" — one palm pushed forward toward the viewer.
  G.stop = {
    "LeftShoulder.rotation": { x: 1.6, y: 0.3, z: -1.3 },
    "LeftArm.rotation": { x: 1.4, y: -0.4, z: 0.6 },
    "LeftForeArm.rotation": { x: -0.2, y: 0, z: 1.5 },
    "LeftHand.rotation": { x: -1.4, y: -0.2, z: 0 },
  };
  // "come here" — hand up, beckoning toward self.
  G.comehere = {
    "LeftShoulder.rotation": { x: 1.6, y: 0.3, z: -1.35 },
    "LeftArm.rotation": { x: 1.5, y: -0.45, z: 0.95 },
    "LeftForeArm.rotation": { x: -1.0, y: 0, z: 1.5 },
    "LeftHand.rotation": { x: 0.4, y: -0.2, z: 0 },
  };
}

// --- Registry of EVERY available action ----------------------------------
const MOODS = ["neutral", "happy", "angry", "sad", "fear", "disgust", "love", "sleep"];
const GESTURES = ["handup", "index", "ok", "thumbup", "thumbdown", "side", "shrug", "namaste"];
const CUSTOM = ["measure", "welcome", "stop", "comehere", "directions-left", "directions-right"];
const POSES = ["straight", "wide", "side", "hip", "bend", "back", "turn", "kneel", "oneknee", "sitting"];
const HEAD = ["yes", "no"];
const GAZES = ["camera", "away"];
const EMOJIS = [
  "😐","😶","😏","🙂","🙃","😊","😇","😀","😃","😄","😁","😆","😝","😋","😛","😜",
  "🤪","😂","🤣","😅","😉","😭","🥺","😞","😔","😳","☹️","😚","😘","🥰","😍","🤩",
  "😡","😠","🤬","😒","😱","😬","🙄","🤔","👀","😴","✋","🤚","👋","👍","👎","👌","🙏",
];

// A realistic, spoken-style sentence for each action — shown as a subtitle so
// each test step reads like a real moment, not just a label. (These are the
// same lines verified to trigger the matching avatar-manager mood/gesture.)
const SENTENCES = {
  "mood:neutral": "The meeting is scheduled for three o'clock on Tuesday.",
  "mood:happy": "Yes! That is wonderful — I am so happy for you!",
  "mood:angry": "That is outrageous and completely unacceptable!",
  "mood:sad": "I am so sorry… that is heartbreaking news.",
  "mood:fear": "Look out — something is wrong, I am scared!",
  "mood:disgust": "Ugh, that is absolutely revolting and gross.",
  "mood:love": "I adore you — you mean the world to me.",
  "mood:sleep": "I am completely exhausted… I need to sleep now.",
  "gesture:handup": "Hi there! Welcome, so nice to meet you!",
  "gesture:index": "Pay attention to this one important point.",
  "gesture:ok": "Okay, perfect — that all sounds good to me.",
  "gesture:thumbup": "Great job — that is excellent work, well done!",
  "gesture:thumbdown": "That movie was awful — a thumbs down from me.",
  "gesture:side": "Let's set that aside and look over here.",
  "gesture:shrug": "Honestly, I have no idea — who knows, really.",
  "gesture:namaste": "Thank you, I'm truly grateful. Namaste.",
  "custom:measure": "It was about this big — roughly a meter wide.",
  "custom:welcome": "Welcome, everyone! Come in, make yourselves at home.",
  "custom:stop": "Stop right there — please don't go any further.",
  "custom:comehere": "Come here, come closer — I want to show you something.",
  "custom:directions-left": "Go that way — it's just down the street on your left.",
  "custom:directions-right": "Head over there — turn right at the corner.",
  "pose:straight": "Standing straight, ready to begin.",
  "pose:wide": "A confident, wide stance.",
  "pose:side": "Leaning a little to one side.",
  "pose:hip": "Hand on the hip, casual and relaxed.",
  "pose:bend": "Bending forward to take a closer look.",
  "pose:back": "Leaning back, taking it all in.",
  "pose:turn": "Turning to look the other way.",
  "pose:kneel": "Kneeling down for a moment.",
  "pose:oneknee": "Down on one knee.",
  "pose:sitting": "Taking a seat, getting comfortable.",
  "head:yes": "Yes, absolutely — I completely agree.",
  "head:no": "No, I don't think that's right.",
  "gaze:camera": "Looking right at you — let's talk.",
  "gaze:away": "Hmm, let me think about that for a second…",
};
const sentenceFor = (kind, name) => SENTENCES[`${kind}:${name}`] || `Reacting ${name}`;

// One dispatcher used by BOTH the auto-test and the picker.
function playAction(kind, name) {
  subtitleEl.textContent = sentenceFor(kind, name); // show the real sentence
  switch (kind) {
    case "mood":
      head.setMood(name);
      break;
    case "gesture":
      head.setMood("neutral");
      head.playGesture(name, 4, mirrorBox.checked);
      break;
    case "custom":
      head.setMood("neutral");
      if (name === "directions-left") head.playGesture("side", 4, false);
      else if (name === "directions-right") head.playGesture("side", 4, true);
      else head.playGesture(name, 4, mirrorBox.checked);
      break;
    case "pose":
      head.playPose(name, null, 6);
      break;
    case "emoji":
      head.playGesture(name, 3);
      break;
    case "head":
      head.playGesture(name, 2); // 'yes' nod / 'no' shake
      break;
    case "gaze":
      if (name === "camera") head.makeEyeContact?.(4000);
      else head.lookAhead?.(4000);
      break;
  }
}

// Build the flat list the auto-test walks through (every option, in order).
function buildSteps() {
  const steps = [];
  const add = (kind, names, dwell) =>
    names.forEach((n) => steps.push({ kind, name: n, label: `${kind} · ${n}`, dwell }));
  add("mood", MOODS, 4000);
  add("gesture", GESTURES, 4500);
  add("custom", CUSTOM, 4500);
  add("pose", POSES, 4500);
  add("head", HEAD, 2500);
  add("gaze", GAZES, 3500);
  add("emoji", EMOJIS, 2600);
  return steps;
}

// Populate the on-demand picker with grouped options.
function populatePicker() {
  const group = (label, kind, names) => {
    const og = document.createElement("optgroup");
    og.label = label;
    names.forEach((n) => {
      const o = document.createElement("option");
      o.value = `${kind}:${n}`;
      o.textContent = n;
      og.appendChild(o);
    });
    picker.appendChild(og);
  };
  group("Moods", "mood", MOODS);
  group("Gestures", "gesture", GESTURES);
  group("Custom hands", "custom", CUSTOM);
  group("Poses", "pose", POSES);
  group("Head", "head", HEAD);
  group("Gaze", "gaze", GAZES);
  group("Emojis", "emoji", EMOJIS);
}

let testRunning = false;
async function runTest() {
  if (!avatarReady) { setStatus("Avatar not loaded yet — wait a moment."); return; }
  if (testRunning) { testRunning = false; return; } // second click = stop

  const steps = buildSteps();
  testRunning = true;
  testBtn.classList.add("live");
  testBtn.textContent = "■ Stop test";

  for (let i = 0; i < steps.length && testRunning; i++) {
    const s = steps[i];
    cueEl.textContent = `TEST (${i + 1}/${steps.length}) → ${s.label}`;
    try { playAction(s.kind, s.name); } catch (e) { console.warn("test step", s.label, e); }
    await sleep(s.dwell);
    if (s.kind === "pose") { try { head.playPose("straight", null, 4); } catch {} }
  }

  testRunning = false;
  testBtn.classList.remove("live");
  testBtn.textContent = "🎬 Test all";
  cueEl.textContent = "";
  subtitleEl.textContent = "";
  try { head.playPose("straight", null, 3); head.setMood("neutral"); } catch {}
}

// Wire up. gestureTemplates exist from construction, so register unconditionally.
registerCustomGestures();
populatePicker();
testBtn.addEventListener("click", runTest);
playBtn.addEventListener("click", () => {
  if (!avatarReady) { setStatus("Avatar not loaded yet."); return; }
  const [kind, name] = picker.value.split(/:(.*)/s); // split on first ':'
  cueEl.textContent = `▶ ${kind} · ${name}`;
  try { playAction(kind, name); } catch (e) { console.warn(e); }
});
