// Avatar Manager
// ----------------------------------------------------------------------------
// Turns a snippet of the live conversation into AVATAR BEHAVIOR cues (mood +
// gesture + gaze) that the browser applies to the TalkingHead avatar. This is
// the "manage avatar by LLM" brain — deliberately separate from the GPT-Realtime
// voice so you can later host it on AWS Bedrock without touching anything else.
//
// Swap backends with AVATAR_MANAGER_BACKEND=openai|bedrock (default openai).
// The Bedrock branch is written out but commented so the app installs/runs with
// zero AWS deps until you have keys.

// Valid vocab — MUST match TalkingHead's built-in moods/gestures. We hard-clamp
// the model's output to these so a hallucinated value can never break playback.
export const MOODS = [
  "neutral", "happy", "angry", "sad", "fear", "disgust", "love", "sleep",
];
export const GESTURES = [
  "handup", "index", "ok", "thumbup", "thumbdown", "side", "shrug",
];
const GAZES = ["camera", "away", null];

// Facial micro-expressions (TalkingHead animated emojis). These layer ON TOP of
// the base mood and are the main lever for moment-to-moment expressiveness — the
// 8 moods are coarse, these give nuance and variety. This exact set is the one
// the in-app test rig verified animates correctly.
export const EMOJIS = [
  "😐", "😶", "😏", "🙂", "🙃", "😊", "😇", "😀", "😃", "😄", "😁", "😆",
  "😝", "😋", "😛", "😜", "🤪", "😂", "🤣", "😅", "😉", "😭", "🥺", "😞",
  "😔", "😳", "☹️", "😚", "😘", "🥰", "😍", "🤩", "😡", "😠", "🤬", "😒",
  "😱", "😬", "🙄", "🤔", "👀", "😴",
];

const SYSTEM_PROMPT = `You are the "avatar manager" for a 3D talking avatar.
You are given a snippet of what the avatar is saying right now. Decide how it
should physically emote AS it speaks that line. Be expressive and natural — a
real person's face shifts constantly, so favor variety over repeating "neutral".

Reply with ONLY a JSON object, no prose:
{
  "mood":      one of ${JSON.stringify(MOODS)},
  "intensity": number 0.0-1.0 (how strongly to show it),
  "emoji":     one of ${JSON.stringify(EMOJIS)} or null,
  "gesture":   one of ${JSON.stringify(GESTURES)} or null,
  "gaze":      "camera" | "away" | null
}

Guidance:
- "mood" is the coarse baseline; keep it sensible (don't swing to "angry" for a
  mildly firm line).
- "emoji" is the star: almost ALWAYS pick one that matches the exact nuance of
  this line (a wink 😉 for a joke, 🤔 while pondering, 🥺 for empathy, 🙂/😊 for
  warmth, 😏 for playful, 😮 for surprise). Vary it turn to turn.
- "intensity": small talk ~0.3, genuine emotion ~0.7-1.0.
- "gesture" sparingly — only when it truly fits (thumbup praise, shrug
  uncertainty, index/handup explaining/greeting). Usually null.
- "gaze": "camera" for direct/engaging moments, "away" when thinking, else null.`;

function clamp01(n) {
  return typeof n === "number" && n >= 0 && n <= 1 ? n : 0.6;
}

function sanitize(obj) {
  const mood = MOODS.includes(obj?.mood) ? obj.mood : "neutral";
  const intensity = clamp01(obj?.intensity);
  const emoji = EMOJIS.includes(obj?.emoji) ? obj.emoji : null;
  const gesture = GESTURES.includes(obj?.gesture) ? obj.gesture : null;
  const gaze = GAZES.includes(obj?.gaze) ? obj.gaze : null;
  return { mood, intensity, emoji, gesture, gaze };
}

// ---- OpenAI backend (gpt-4o-mini) — active now ----------------------------
async function getCuesOpenAI(context) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.AVATAR_MANAGER_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: context },
      ],
      response_format: { type: "json_object" },
      temperature: 0.85, // higher → more varied expression choices
      max_tokens: 80,
    }),
  });
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return JSON.parse(data.choices[0].message.content);
}

// ---- Bedrock backend — UNCOMMENT when you have AWS keys -------------------
// 1) npm i @aws-sdk/client-bedrock-runtime
// 2) at top of file: import { BedrockRuntimeClient, ConverseCommand } from
//      "@aws-sdk/client-bedrock-runtime";
//    const bedrock = new BedrockRuntimeClient({ region: process.env.AWS_REGION });
// 3) set AVATAR_MANAGER_BACKEND=bedrock in .env
//
// async function getCuesBedrock(context) {
//   const out = await bedrock.send(new ConverseCommand({
//     modelId: process.env.BEDROCK_MODEL_ID || "openai.gpt-oss-20b-1:0",
//     system: [{ text: SYSTEM_PROMPT }],
//     messages: [{ role: "user", content: [{ text: context }] }],
//     inferenceConfig: { maxTokens: 80, temperature: 0.5 },
//   }));
//   const text = out.output.message.content.map(c => c.text).join("");
//   // models may wrap JSON in prose/fences — grab the first {...} block:
//   const json = text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
//   return JSON.parse(json);
// }

export async function getAvatarCues(context) {
  const backend = process.env.AVATAR_MANAGER_BACKEND || "openai";
  try {
    let raw;
    if (backend === "bedrock") {
      throw new Error(
        "Bedrock backend selected but not enabled — uncomment getCuesBedrock in avatarManager.js."
      );
      // raw = await getCuesBedrock(context);
    } else {
      raw = await getCuesOpenAI(context);
    }
    return sanitize(raw);
  } catch (err) {
    console.error("[avatarManager]", err.message);
    // safe fallback
    return { mood: "neutral", intensity: 0.5, emoji: null, gesture: null, gaze: null };
  }
}
