// Serverless proxy so the page can get a written answer without exposing the API key.
// The key lives in Netlify's environment (Site configuration -> Environment variables).
//
// Abuse controls, because this endpoint spends real money:
//   1. Origin/Referer must match this site. Stops curl/script calls that aren't from the page.
//      (CORS would NOT stop those — it only blocks a browser from READING a cross-origin reply.)
//   2. Durable budgets in Netlify Blobs — shared across every instance and cold start, unlike an
//      in-memory counter: a per-IP per-minute cap and a global daily cap.
//   3. Hard caps on prompt size and response tokens.
import { getStore } from "@netlify/blobs";

const PER_IP_PER_MIN = 6;
const GLOBAL_PER_DAY = 400;
const MAX_PROMPT = 40_000;

const deny = (msg, code) => new Response(msg, { status: code });

async function bump(store, key, ttlSeconds) {
  // returns the new count; falls back to 0 (fail-open) only if Blobs itself is unavailable
  try {
    const rec = await store.get(key, { type: "json" });
    const now = Date.now();
    const fresh = rec && now - rec.start < ttlSeconds * 1000 ? rec : { start: now, n: 0 };
    fresh.n += 1;
    await store.setJSON(key, fresh);
    return fresh.n;
  } catch {
    return 0;
  }
}

export default async (req, context) => {
  if (req.method !== "POST") return deny("POST only", 405);

  // 1. only serve requests that came from this site
  const site = (process.env.URL || "").replace(/\/$/, "");
  const origin = req.headers.get("origin") || "";
  const referer = req.headers.get("referer") || "";
  const allowed = [site, process.env.DEPLOY_PRIME_URL, process.env.CUSTOM_DOMAIN].filter(Boolean);
  const from = origin || referer;
  if (site && !allowed.some(a => from.startsWith(a))) return deny("Forbidden", 403);

  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return deny("Not configured", 503);

  // 2. durable budgets
  let store = null;
  try { store = getStore("ask-limits"); } catch { store = null; }
  if (store) {
    const ip = context?.ip || req.headers.get("x-nf-client-connection-ip") || "anon";
    const minute = await bump(store, `ip:${ip}`, 60);
    if (minute > PER_IP_PER_MIN) return deny("Slow down a moment, then try again", 429);
    const day = await bump(store, `day:${new Date().toISOString().slice(0, 10)}`, 86_400);
    if (day > GLOBAL_PER_DAY) return deny("Busy today — picks below still work", 429);
  }

  let prompt = "";
  try { ({ prompt } = await req.json()); } catch { return deny("Bad request", 400); }
  if (typeof prompt !== "string" || !prompt.length || prompt.length > MAX_PROMPT) return deny("Bad request", 400);

  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model: "claude-haiku-4-5-20251001", max_tokens: 700,
                           messages: [{ role: "user", content: prompt }] })
  });
  if (!r.ok) return deny("Upstream error", 502);
  const data = await r.json();
  const text = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("\n");
  return new Response(JSON.stringify({ text }), { headers: { "content-type": "application/json" } });
};
