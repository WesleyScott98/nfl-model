// Serverless proxy so the page can get a written answer without exposing your API key.
// The key lives in Netlify's environment (Site settings -> Environment variables -> ANTHROPIC_API_KEY).
const WINDOW_MS = 60_000, MAX_PER_WINDOW = 20;   // crude shared throttle, keeps costs sane
let hits = [];

export default async (req) => {
  if (req.method !== "POST") return new Response("POST only", { status: 405 });

  const now = Date.now();
  hits = hits.filter(t => now - t < WINDOW_MS);
  if (hits.length >= MAX_PER_WINDOW) return new Response("Busy, try again in a minute", { status: 429 });
  hits.push(now);

  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return new Response("Not configured", { status: 503 });

  let prompt = "";
  try { ({ prompt } = await req.json()); } catch { return new Response("Bad request", { status: 400 }); }
  if (typeof prompt !== "string" || prompt.length > 60_000) return new Response("Bad request", { status: 400 });

  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model: "claude-haiku-4-5-20251001", max_tokens: 700,
                           messages: [{ role: "user", content: prompt }] })
  });
  if (!r.ok) return new Response("Upstream error", { status: 502 });
  const data = await r.json();
  const text = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("\n");
  return new Response(JSON.stringify({ text }), { headers: { "content-type": "application/json" } });
};
