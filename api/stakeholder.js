// /api/stakeholder.js
// Analyses stakeholder review transcript → what to take / politely ignore / note

const { GoogleGenerativeAI } = require('@google/generative-ai');

// ─── SHARED JSON EXTRACTOR (same logic as analyze.js) ────────────────────────
function extractJSON(raw) {
  if (!raw || typeof raw !== 'string') throw new Error('Empty response');
  let text = raw.replace(/^\uFEFF/, '').replace(/[\u200B-\u200D\uFEFF]/g, '').trim();
  try { return JSON.parse(text); } catch (_) {}
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) { try { return JSON.parse(fenceMatch[1].trim()); } catch (_) {} }
  const firstBracket = text.indexOf('[');
  const firstBrace   = text.indexOf('{');
  const start = (firstBracket !== -1 && (firstBrace === -1 || firstBracket < firstBrace))
    ? firstBracket : firstBrace;
  if (start !== -1) {
    const openChar  = text[start];
    const closeChar = openChar === '[' ? ']' : '}';
    let depth = 0, inString = false, escape = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (escape)     { escape = false; continue; }
      if (ch === '\\') { escape = true; continue; }
      if (ch === '"')  { inString = !inString; continue; }
      if (inString)    continue;
      if (ch === openChar) depth++;
      else if (ch === closeChar) {
        depth--;
        if (depth === 0) {
          const candidate = text.slice(start, i + 1);
          try { return JSON.parse(candidate); } catch (_) {}
          const fixed = candidate
            .replace(/,\s*([}\]])/g, '$1')
            .replace(/([{,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)\s*:/g, '$1"$2":');
          try { return JSON.parse(fixed); } catch (e) {
            throw new Error(`JSON cleanup failed: ${e.message}`);
          }
        }
      }
    }
  }
  throw new Error('No JSON found in Gemini response');
}

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  // Explicit body parsing
  let body = req.body;
  if (!body || typeof body === 'string') {
    try { body = JSON.parse(req.body || '{}'); } catch(_) { body = {}; }
  }

  const { transcript } = body;
  if (!transcript || typeof transcript !== 'string' || transcript.trim().length < 10) {
    return res.status(400).json({ error: 'Missing or empty transcript' });
  }

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENAI_API_KEY) {
    return res.status(500).json({ error: 'OPENAI_API_KEY not set in Vercel environment variables' });
  }

  const prompt = `You are an expert product manager coach. A stakeholder design review just happened.
Transcript: "${transcript.substring(0, 3000)}"

Analyse the feedback. Return ONLY a raw JSON array — no markdown, no backticks, no explanation:
[
  {"type":"take","text":"<actionable, user-grounded feedback worth implementing — be specific, quote the transcript>"},
  {"type":"ignore","text":"<feedback that is vague, political, or not user-grounded — explain briefly why to skip>"},
  {"type":"neutral","text":"<something that needs more context or data before deciding>"}
]
Be honest, witty, and specific. Always reference actual words from the transcript.`;

  try {
    const openaiRes = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        max_tokens: 800,
        temperature: 0.7,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: 'You are a product manager coach. Always respond with raw JSON only.' },
          { role: 'user', content: prompt },
        ],
      }),
    });

    if (!openaiRes.ok) {
      const errBody = await openaiRes.json().catch(() => ({}));
      throw new Error(errBody.error?.message || `HTTP ${openaiRes.status}`);
    }

    const data = await openaiRes.json();
    const rawText = data.choices?.[0]?.message?.content || '';

    let parsed;
    try { parsed = extractJSON(rawText); }
    catch (e) {
      console.error('[stakeholder] JSON extraction failed:', e.message);
      return res.status(502).json({ error: 'AI returned an unreadable response — try again' });
    }

    const insights = Array.isArray(parsed) ? parsed : (parsed.insights || parsed.feedback || []);
    const safe = insights
      .filter(i => i && typeof i.type === 'string' && typeof i.text === 'string')
      .map(i => ({ type: ['take','ignore','neutral'].includes(i.type) ? i.type : 'neutral', text: i.text.trim() }))
      .slice(0, 8);
    return res.status(200).json(safe);

  } catch (err) {
    const msg = err.message || String(err);
    console.error('[stakeholder] OpenAI error:', msg);
    if (msg.includes('401')) return res.status(401).json({ error: 'Invalid OpenAI API key' });
    if (msg.includes('429')) return res.status(429).json({ error: 'OpenAI quota exceeded — check platform.openai.com/usage' });
    return res.status(500).json({ error: 'Even AI gave up 😅 — ' + msg.substring(0, 100) });
  }
};
