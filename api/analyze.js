// /api/analyze.js
// Vercel Serverless Function — PRD Analysis via Gemini API

const { GoogleGenerativeAI } = require('@google/generative-ai');

// ─── PROMPT ─────────────────────────────────────────────────────────────────
function buildPrompt(prd, intensity) {
  const toneMap = {
    mild:   'warm and encouraging, with gentle wit',
    spicy:  'sharp, witty and critically honest — use Hinglish naturally where it adds punch',
    savage: 'devastatingly brutal and mercilessly funny — like a principal PM who has read 1000 bad PRDs',
  };
  const tone = toneMap[intensity] || toneMap.spicy;

  return `You are a world-class senior product critic reviewing a PRD. Tone: ${tone}.

MANDATORY PROCESS (follow strictly before generating output):

STEP 1 — PARSE SECTIONS
Identify which of these 7 sections exist vs are absent:
Problem Statement | User Persona | Goals/Success Metrics | Features/Scope | User Flows | Edge Cases | Assumptions

STEP 2 — SCORE EACH DIMENSION
- Clarity (0-100): Are statements specific? Can an engineer build from this?
- Structure (0-100): Is the doc logically organised? Does it flow problem→user→goals→solution?
- ProductThinking (0-100): Is there user empathy, quantified problems, "why now"?
- Completeness (0-100): Are all critical sections covered with enough depth?

STEP 3 — QUOTE RULE (STRICT)
Every roast line and every fix MUST either:
  (a) Quote an exact phrase from the PRD in "double quotes", OR
  (b) Name a specific absent section by name
FORBIDDEN: Generic lines like "metrics are missing" with no text reference.
GOOD: "The Goals section says 'increase engagement' — engagement is not a metric. Define the event, baseline, and target."

STEP 4 — IMPROVEMENT GUIDE
If overallScore < 75, generate 3-5 ordered nextSteps: concrete PM-voice instructions referencing actual gaps.

Return ONLY this raw JSON. No markdown. No explanation. No preamble. No backticks.

{
  "overallScore": <integer 0-100>,
  "clarity": <integer 0-100>,
  "clarityReason": "<1 sentence quoting PRD text or naming issue>",
  "structure": <integer 0-100>,
  "structureReason": "<1 sentence>",
  "productThinking": <integer 0-100>,
  "productThinkingReason": "<1 sentence>",
  "completeness": <integer 0-100>,
  "completenessReason": "<1 sentence>",
  "sectionsFound": ["<section name>"],
  "sectionsMissing": ["<section name>"],
  "roasts": ["<line quoting PRD text or naming absent section>", "<line 2>", "<line 3>"],
  "applause": ["<genuine praise referencing actual PRD strength>", "<line 2>", "<line 3>"],
  "fixes": ["<actionable fix quoting weak phrase or naming missing section>", "<fix 2>", "<fix 3>"],
  "nextSteps": ["<step 1 — only if overallScore < 75>", "<step 2>", "<step 3>"],
  "bestRoast": "<single most surgical roast line quoting PRD>",
  "bestApplause": "<single strongest genuine praise line>"
}

PRD to review:
---
${prd.substring(0, 4000)}
---`;
}

// ─── JSON EXTRACTOR ──────────────────────────────────────────────────────────
/**
 * extractJSON(text)
 *
 * Gemini can return JSON wrapped in many messy ways:
 *   • Plain JSON                          ✓ direct parse
 *   • ```json … ```  fences               ✓ strip fences
 *   • ``` … ```  fences (no lang tag)     ✓ strip fences
 *   • Preamble text then JSON             ✓ find first { in text
 *   • Trailing commentary after JSON      ✓ balanced-brace extraction
 *   • Unicode zero-width / BOM chars      ✓ strip before parse
 *   • Single-quoted strings               ✓ convert to double-quoted
 *   • Trailing commas  { "a":1, }         ✓ remove before parse
 *   • Unquoted keys    { score: 42 }      ✓ quote bare keys
 *   • Escaped newlines inside strings     ✓ handled by JSON.parse
 *
 * Returns parsed object or throws with a descriptive message.
 */
function extractJSON(raw) {
  if (!raw || typeof raw !== 'string') throw new Error('Empty response from Gemini');

  // 1. Strip BOM and zero-width characters
  let text = raw.replace(/^\uFEFF/, '').replace(/[\u200B-\u200D\uFEFF]/g, '').trim();

  // 2. Try direct parse first (cheapest path — works when Gemini behaves)
  try { return JSON.parse(text); } catch (_) {}

  // 3. Strip markdown code fences  ```json ... ```  or  ``` ... ```
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) {
    try { return JSON.parse(fenceMatch[1].trim()); } catch (_) {}
  }

  // 4. Extract the first balanced { … } block from anywhere in the text
  //    (handles "Here is the analysis:\n{...}" preamble patterns)
  const firstBrace = text.indexOf('{');
  if (firstBrace !== -1) {
    let depth = 0, inString = false, escape = false;
    for (let i = firstBrace; i < text.length; i++) {
      const ch = text[i];
      if (escape)          { escape = false; continue; }
      if (ch === '\\')     { escape = true;  continue; }
      if (ch === '"')      { inString = !inString; continue; }
      if (inString)        continue;
      if (ch === '{')      depth++;
      else if (ch === '}') { depth--; if (depth === 0) {
        const candidate = text.slice(firstBrace, i + 1);
        try { return JSON.parse(candidate); } catch (_) {}

        // 5. Aggressive cleanup on the candidate before retrying
        const fixed = candidate
          // trailing commas before } or ]
          .replace(/,\s*([}\]])/g, '$1')
          // single-quoted string values → double-quoted  (simple heuristic)
          .replace(/:\s*'([^'\\]*(\\.[^'\\]*)*)'/g, ': "$1"')
          // unquoted object keys  {  key: value  }
          .replace(/([{,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)\s*:/g, '$1"$2":')
          // stray newlines inside strings (Gemini sometimes adds literal \n)
          .replace(/("(?:[^"\\]|\\.)*")|(\n)/g, (m, str) => str ? str : ' ');

        try { return JSON.parse(fixed); } catch (e2) {
          throw new Error(`JSON extraction failed after cleanup: ${e2.message}`);
        }
      }}
    }
  }

  throw new Error('No JSON object found in Gemini response');
}

// ─── FIELD SANITISER ─────────────────────────────────────────────────────────
/**
 * sanitiseResponse(parsed)
 *
 * Accepts the raw parsed object (keys may differ slightly from expected)
 * and returns a guaranteed-shape response with all fields present,
 * all numbers clamped, all arrays trimmed, all strings safe.
 */
function sanitiseResponse(parsed) {
  const clamp  = (v, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, Math.round(Number(v) || 50)));
  const str    = (v, fallback = '')    => (typeof v === 'string' && v.trim()) ? v.trim() : fallback;
  const arr    = (v, limit = 5)        => (Array.isArray(v) ? v : []).map(String).filter(Boolean).slice(0, limit);

  // Handle slight key name variations Gemini sometimes produces
  // e.g. "product_thinking" vs "productThinking" vs "ProductThinking"
  const get = (keys) => {
    for (const k of keys) {
      if (parsed[k] !== undefined) return parsed[k];
      // case-insensitive scan
      const lk = k.toLowerCase();
      const found = Object.keys(parsed).find(pk => pk.toLowerCase() === lk);
      if (found) return parsed[found];
    }
    return undefined;
  };

  const roasts   = arr(get(['roasts',   'roast']));
  const applause = arr(get(['applause', 'praises', 'praise']));
  const fixes    = arr(get(['fixes',    'fix',  'improvements', 'suggestions']));
  const steps    = arr(get(['nextSteps','next_steps','steps']));

  return {
    overallScore:          clamp(get(['overallScore', 'overall_score', 'score', 'overallScore'])),
    clarity:               clamp(get(['clarity'])),
    clarityReason:         str(get(['clarityReason',  'clarity_reason',  'clarityDescription'])),
    structure:             clamp(get(['structure'])),
    structureReason:       str(get(['structureReason', 'structure_reason', 'structureDescription'])),
    productThinking:       clamp(get(['productThinking', 'product_thinking', 'productThought'])),
    productThinkingReason: str(get(['productThinkingReason', 'product_thinking_reason', 'productThinkingDescription'])),
    completeness:          clamp(get(['completeness'])),
    completenessReason:    str(get(['completenessReason', 'completeness_reason', 'completenessDescription'])),
    sectionsFound:         arr(get(['sectionsFound',   'sections_found',   'foundSections']),  10),
    sectionsMissing:       arr(get(['sectionsMissing', 'sections_missing', 'missingSections']), 10),
    roasts,
    applause,
    fixes,
    nextSteps:             steps.filter(Boolean),
    bestRoast:             str(get(['bestRoast',    'best_roast']))    || roasts[0]   || '',
    bestApplause:          str(get(['bestApplause', 'best_applause'])) || applause[0] || '',
  };
}

// ─── HANDLER ────────────────────────────────────────────────────────────────
module.exports = async function handler(req, res) {
  // CORS headers — allow any origin for demo use
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // ── Validate input ──────────────────────────────────────────────────────
  const { prd, intensity } = req.body || {};

  if (!prd || typeof prd !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid "prd" field' });
  }

  if (prd.trim().length < 20) {
    return res.status(400).json({ error: 'PRD text is too short — paste the full document' });
  }

  // ── Check API key ────────────────────────────────────────────────────────
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  if (!GEMINI_API_KEY) {
    console.error('GEMINI_API_KEY environment variable is not set');
    return res.status(500).json({ error: 'Server configuration error — API key not set' });
  }

  // ── Call Gemini ──────────────────────────────────────────────────────────
  try {
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

    const prompt = buildPrompt(prd, intensity || 'spicy');

    const result = await model.generateContent(prompt);
    const rawText = result.response.text();

    // ── Extract + sanitise JSON ──────────────────────────────────────────
    let parsed;
    try {
      parsed = extractJSON(rawText);
    } catch (parseError) {
      console.error('[analyze] JSON extraction failed:', parseError.message);
      console.error('[analyze] Raw response (first 600 chars):', rawText.substring(0, 600));
      return res.status(502).json({
        error: 'AI returned an unreadable response — please try again',
      });
    }

    const response = sanitiseResponse(parsed);
    return res.status(200).json(response);

  } catch (err) {
    console.error('Gemini API error:', err.message || err);

    // Specific error messages for common failures
    if (err.message?.includes('API_KEY_INVALID') || err.message?.includes('403')) {
      return res.status(401).json({ error: 'Invalid Gemini API key — check your GEMINI_API_KEY env variable' });
    }
    if (err.message?.includes('QUOTA') || err.message?.includes('429')) {
      return res.status(429).json({ error: 'Gemini quota exceeded — try again in a moment' });
    }
    if (err.message?.includes('SAFETY')) {
      return res.status(422).json({ error: 'Content flagged by safety filter — try a different PRD' });
    }

    return res.status(500).json({ error: 'Even AI gave up on this PRD 😅 — please try again' });
  }
};
