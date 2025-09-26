// --- Imports (pin versions + ESM) ---
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1';
import { CreateMLCEngine } from 'https://esm.run/@mlc-ai/web-llm@0.2.70';


// --- Globals / UI refs ---
let INDEX = [];
let qEmbedder, llm;
const statusEl = document.getElementById('status');
const qEl = document.getElementById('question');
const askBtn = document.getElementById('ask');
const ansEl = document.getElementById('answer');
const srcEl = document.getElementById('sources');
const topkEl = document.getElementById('topk');

function setStatus(text) { statusEl.textContent = text; }
function showError(prefix, e) {
  console.error(prefix, e);
  setStatus(`${prefix}: ${e?.message || e}`);
}

// --- Init steps ---
async function loadIndex() {
  setStatus('Loading document index…');
  const resp = await fetch('./public/index.json');
  if (!resp.ok) throw new Error(`index.json ${resp.status}`);
  const data = await resp.json();
  INDEX = data.records || [];
  if (!INDEX.length) throw new Error('No records found in index.json');
}

async function initEmbeddings() {
  setStatus('Loading embedding model (Transformers.js)…');
  qEmbedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}

async function initLLM() {
  setStatus('Initializing WebLLM (first run downloads weights)…');

  const candidates = [
    'Qwen2.5-1.5B-Instruct-q4f16_1',
    'Phi-3-mini-4k-instruct-q4f16_1',
  ];

  let lastErr = null;
  for (const modelId of candidates) {
    try {
      setStatus(`WebLLM: loading ${modelId}…`);
      llm = await CreateMLCEngine(modelId, {
        initProgressCallback: (p) => {
          if (p?.text) setStatus(`WebLLM: ${p.text}`);
          console.log('WebLLM progress', p);
        },
        use_web_worker: false, // important on GitHub Pages
      });
      setStatus(`Ready! (${modelId}) Ask a question.`);
      return;
    } catch (e) {
      console.warn(`Model ${modelId} failed:`, e);
      lastErr = e;
    }
  }
  // If we got here, all candidates failed
  throw new Error(
    `No supported WebLLM model could be loaded. Last error: ${lastErr?.message || lastErr}`
  );
}

// --- Retrieval utils ---
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function cosineSim(qv, dv) { return dot(qv, dv); } // normalized
function topKSimilar(qv, k=5) {
  const scores = INDEX.map((rec, i) => ({ i, s: cosineSim(qv, rec.embedding) }));
  scores.sort((a,b) => b.s - a.s);
  return scores.slice(0, k).map(({i, s}) => ({ ...INDEX[i], score: s }));
}
function buildPrompt(question, passages) {
  const rules = `You are a careful literary assistant. Answer ONLY using the passages. If unsure, say so. Cite page ranges. Keep it concise.`;
  const ctx = passages.map((p, idx) =>
    `# Passage ${idx+1} (score=${p.score.toFixed(3)}, ${p.book}, pp. ${p.pages[0]}-${p.pages[1]})\n${p.text}`
  ).join('\n\n');
  const user = `Question: ${question}\n\nUse the passages above; cite which ones you used.`;
  return [{ role: 'system', content: rules }, { role: 'user', content: ctx + '\n\n' + user }];
}

async function ask() {
  const question = qEl.value.trim();
  if (!question) return;
  ansEl.textContent = 'Thinking…';
  srcEl.innerHTML = '';

  try {
    const out = await qEmbedder(question, { pooling: 'mean', normalize: true });
    const qv = Array.from(out.data);
    const k = Math.max(1, Math.min(10, parseInt(topkEl.value || '5', 10)));
    const passages = topKSimilar(qv, k);
    const messages = buildPrompt(question, passages);

    const completion = await llm.chat.completions.create({
      messages, temperature: 0.2, max_tokens: 512
    });
    const text = completion.choices?.[0]?.message?.content || '(no response)';
    ansEl.textContent = text;

    // sources
    passages.forEach(p => {
      const div = document.createElement('div');
      div.className = 'source';
      div.innerHTML = `<div class="meta"><strong>Source:</strong> ${p.source} — <em>${p.book}</em> (score ${p.score.toFixed(3)})</div>` +
        `<div>${escapeHtml(p.text.slice(0, 800))}${p.text.length>800?'…':''}</div>`;
      srcEl.appendChild(div);
    });
  } catch (e) {
    showError('Ask failed', e);
  }
}

function escapeHtml(s){return s.replace(/[&<>\"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}

// Boot
(async () => {
  try {
    await loadIndex();
    await initEmbeddings();
    await initLLM();
  } catch (e) {
    // setStatus already updated; nothing else to do
  }
})();
askBtn.addEventListener('click', ask);
qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
