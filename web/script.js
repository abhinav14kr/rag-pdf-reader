// --- Imports (pin stable versions that expose prebuiltAppConfig) ---
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1';
import { CreateMLCEngine, prebuiltAppConfig } from 'https://esm.run/@mlc-ai/web-llm@0.2.70';

// ------------------------ Globals / UI refs ------------------------
let INDEX = [];
let qEmbedder, llm;
let ready = false;

const statusEl = document.getElementById('status');
const qEl = document.getElementById('question');
const askBtn = document.getElementById('ask');
const ansEl = document.getElementById('answer');
const srcEl = document.getElementById('sources');
const topkEl = document.getElementById('topk');

function setStatus(t) { statusEl.textContent = t; }
function showError(prefix, e) {
  console.error(prefix, e);
  setStatus(`${prefix}: ${e?.message || e}`);
}

// ------------------------ Init: index + embeddings -----------------
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

// ------------------------ Init: WebLLM (dynamic model pick) -------
async function initLLM() {
  setStatus('Initializing WebLLM (first run downloads weights)…');

  // 1) Ask web-llm which models it actually bundles
  const available = (prebuiltAppConfig?.model_list || []).map(m => m.model_id);
  console.log('WebLLM available models:', available);
  if (!available.length) {
    throw new Error('No models in prebuiltAppConfig.model_list — check the web-llm version import.');
  }

  // 2) Prefer small, fast models if present; otherwise take the first
  const preferred = [
    'Qwen2.5-1.5B-Instruct-q4f16_1',
    'Qwen2-1.5B-Instruct-q4f16_1',
    'Phi-3-mini-4k-instruct-q4f16_1',
    'Phi-2-q4f16_1',
  ];
  const chosen = preferred.find(id => available.includes(id)) || available[0];
  setStatus(`WebLLM: loading ${chosen}…`);

  // 3) Create engine; avoid web workers on GitHub Pages (no COOP/COEP headers)
  try {
    llm = await CreateMLCEngine(chosen, {
      appConfig: prebuiltAppConfig,
      use_web_worker: false,
      initProgressCallback: (p) => {
        if (p?.text) setStatus(`WebLLM: ${p.text}`);
        console.log('WebLLM progress', p);
      },
    });
  } catch (e) {
    throw new Error(`Failed to load ${chosen}: ${e?.message || e}`);
  }

  // 4) Ready!
  ready = true;
  askBtn.disabled = false;
  qEl.disabled = false;
  setStatus(`Ready! (${chosen}) Ask a question.`);
}

// ------------------------ Retrieval helpers -----------------------
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function cosineSim(qv, dv) { return dot(qv, dv); } // normalized embeddings -> dot == cosine
function topKSimilar(qv, k = 5) {
  const scores = INDEX.map((rec, i) => ({ i, s: cosineSim(qv, rec.embedding) }));
  scores.sort((a, b) => b.s - a.s);
  return scores.slice(0, k).map(({ i, s }) => ({ ...INDEX[i], score: s }));
}
function buildPrompt(question, passages) {
  const rules = `You are a careful literary assistant. Answer ONLY using the passages below. If unsure, say you cannot find it. Cite page ranges. Keep answers concise.`;
  const ctx = passages.map((p, idx) =>
    `# Passage ${idx + 1} (score=${p.score.toFixed(3)}, ${p.book}, pp. ${p.pages[0]}-${p.pages[1]})\n${p.text}`
  ).join('\n\n');
  const user = `Question: ${question}\n\nUse the passages above; cite which ones you used.`;
  return [{ role: 'system', content: rules }, { role: 'user', content: ctx + '\n\n' + user }];
}
function escapeHtml(s) { return s.replace(/[&<>\"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[c])); }

// ------------------------ Ask flow --------------------------------
async function ask() {
  if (!ready || !llm) {
    setStatus('Model is still loading. Please wait until it says “Ready!”.');
    return;
  }
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

    const res = await llm.chat.completions.create({
      messages,
      temperature: 0.2,
      max_tokens: 512,
    });
    const text = res.choices?.[0]?.message?.content || '(no response)';
    ansEl.textContent = text;

    // sources
    passages.forEach(p => {
      const div = document.createElement('div');
      div.className = 'source';
      div.innerHTML =
        `<div class="meta"><strong>Source:</strong> ${p.source} — <em>${p.book}</em> (score ${p.score.toFixed(3)})</div>` +
        `<div>${escapeHtml(p.text.slice(0, 800))}${p.text.length > 800 ? '…' : ''}</div>`;
      srcEl.appendChild(div);
    });
  } catch (e) {
    showError('Ask failed', e);
  }
}

// ------------------------ Boot ------------------------------------
(async () => {
  try {
    // disable UI until ready
    askBtn.disabled = true;
    qEl.disabled = true;

    await loadIndex();
    await initEmbeddings();
    await initLLM();
  } catch (e) {
    showError('Error during initialization', e);
  }
})();

askBtn.addEventListener('click', ask);
qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
