import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1';
import { CreateMLCEngine } from 'https://esm.run/@mlc-ai/web-llm@0.2.70';

let INDEX = [];
let qEmbedder, llm;
let ready = false;

const statusEl = document.getElementById('status');
const qEl = document.getElementById('question');
const askBtn = document.getElementById('ask');
const ansEl = document.getElementById('answer');
const srcEl = document.getElementById('sources');
const topkEl = document.getElementById('topk');

function setStatus(t){ statusEl.textContent = t; }
function showError(prefix, e){ console.error(prefix, e); setStatus(`${prefix}: ${e?.message || e}`); }

// --- init steps ---
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
      ready = true;
      askBtn.disabled = false;
      qEl.disabled = false;
      setStatus(`Ready! (${modelId}) Ask a question.`);
      return;
    } catch (e) {
      console.warn(`Model ${modelId} failed:`, e);
      lastErr = e;
    }
  }
  throw new Error(`No supported WebLLM model could be loaded. Last error: ${lastErr?.message || lastErr}`);
}

// --- ask flow ---
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
      messages, temperature: 0.2, max_tokens: 512,
    });
    const text = res.choices?.[0]?.message?.content || '(no response)';
    ansEl.textContent = text;

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

// --- boot ---
(async () => {
  try {
    // disable UI until ready
    askBtn.disabled = true; qEl.disabled = true;

    await loadIndex();
    await initEmbeddings();
    await initLLM();
  } catch (e) {
    showError('Error during initialization', e);
  }
})();
askBtn.addEventListener('click', ask);
qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
