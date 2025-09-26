// --- Imports (browser modules from CDNs) ---
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
import { CreateMLCEngine } from 'https://esm.run/@mlc-ai/web-llm';

// --- Global variables ---
let INDEX = [];
let qEmbedder, llm;
const statusEl = document.getElementById('status');
const qEl = document.getElementById('question');
const askBtn = document.getElementById('ask');
const ansEl = document.getElementById('answer');
const srcEl = document.getElementById('sources');
const topkEl = document.getElementById('topk');

function setStatus(text) {
  statusEl.textContent = text;
}

async function loadIndex() {
  setStatus('Loading document index…');
  const resp = await fetch('./public/index.json');
  const data = await resp.json();
  INDEX = data.records || [];
  if (!INDEX.length) throw new Error('No records found in index.json');
}

async function initEmbeddings() {
  setStatus('Loading embedding model…');
  qEmbedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}


async function initLLM() {
  setStatus('Initializing WebLLM model (downloads on first run)…');
  llm = await CreateMLCEngine('Llama-3.2-3B-Instruct-q4f16_1', {
    initProgressCallback: (p) => setStatus(`WebLLM: ${p.text || ''}`)
  });
  setStatus('Ready! Ask a question.');
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function cosineSim(qv, dv) { // embeddings are normalized -> dot == cosine
  return dot(qv, dv);
}

function topKSimilar(queryVec, k = 5) {
  const scores = INDEX.map((rec, i) => ({ i, s: cosineSim(queryVec, rec.embedding) }));
  scores.sort((a, b) => b.s - a.s);
  return scores.slice(0, k).map(({ i, s }) => ({ ...INDEX[i], score: s }));
}


function buildPrompt(question, passages) {
  const rules = `You are a careful literary assistant. Answer the user's question using ONLY the passages below. If unsure, say you cannot find it. Quote page ranges when helpful. Keep answers concise (<= 6 sentences).`;
  const context = passages.map((p, idx) => `# Passage ${idx + 1} (score=${p.score.toFixed(3)}, ${p.book}, pp. ${p.pages[0]}-${p.pages[1]})\n${p.text}`).join('\n\n');
  const user = `Question: ${question}\n\nUse the passages to answer. If the answer requires interpretation, explain briefly and cite which passages you used.`;
  return [
    { role: 'system', content: rules },
    { role: 'user', content: context + '\n\n' + user }
  ];
}


async function ask() {
  const question = qEl.value.trim();
  if (!question) return;
  ansEl.textContent = 'Thinking…';
  srcEl.innerHTML = '';

  // Embed query
  const out = await qEmbedder(question, { pooling: 'mean', normalize: true });
  const qv = Array.from(out.data);

  // Retrieve
  const k = Math.max(1, Math.min(10, parseInt(topkEl.value || '5', 10)));
  const passages = topKSimilar(qv, k);

  // Build prompt and run LLM locally
  const messages = buildPrompt(question, passages);
  const completion = await llm.chat.completions.create({ messages, temperature: 0.2, max_tokens: 512 });
  const text = completion.choices?.[0]?.message?.content || '(no response)';
  ansEl.textContent = text;

  // Show sources
  passages.forEach(p => {
    const div = document.createElement('div');
    div.className = 'source';
    div.innerHTML = `<div class="meta"><strong>Source:</strong> ${p.source} — <em>${p.book}</em> (score ${p.score.toFixed(3)})</div>` +
      `<div>${escapeHtml(p.text.slice(0, 800))}${p.text.length > 800 ? '…' : ''}</div>`;
    srcEl.appendChild(div);
  });
}


function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
}

// Init
(async () => {
  try {
    await loadIndex();
    await initEmbeddings();
    await initLLM();
  } catch (e) {
    console.error(e);
    setStatus('Error during initialization. See console for details.');
  }
})();

askBtn.addEventListener('click', ask);
qEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') ask();
});