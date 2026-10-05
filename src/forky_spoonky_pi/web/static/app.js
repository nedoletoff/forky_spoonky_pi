const el = id => document.getElementById(id);
let isFrozen = false;
let trainChart = null;
let trainPollTimer = null;

function toast(msg) {
  const t = el('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove('show'), 1800);
}

function setMode(frozen) {
  isFrozen = frozen;
  el('btn-snap').disabled = frozen;
  el('btn-resume').disabled = !frozen;
  el('btn-redetect').disabled = !frozen;
  const tag = el('live-tag');
  tag.classList.toggle('frozen', frozen);
  tag.textContent = frozen ? 'Frozen' : 'Live';
}

async function api(path, opts = {}) {
  try {
    const r = await fetch(path, { cache: 'no-store', ...opts });
    if (!r.ok) {
      let err = 'HTTP ' + r.status;
      try { err = (await r.json()).error || err; } catch (e) {}
      throw new Error(err);
    }
    const ct = r.headers.get('content-type') || '';
    return ct.includes('json') ? await r.json() : await r.text();
  } catch (e) {
    toast('💥 Облом: ' + e.message);
    return null;
  }
}

// ---------- Live-контролы ----------
el('btn-snap').onclick = async () => {
  const r = await api('/snapshot', { method: 'POST' });
  if (r && r.ok) { setMode(true); toast('📸 Кадр в коллекции'); loadGallery(); }
};
el('btn-resume').onclick = async () => {
  const r = await api('/resume', { method: 'POST' });
  if (r && r.ok) { setMode(false); toast('▶ Поток ожил'); }
};
el('btn-redetect').onclick = async () => {
  toast('🔁 Считаю заново…');
  const r = await api('/redetect', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold: parseFloat(el('thr').value) }),
  });
  if (r && r.ok) toast('✅ Готово за ' + r.infer_ms + ' мс');
};
el('btn-download').onclick = () => { window.location = '/download'; };

el('thr').addEventListener('input', e => {
  const v = parseFloat(e.target.value).toFixed(2);
  el('thr-val').textContent = v;
  el('f-thr').textContent = v;
});
el('thr').addEventListener('change', async e => {
  const v = parseFloat(e.target.value);
  await api('/config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold: v }),
  });
  if (isFrozen) {
    await api('/redetect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold: v }),
    });
  }
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || !el('editor-modal').hidden) return;
  const k = e.key.toLowerCase();
  if (k === ' ' || k === 's') { e.preventDefault(); el('btn-snap').click(); }
  else if (k === 'r' || k === 'escape') { e.preventDefault(); el('btn-resume').click(); }
  else if (k === 'd') { e.preventDefault(); el('btn-download').click(); }
});

// ---------- Галерея ----------
async function loadGallery() {
  const list = await api('/snapshots');
  if (!Array.isArray(list)) return;
  el('snap-count').textContent = list.length;
  const g = el('gallery');
  g.innerHTML = '';
  el('gallery-empty').style.display = list.length ? 'none' : 'block';

  for (const s of list) {
    const card = document.createElement('div');
    card.className = 'snap-card';
    card.innerHTML = `
      <div class="snap-thumb">
        <img src="/snapshots/${s.id}/thumb?t=${Date.now()}" loading="lazy" alt="">
        <span class="snap-badge ${s.labeled ? 'ok' : ''}">
          ${s.labeled ? '✔ ' + s.labels_count : 'без меток'}
        </span>
      </div>
      <div class="snap-meta">
        <div class="snap-time">${s.created_str}</div>
        <div class="snap-det">🍴 ${s.fork} · 🥄 ${s.spoon} · ❓ ${s.unknown}</div>
      </div>
      <div class="snap-actions">
        <button data-act="view" title="Показать в потоке">👁</button>
        <button data-act="label" title="Разметить вручную">✏️</button>
        <button data-act="auto" title="Разметить YOLO">🪄</button>
        <button data-act="dl" title="Скачать кадр">⬇</button>
        <button data-act="del" title="Удалить">🗑</button>
      </div>
    `;
    card.querySelector('[data-act="view"]').onclick = () => loadToView(s.id);
    card.querySelector('[data-act="label"]').onclick = () => openEditor(s.id);
    card.querySelector('[data-act="auto"]').onclick = () => autoLabel(s.id);
    card.querySelector('[data-act="dl"]').onclick = () => { window.location = `/snapshots/${s.id}/download?annotated=1`; };
    card.querySelector('[data-act="del"]').onclick = () => deleteSnap(s.id);
    g.appendChild(card);
  }
}

async function autoLabel(id) {
  const r = await api('/snapshots/' + id + '/detect', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold: parseFloat(el('thr').value) }),
  });
  if (r && r.ok) {
    const save = await api('/snapshots/' + id + '/labels', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labels: r.labels }),
    });
    if (save && save.ok) { toast('🪄 Разметил: ' + r.labels.length + ' боксов'); loadGallery(); }
  }
}

async function loadToView(id) {
  const r = await api('/snapshots/' + id + '/load', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold: parseFloat(el('thr').value) }),
  });
  if (r && r.ok) { setMode(true); toast('📌 Кадр приколот к потоку'); }
}

async function deleteSnap(id) {
  if (!confirm('Стереть кадр ' + id + '?')) return;
  const r = await api('/snapshots/' + id, { method: 'DELETE' });
  if (r && r.ok) { toast('🗑 Стерли'); loadGallery(); }
}

el('btn-reprocess-all').onclick = async () => {
  if (!confirm('Пересчитать все кадры текущим порогом?')) return;
  toast('🔁 Считаю заново…');
  const r = await api('/snapshots/reprocess-all', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold: parseFloat(el('thr').value) }),
  });
  if (r && r.ok) { toast('✅ Готово: ' + r.processed + ' кадров'); loadGallery(); }
};

el('btn-export').onclick = async () => {
  const r = await api('/dataset/export', { method: 'POST' });
  if (r && r.ok) toast(`📦 Датасет собран: train ${r.train} / val ${r.val}`);
};

el('btn-zip-clean').onclick = () => { window.location = '/photos/zip?annotated=0'; };
el('btn-zip-annotated').onclick = () => { window.location = '/photos/zip?annotated=1'; };

// ---------- Редактор разметки ----------
let editor = null;

document.querySelectorAll('.editor-toolbar .tool').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.editor-toolbar .tool').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (editor) editor.tool = btn.dataset.tool;
  };
});

el('editor-close').onclick = () => { el('editor-modal').hidden = true; editor = null; };

async function openEditor(id) {
  const meta = await api('/snapshots/' + id);
  if (!meta) return;

  let labels = Array.isArray(meta.labels) ? meta.labels.slice() : [];
  if (!labels.length && Array.isArray(meta.detections)) {
    const W = meta.width || 640, H = meta.height || 480;
    labels = meta.detections
      .filter(d => d.label === 'fork' || d.label === 'spoon')
      .map(d => {
        const [x1, y1, x2, y2] = d.xyxy;
        return {
          cls_id: d.label === 'fork' ? 0 : 1,
          x: ((x1 + x2) / 2) / W, y: ((y1 + y2) / 2) / H,
          w: (x2 - x1) / W, h: (y2 - y1) / H,
        };
      });
  }

  const wrap = el('editor-wrap');
  wrap.innerHTML = '';

  const img = new Image();
  img.src = '/snapshots/' + id + '/image?t=' + Date.now();
  await img.decode();

  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  wrap.appendChild(img);
  wrap.appendChild(canvas);

  editor = {
    id, img, canvas, ctx: canvas.getContext('2d'),
    naturalW: img.naturalWidth, naturalH: img.naturalHeight,
    boxes: labels,
    tool: document.querySelector('.editor-toolbar .tool.active')?.dataset.tool || 'fork',
    dragStart: null, dragCurrent: null,
  };

  el('editor-id').textContent = '# ' + id;

  const getPos = e => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / rect.width * editor.naturalW,
      y: (e.clientY - rect.top) / rect.height * editor.naturalH,
    };
  };

  canvas.addEventListener('mousedown', e => {
    const pos = getPos(e);
    if (editor.tool === 'delete') {
      const hit = hitTest(pos);
      if (hit >= 0) { editor.boxes.splice(hit, 1); drawEditor(); }
      return;
    }
    editor.dragStart = pos;
    editor.dragCurrent = pos;
  });

  canvas.addEventListener('mousemove', e => {
    if (!editor.dragStart) return;
    editor.dragCurrent = getPos(e);
    drawEditor();
  });

  canvas.addEventListener('mouseup', e => {
    if (!editor.dragStart) return;
    const start = editor.dragStart;
    const end = getPos(e);
    editor.dragStart = null;
    editor.dragCurrent = null;

    const x1 = Math.min(start.x, end.x), y1 = Math.min(start.y, end.y);
    const x2 = Math.max(start.x, end.x), y2 = Math.max(start.y, end.y);
    if (Math.abs(x2 - x1) < 8 || Math.abs(y2 - y1) < 8) { drawEditor(); return; }

    const cls_id = editor.tool === 'spoon' ? 1 : 0;
    editor.boxes.push({
      cls_id,
      x: ((x1 + x2) / 2) / editor.naturalW,
      y: ((y1 + y2) / 2) / editor.naturalH,
      w: (x2 - x1) / editor.naturalW,
      h: (y2 - y1) / editor.naturalH,
    });
    drawEditor();
  });

  canvas.addEventListener('mouseleave', () => {
    if (editor.dragStart) { editor.dragStart = null; editor.dragCurrent = null; drawEditor(); }
  });

  function hitTest(pos) {
    for (let i = editor.boxes.length - 1; i >= 0; i--) {
      const b = editor.boxes[i];
      const x = (b.x - b.w / 2) * editor.naturalW;
      const y = (b.y - b.h / 2) * editor.naturalH;
      const w = b.w * editor.naturalW;
      const h = b.h * editor.naturalH;
      if (pos.x >= x && pos.x <= x + w && pos.y >= y && pos.y <= y + h) return i;
    }
    return -1;
  }

  drawEditor();
  el('editor-modal').hidden = false;
}

function drawEditor() {
  if (!editor) return;
  const { ctx, naturalW: W, naturalH: H, boxes } = editor;
  ctx.clearRect(0, 0, W, H);

  const colors = { 0: '#ffffff', 1: '#9aa0a6' };
  const names = { 0: 'Вилка', 1: 'Ложка' };

  for (const b of boxes) {
    const x = (b.x - b.w / 2) * W;
    const y = (b.y - b.h / 2) * H;
    const w = b.w * W;
    const h = b.h * H;
    const color = colors[b.cls_id] || '#6b7076';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    const text = names[b.cls_id] || '?';
    ctx.font = 'bold 16px Inter, sans-serif';
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 22, tw + 12, 22);
    ctx.fillStyle = b.cls_id === 0 ? '#0a0a0a' : '#0a0a0a';
    ctx.fillText(text, x + 6, y - 6);
  }

  if (editor.dragStart && editor.dragCurrent) {
    const { x: x1, y: y1 } = editor.dragStart;
    const { x: x2, y: y2 } = editor.dragCurrent;
    ctx.strokeStyle = '#ffffff';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
    ctx.setLineDash([]);
  }
}

el('editor-clear').onclick = () => { if (editor) { editor.boxes = []; drawEditor(); } };

el('editor-auto').onclick = async () => {
  if (!editor) return;
  toast('🪄 YOLO колдует…');
  const r = await api('/snapshots/' + editor.id + '/detect', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold: parseFloat(el('thr').value) }),
  });
  if (r && r.ok) { editor.boxes = r.labels; drawEditor(); toast('🪄 Нашёл боксов: ' + r.labels.length); }
};

el('editor-save').onclick = async () => {
  if (!editor) return;
  const r = await api('/snapshots/' + editor.id + '/labels', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels: editor.boxes }),
  });
  if (r && r.ok) {
    toast('💾 Метки в деле');
    el('editor-modal').hidden = true;
    editor = null;
    loadGallery();
  }
};

// ---------- Обучение ----------
function fmtDuration(sec) {
  if (sec == null) return '—';
  sec = Math.round(sec);
  if (sec < 60) return sec + ' с';
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return m + ' мин ' + s + ' с';
  const h = Math.floor(m / 60);
  return h + ' ч ' + (m % 60) + ' мин';
}

function ensureChart() {
  if (trainChart || typeof Chart === 'undefined') return;
  const ctx = el('ov-chart').getContext('2d');
  trainChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'box_loss', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'loss' },
      { label: 'cls_loss', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'loss' },
      { label: 'mAP50', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'metric' },
    ] },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b93a7', maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.05)' } },
        loss: { position: 'left', ticks: { color: '#f59e0b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        metric: { position: 'right', min: 0, max: 1, ticks: { color: '#6366f1' }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

function updateChart(series) {
  ensureChart();
  if (!trainChart) return;
  const n = Math.max(series.box_loss.length, series.mAP50.length);
  trainChart.data.labels = Array.from({ length: n }, (_, i) => i + 1);
  trainChart.data.datasets[0].data = series.box_loss;
  trainChart.data.datasets[1].data = series.cls_loss;
  trainChart.data.datasets[2].data = series.mAP50;
  trainChart.update('none');
}

async function refreshTrainStatus() {
  const s = await api('/train/status');
  if (!s) return;

  const sel = el('t-base');
  if (s.best) {
    const found = Array.from(sel.options).some(o => o.value === s.best);
    if (!found) {
      const opt = document.createElement('option');
      opt.value = s.best;
      opt.textContent = 'fine-tuned: ' + s.name + '/best.pt';
      sel.appendChild(opt);
    }
    el('btn-promote').disabled = false;
    el('btn-promote').dataset.path = s.best;
    el('btn-promote').dataset.name = s.name;
    el('btn-export-ncnn').disabled = false;
    el('btn-export-ncnn').dataset.path = s.best;
  }

  if (s.log) {
    const logEl = el('train-log');
    logEl.textContent = s.log.replace(/\r/g, '\n');
    logEl.scrollTop = logEl.scrollHeight;
    el('ov-log').textContent = s.log.split('\n').slice(-6).join('\n');
  }

  if (s.running) {
    el('btn-train').disabled = true;
    el('btn-train').textContent = '⏳ Учится…';
    el('btn-train-stop').disabled = false;
    showTrainOverlay(true);
    const m = await api('/train/metrics');
    if (m) updateTrainOverlay(m);
  } else {
    el('btn-train').disabled = false;
    el('btn-train').textContent = '🚀 Погнали учить';
    el('btn-train-stop').disabled = true;
    showTrainOverlay(false);
  }
}

function showTrainOverlay(show) {
  const ov = el('train-overlay');
  if (show) ov.hidden = false;
  else { ov.hidden = true; if (trainChart) { trainChart.destroy(); trainChart = null; } }
}

function updateTrainOverlay(m) {
  el('ov-epoch').textContent = m.current_epoch + ' / ' + m.epochs;
  el('ov-eta').textContent = fmtDuration(m.eta_seconds);
  el('ov-avg').textContent = fmtDuration(m.avg_epoch_seconds);
  el('ov-bar').style.width = (m.progress * 100).toFixed(1) + '%';
  updateChart(m.series);
}

el('btn-train').onclick = async () => {
  const payload = {
    base_model: el('t-base').value,
    epochs: parseInt(el('t-epochs').value),
    imgsz: parseInt(el('t-imgsz').value),
    batch: parseInt(el('t-batch').value),
  };
  const r = await api('/train', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (r && r.ok) { el('train-log').classList.add('show'); toast('🚀 Учёба пошла'); refreshTrainStatus(); }
};

el('btn-train-stop').onclick = async () => {
  if (!confirm('Стоп-кран: прервать обучение?')) return;
  const r = await api('/train/stop', { method: 'POST' });
  if (r && r.ok) { toast('⏹ Учёба на паузе'); refreshTrainStatus(); }
};

el('btn-train-log').onclick = () => { el('train-log').classList.toggle('show'); };

el('btn-promote').onclick = async () => {
  const path = el('btn-promote').dataset.path;
  if (!path) return;
  const r = await api('/model/promote', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (r && r.ok) { toast('♻ Рабочая модель обновлена'); el('m-model').textContent = 'fine-tuned'; }
};

el('btn-export-ncnn').onclick = async () => {
  const path = el('btn-export-ncnn').dataset.path;
  if (!path) return;
  toast('📦 Пакуем NCNN…');
  const r = await api('/model/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, format: 'ncnn' }),
  });
  if (r && r.ok) toast('📦 NCNN готов: ' + r.path);
};

// ---------- Live-обновление метрик ----------
async function tick() {
  try {
    const d = await fetch('/stats', { cache: 'no-store' }).then(r => r.json());
    el('c-fork').textContent = d.fork;
    el('c-spoon').textContent = d.spoon;
    el('c-unknown').textContent = d.unknown;
    el('m-fps').textContent = (d.fps || 0).toFixed(1);
    el('m-infer').textContent = d.infer_ms ? (d.infer_ms + ' мс') : '—';
    if (d.model) el('m-model').textContent = d.model;

    if (!el('thr').matches(':active')) {
      el('thr').value = d.threshold;
      el('thr-val').textContent = d.threshold.toFixed(2);
      el('f-thr').textContent = d.threshold.toFixed(2);
    }

    const box = el('status');
    box.classList.remove('offline', 'frozen');
    let label = '🟢 В эфире';
    if (d.status !== 'online') { box.classList.add('offline'); label = '🔴 Камеры нет'; }
    else if (d.mode === 'frozen') { box.classList.add('frozen'); label = '⏸ Кадр замер'; }
    el('status-text').textContent = label;
    setMode(d.mode === 'frozen');
  } catch (e) {
    el('status').classList.add('offline');
    el('status-text').textContent = '📡 Связи нет';
  }
}

setInterval(tick, 700);
tick();
loadGallery();
refreshTrainStatus();
setInterval(refreshTrainStatus, 2000);
