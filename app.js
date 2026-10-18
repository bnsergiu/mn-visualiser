(() => {
  'use strict';

  const PYODIDE_INDEX = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/';

  const $ = (id) => document.getElementById(id);
  const els = {
    status: $('status'), statusText: $('status-text'),
    examples: $('examples'), runBtn: $('run-btn'), editBtn: $('edit-btn'),
    editor: $('editor'), codeView: $('code-view'), stdin: $('stdin'),
    firstBtn: $('first-btn'), prevBtn: $('prev-btn'), playBtn: $('play-btn'),
    nextBtn: $('next-btn'), lastBtn: $('last-btn'), slider: $('step-slider'),
    stepLabel: $('step-label'), speed: $('speed'), explain: $('explain'),
    shade: $('shade'), decimals: $('decimals'), maxSteps: $('max-steps'),
    banner: $('banner'), frames: $('frames'), stdout: $('stdout'),
  };

  // Per-viewer conveniences only; everything works without storage.
  const store = {
    get(key, fallback) {
      try {
        const value = localStorage.getItem('mn-visualiser:' + key);
        return value === null ? fallback : value;
      } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem('mn-visualiser:' + key, value); } catch { /* ignore */ }
    },
  };

  let pyodide = null;
  let traceJson = null;
  const view = {
    trace: null, lines: [], lineEls: [], marked: [],
    step: 0, timer: null, mode: 'edit',
  };

  // ---------------------------------------------------------------- helpers

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function setStatus(kind, text) {
    els.status.className = 'status ' + kind;
    els.statusText.textContent = text;
    els.status.title = text;
  }

  const decimals = () => Number(els.decimals.value);

  function fmtCell(v, intLike) {
    if (typeof v === 'boolean') return v ? 'True' : 'False';
    if (typeof v === 'string') return v; // nan, inf, complex
    if (intLike) return String(v);
    if (v !== 0 && Math.abs(v) >= 1e6) return v.toExponential(Math.max(1, decimals() - 1));
    const s = v.toFixed(decimals());
    return /^-0(\.0*)?$/.test(s) ? s.slice(1) : s;
  }

  function fmtScalar(blob) {
    const v = blob.v;
    if (typeof v === 'boolean') return v ? 'True' : 'False';
    if (typeof v === 'string') return v;
    if (/int/.test(blob.py)) return String(v);
    const a = Math.abs(v);
    if (v !== 0 && (a >= 1e6 || a < 1e-3)) return v.toPrecision(4);
    let s = v.toFixed(4).replace(/0+$/, '');
    if (s.endsWith('.')) s += '0';
    return s;
  }

  const isIntLike = (blob) => (
    blob.dtype ? /^(u?int|bool)/.test(blob.dtype) : blob.d.every((x) => typeof x !== 'number' || Number.isInteger(x)));

  const has = (set, i) => set === 'all' || (set instanceof Set && set.has(i));

  // ---------------------------------------------------------------- syntax colouring

  const KEYWORDS = new Set(('False None True and as assert break class continue def del elif else except ' +
    'finally for from global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '));
  const BUILTINS = new Set('print range len input int float abs min max sum round enumerate zip list tuple type str'.split(' '));
  const TOKEN = /(#.*$)|([rbfuRBFU]{0,2}"(?:[^"\\]|\\.)*"?|[rbfuRBFU]{0,2}'(?:[^'\\]|\\.)*'?)|(\b\d+\.?\d*(?:e[+-]?\d+)?\b)|([A-Za-z_]\w*)/g;

  function highlight(line) {
    let out = '';
    let last = 0;
    let m;
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(line))) {
      out += esc(line.slice(last, m.index));
      if (m[1]) out += `<span class="t-com">${esc(m[1])}</span>`;
      else if (m[2]) out += `<span class="t-str">${esc(m[2])}</span>`;
      else if (m[3]) out += `<span class="t-num">${esc(m[3])}</span>`;
      else if (KEYWORDS.has(m[4])) out += `<span class="t-kw">${m[4]}</span>`;
      else if (BUILTINS.has(m[4])) out += `<span class="t-fn">${m[4]}</span>`;
      else out += esc(m[4]);
      last = TOKEN.lastIndex;
    }
    return out + esc(line.slice(last));
  }

  // ---------------------------------------------------------------- Python

  async function boot() {
    try {
      if (typeof loadPyodide !== 'function') {
        throw new Error('Could not load Pyodide from the CDN. Check the internet connection and reload.');
      }
      setStatus('loading', 'Loading Python (first time takes a few seconds)…');
      pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });
      setStatus('loading', 'Loading NumPy…');
      await pyodide.loadPackage('numpy', { messageCallback: () => {} });
      let source;
      try {
        const response = await fetch('tracer.py', { cache: 'no-cache' });
        if (!response.ok) throw new Error(response.statusText);
        source = await response.text();
      } catch {
        throw new Error('Could not read tracer.py. Open the page through a local server (see README).');
      }
      const namespace = pyodide.globals.get('dict')();
      pyodide.runPython(source, { globals: namespace });
      traceJson = namespace.get('trace_json');
      setStatus('ready', 'Python ready');
      els.runBtn.disabled = false;
    } catch (err) {
      setStatus('error', err.message || String(err));
    }
  }

  async function run() {
    if (!traceJson || els.runBtn.disabled) return;
    const code = els.editor.value;
    stopPlay();
    els.runBtn.disabled = true;
    setStatus('busy', 'Running…');
    await new Promise((resolve) => setTimeout(resolve, 20)); // let the status paint
    try {
      try {
        await pyodide.loadPackagesFromImports(code, { messageCallback: () => {} });
      } catch { /* unknown imports surface as errors in the trace */ }
      const limit = Math.min(50000, Math.max(100, Number(els.maxSteps.value) || 3000));
      const trace = JSON.parse(traceJson(code, els.stdin.value, limit));
      showTrace(code, trace);
      setStatus('ready', `${trace.steps.length} steps recorded`);
    } catch (err) {
      showBanner([`Internal error: ${err.message || err}`], true);
      setStatus('ready', 'Python ready');
    } finally {
      els.runBtn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- modes

  function setMode(mode) {
    view.mode = mode;
    const tracing = mode === 'trace';
    els.editor.hidden = tracing;
    els.codeView.hidden = !tracing;
    els.editBtn.hidden = !tracing;
    els.runBtn.textContent = tracing ? 'Run again' : 'Visualise';
    for (const el of [els.firstBtn, els.prevBtn, els.playBtn, els.nextBtn, els.lastBtn, els.slider]) {
      el.disabled = !tracing;
    }
    if (!tracing) {
      stopPlay();
      els.editor.focus();
    }
  }

  function showTrace(code, trace) {
    view.trace = trace;
    view.lines = code.split('\n');
    if (trace.inputs.length && !els.stdin.value.trim()) {
      els.stdin.value = trace.inputs.join('\n'); // keep what was typed into prompt() for the next run
      store.set('stdin', els.stdin.value);
    }

    const notes = [];
    if (trace.error) {
      const where = trace.error.line ? ` on line ${trace.error.line}` : '';
      notes.push(`The code stopped with <b>${esc(trace.error.type)}</b>${where}: ${esc(trace.error.msg)}`);
    }
    if (trace.truncated) {
      notes.push(`Stopped after ${trace.maxSteps} steps (the step limit). Use smaller sizes, e.g. n = 3, or raise the step limit.`);
    }
    showBanner(notes, Boolean(trace.error));

    if (!trace.steps.length) {
      setMode('edit');
      if (trace.error && trace.error.line) selectEditorLine(trace.error.line);
      if (!trace.error) showBanner(['Nothing ran: the code is empty.'], false);
      return;
    }
    buildCodeView();
    setMode('trace');
    els.slider.max = String(trace.steps.length - 1);
    goTo(0);
  }

  function showBanner(paragraphs, isError) {
    els.banner.hidden = paragraphs.length === 0;
    els.banner.className = 'banner' + (isError ? ' error' : '');
    els.banner.innerHTML = paragraphs.map((p) => `<p>${p}</p>`).join('');
  }

  function selectEditorLine(line) {
    const lines = els.editor.value.split('\n');
    const start = lines.slice(0, line - 1).reduce((n, l) => n + l.length + 1, 0);
    els.editor.focus();
    els.editor.setSelectionRange(start, start + (lines[line - 1] || '').length);
  }

  function buildCodeView() {
    els.codeView.innerHTML = view.lines.map((text, i) => (
      `<div class="ln" data-line="${i + 1}"><span class="no">${i + 1}</span>` +
      `<span class="hits"></span><span class="src">${highlight(text) || ' '}</span></div>`
    )).join('');
    view.lineEls = Array.from(els.codeView.children);
    view.marked = [];
  }

  // ---------------------------------------------------------------- stepping

  function goTo(index) {
    const steps = view.trace.steps;
    view.step = Math.max(0, Math.min(steps.length - 1, index));
    render();
  }

  function startPlay() {
    if (!view.trace) return;
    const last = view.trace.steps.length - 1;
    if (view.step >= last) goTo(0);
    view.timer = setInterval(() => {
      if (view.step >= view.trace.steps.length - 1) stopPlay();
      else goTo(view.step + 1);
    }, 1000 / Number(els.speed.value));
    els.playBtn.textContent = '❚❚';
  }

  function stopPlay() {
    clearInterval(view.timer);
    view.timer = null;
    els.playBtn.textContent = '▶';
  }

  const togglePlay = () => (view.timer ? stopPlay() : startPlay());

  function jumpToLine(line) {
    const steps = view.trace.steps;
    for (let k = 1; k <= steps.length; k++) {
      const i = (view.step + k) % steps.length;
      if (steps[i].line === line) {
        stopPlay();
        goTo(i);
        return;
      }
    }
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    const trace = view.trace;
    const step = trace.steps[view.step];
    const prev = view.step > 0 ? trace.steps[view.step - 1] : null;
    const isLast = view.step === trace.steps.length - 1;

    els.slider.value = String(view.step);
    els.stepLabel.textContent = `Step ${view.step + 1} / ${trace.steps.length}`;

    renderCode(step, prev, isLast);
    renderExplain(step, isLast);
    renderFrames(step, prev);
    renderStdout(step, prev);
  }

  function renderCode(step, prev, isLast) {
    for (const el of view.marked) el.classList.remove('current', 'prev', 'err');
    view.marked = [];
    const mark = (from, to, cls) => {
      for (let line = from; line <= to; line++) {
        const el = view.lineEls[line - 1];
        if (el) { el.classList.add(cls); view.marked.push(el); }
      }
    };
    if (prev && prev.line !== step.line) mark(prev.span[0], prev.span[1], 'prev');
    const failed = isLast && step.error && view.trace.error;
    mark(step.span[0], step.span[1], failed ? 'err' : 'current');

    const counts = new Map();
    const steps = view.trace.steps;
    for (let i = 0; i <= view.step; i++) counts.set(steps[i].line, (counts.get(steps[i].line) || 0) + 1);
    view.lineEls.forEach((el, i) => {
      const n = counts.get(i + 1);
      el.children[1].textContent = n > 1 ? `×${n}` : '';
    });

    const el = view.lineEls[step.span[0] - 1];
    if (el) {
      const box = els.codeView;
      const top = el.offsetTop - box.offsetTop;
      if (top < box.scrollTop + 24 || top > box.scrollTop + box.clientHeight - 48) {
        box.scrollTop = top - box.clientHeight / 3;
      }
    }
  }

  function renderExplain(step, isLast) {
    const trace = view.trace;
    const frame = step.state.find((f) => f.uid === step.frame);
    const where = `Line ${step.line}` + (frame && frame.name !== 'Global' ? ` · ${frame.name}` : '');
    const failed = isLast && step.error && trace.error;
    let body;
    if (failed) {
      body = `${esc(trace.error.type)}: ${esc(trace.error.msg)}`;
    } else {
      const text = step.explain || (view.lines[step.line - 1] || '').trim();
      body = esc(text).replace(/→/g, '<span class="arrow">→</span>');
    }
    els.explain.className = 'explain' + (failed ? ' err' : '');
    els.explain.innerHTML = `<span class="where">${esc(where)}</span>${body}`;
  }

  function accessMap(step) {
    const map = new Map();
    const add = (list, kind) => {
      for (const [uid, name, cells] of list) {
        const key = `${uid}:${name}`;
        const entry = map.get(key) || {};
        entry[kind] = cells === null ? 'all' : new Set(cells);
        map.set(key, entry);
      }
    };
    add(step.reads, 'r');
    add(step.writes, 'w');
    return map;
  }

  /** Cells whose value differs from the previous step: null, 'all' or a Set. */
  function changedCells(blob, prevBlob) {
    if (!prevBlob) return 'all';
    if (prevBlob.t !== 'arr' || blob.t !== 'arr' || prevBlob.shape.join() !== blob.shape.join()) return 'all';
    const changed = new Set();
    blob.d.forEach((v, i) => { if (v !== prevBlob.d[i]) changed.add(i); });
    return changed;
  }

  function union(a, b) {
    if (a === 'all' || b === 'all') return 'all';
    if (!a) return b || null;
    if (!b) return a;
    return new Set([...a, ...b]);
  }

  function renderFrames(step, prev) {
    const trace = view.trace;
    const prevFrames = new Map();
    if (prev) for (const f of prev.state) prevFrames.set(f.uid, new Map(f.vars));
    const access = accessMap(step);

    els.frames.innerHTML = step.state.map((frame) => {
      const prevVars = prevFrames.get(frame.uid);
      const chips = [];
      const mats = [];
      for (const [name, index] of frame.vars) {
        const blob = trace.blobs[index];
        const prevIndex = prevVars ? prevVars.get(name) : undefined;
        const prevBlob = prevIndex === undefined ? null : trace.blobs[prevIndex];
        const acc = access.get(`${frame.uid}:${name}`) || {};
        const changed = prevIndex === index ? null : changedCells(blob, prevBlob);
        const written = union(acc.w || null, changed);
        if (blob.t === 'arr') mats.push(renderMatrix(name, blob, prevBlob, acc.r || null, written));
        else if (blob.t === 'big') mats.push(renderBig(name, blob, acc.r, written));
        else chips.push(renderChip(name, blob, acc.r, written));
      }
      const active = frame.uid === step.frame ? ' active' : '';
      const body = chips.length || mats.length
        ? (chips.length ? `<div class="chips">${chips.join('')}</div>` : '') +
          (mats.length ? `<div class="mats">${mats.join('')}</div>` : '')
        : '<div class="frame-empty">No variables yet</div>';
      return `<section class="frame${active}"><div class="frame-title">${esc(frame.name)}</div>${body}</section>`;
    }).join('');
  }

  function renderChip(name, blob, read, written) {
    let value;
    let type = '';
    switch (blob.t) {
      case 'num': value = fmtScalar(blob); break;
      case 'str': value = `'${blob.v}'`; break;
      case 'none': value = 'None'; break;
      case 'repr': value = blob.v; type = blob.py; break;
      default: value = `<${blob.v}>`;
    }
    const cls = written ? ' w' : read ? ' r' : '';
    const label = name === 'return' ? '↩ return' : name;
    const full = blob.t === 'num' ? String(blob.v) : value;
    return `<span class="chip${cls}" title="${esc(`${name} = ${full}`)}"><span class="k">${esc(label)}</span>` +
      `<span class="eq">=</span><span class="v">${esc(value)}</span>${type ? `<span class="ty">${esc(type)}</span>` : ''}</span>`;
  }

  function renderBig(name, blob, read, written) {
    const cls = written ? ' w' : read ? ' r' : '';
    return `<div class="mat"><div class="mat-head"><span class="mat-name${cls}">${esc(name)}</span>` +
      `<span class="mat-meta">${esc(blob.kind)} ${blob.shape.join('×')} · ${esc(blob.dtype)} · too big to draw</span></div>` +
      `<pre class="big">${esc(blob.text)}</pre></div>`;
  }

  function renderMatrix(name, blob, prevBlob, read, written) {
    const two = blob.shape.length === 2;
    const rows = two ? blob.shape[0] : 1;
    const cols = two ? blob.shape[1] : blob.shape[0];
    const intLike = isIntLike(blob);
    const samePrev = prevBlob && prevBlob.t === 'arr' && prevBlob.shape.join() === blob.shape.join();

    // Row / column labels light up when a cell in them is touched.
    const rowHit = new Map();
    const colHit = new Map();
    for (const [set, kind] of [[read, 'r'], [written, 'w']]) {
      if (!(set instanceof Set)) continue;
      for (const i of set) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        if (rowHit.get(r) !== 'w') rowHit.set(r, kind);
        if (colHit.get(c) !== 'w') colHit.set(c, kind);
      }
    }

    let lo = Infinity;
    let hi = -Infinity;
    if (els.shade.checked) {
      for (const v of blob.d) {
        if (typeof v === 'number' && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
    }
    const shadeOf = (v) => {
      if (!(typeof v === 'number' && Number.isFinite(v) && hi > -Infinity)) return '';
      let alpha;
      let tone = '--heat-pos';
      if (lo < 0 && hi > 0) {
        const m = Math.max(-lo, hi);
        alpha = Math.abs(v) / m;
        if (v < 0) tone = '--heat-neg';
      } else {
        alpha = hi === lo ? 0.5 : (v - lo) / (hi - lo);
        if (hi <= 0) { tone = '--heat-neg'; alpha = 1 - alpha; }
      }
      return ` style="background: rgb(var(${tone}) / ${(0.06 + 0.5 * alpha).toFixed(3)})"`;
    };

    let html = `<div class="grid${cols > 10 ? ' compact' : ''}" style="grid-template-columns: auto repeat(${cols}, max-content)">`;
    html += '<div class="ix"></div>';
    for (let c = 0; c < cols; c++) html += `<div class="ix ${colHit.get(c) || ''}">${c}</div>`;
    for (let r = 0; r < rows; r++) {
      html += `<div class="ix ${two ? rowHit.get(r) || '' : ''}">${two ? r : ''}</div>`;
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const v = blob.d[i];
        const cls = (has(read, i) ? ' r' : '') + (has(written, i) ? ' w' : '');
        let title = `${name}[${two ? `${r}, ` : ''}${c}] = ${v}`;
        if (samePrev && prevBlob.d[i] !== v) title += `  (was ${prevBlob.d[i]})`;
        html += `<div class="cell${cls}" title="${esc(title)}"${shadeOf(v)}>${esc(fmtCell(v, intLike))}</div>`;
      }
    }
    html += '</div>';

    const nameCls = written ? ' w' : read ? ' r' : '';
    const kind = blob.kind === 'ndarray' ? `ndarray · ${blob.dtype}` : blob.kind;
    return `<div class="mat"><div class="mat-head"><span class="mat-name${nameCls}">${esc(name)}</span>` +
      `<span class="mat-meta">${blob.shape.join('×')} ${esc(kind)}</span></div>` +
      `<div class="mat-scroll">${html}</div></div>`;
  }

  function renderStdout(step, prev) {
    const text = view.trace.stdout;
    const start = prev ? prev.out : 0;
    const end = step.out;
    els.stdout.innerHTML = esc(text.slice(0, Math.min(start, end))) +
      (end > start ? `<mark>${esc(text.slice(start, end))}</mark>` : '');
    els.stdout.scrollTop = els.stdout.scrollHeight;
  }

  // ---------------------------------------------------------------- wiring

  function loadExample(index) {
    const example = window.EXAMPLES[index];
    if (!example) return;
    const current = els.editor.value;
    const untouched = !current.trim() || window.EXAMPLES.some((e) => e.code === current);
    if (!untouched && !window.confirm('Replace the code in the editor with this example?')) return;
    els.editor.value = example.code;
    els.stdin.value = example.stdin;
    store.set('code', example.code);
    store.set('stdin', example.stdin);
    setMode('edit');
    if (traceJson) run();
  }

  function init() {
    window.EXAMPLES.forEach((example, i) => {
      els.examples.add(new Option(example.title, String(i)));
    });
    const fallback = window.EXAMPLES[1];
    els.editor.value = store.get('code', fallback.code);
    els.stdin.value = store.get('stdin', fallback.stdin);
    els.decimals.value = store.get('decimals', '2');
    els.speed.value = store.get('speed', '4');
    els.shade.checked = store.get('shade', '0') === '1';

    els.examples.addEventListener('change', () => {
      const index = els.examples.value;
      els.examples.value = '';
      if (index !== '') loadExample(Number(index));
    });
    els.runBtn.addEventListener('click', run);
    els.editBtn.addEventListener('click', () => setMode('edit'));
    els.editor.addEventListener('input', () => store.set('code', els.editor.value));
    els.stdin.addEventListener('input', () => store.set('stdin', els.stdin.value));
    els.editor.addEventListener('keydown', (e) => {
      if (e.key === 'Tab' && !e.shiftKey) {
        e.preventDefault();
        els.editor.setRangeText('    ', els.editor.selectionStart, els.editor.selectionEnd, 'end');
        store.set('code', els.editor.value);
      }
    });

    els.firstBtn.addEventListener('click', () => { stopPlay(); goTo(0); });
    els.prevBtn.addEventListener('click', () => { stopPlay(); goTo(view.step - 1); });
    els.nextBtn.addEventListener('click', () => { stopPlay(); goTo(view.step + 1); });
    els.lastBtn.addEventListener('click', () => { stopPlay(); goTo(Infinity); });
    els.playBtn.addEventListener('click', togglePlay);
    els.slider.addEventListener('input', () => { stopPlay(); goTo(Number(els.slider.value)); });
    els.speed.addEventListener('change', () => {
      store.set('speed', els.speed.value);
      if (view.timer) { stopPlay(); startPlay(); }
    });
    els.decimals.addEventListener('change', () => {
      store.set('decimals', els.decimals.value);
      if (view.trace && view.trace.steps.length) render();
    });
    els.shade.addEventListener('change', () => {
      store.set('shade', els.shade.checked ? '1' : '0');
      if (view.trace && view.trace.steps.length) render();
    });
    els.codeView.addEventListener('click', (e) => {
      const line = e.target.closest('.ln');
      if (line && view.trace) jumpToLine(Number(line.dataset.line));
    });

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        run();
        return;
      }
      if (view.mode !== 'trace' || e.target.closest('textarea, input, select')) return;
      const actions = {
        ArrowRight: () => { stopPlay(); goTo(view.step + 1); },
        ArrowLeft: () => { stopPlay(); goTo(view.step - 1); },
        Home: () => { stopPlay(); goTo(0); },
        End: () => { stopPlay(); goTo(Infinity); },
        ' ': togglePlay,
      };
      if (actions[e.key]) {
        e.preventDefault();
        actions[e.key]();
      }
    });

    boot();
  }

  init();
})();
