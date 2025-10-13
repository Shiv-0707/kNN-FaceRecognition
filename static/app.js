// static/app.js — frontend-only. Contains camera enhancer (brightness/contrast/saturation/sharpness),
// preview canvas and option to apply processed frames to frames sent to server.
// Replace your existing static/app.js with this file.

document.addEventListener('DOMContentLoaded', () => {
  (function () {
    // --- DOM refs ---
    const video = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const hiddenCanvas = document.getElementById('hiddenCanvas');
    const ctxO = overlay ? overlay.getContext('2d') : null;
    const ctxH = hiddenCanvas ? hiddenCanvas.getContext('2d') : null;

    const startCam = document.getElementById('startCam');
    const stopCam = document.getElementById('stopCam');
    const startEnroll = document.getElementById('startEnroll');
    const stopEnroll = document.getElementById('stopEnroll');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const downloadBtn = document.getElementById('downloadDataset');
    const generateUID = document.getElementById('generateUID');

    const studentName = document.getElementById('studentName');
    const studentUID = document.getElementById('studentUID');
    const savedCountSpan = document.getElementById('savedCount');
    const targetDisplay = document.getElementById('targetDisplay');
    const message = document.getElementById('message');
    const progressFill = document.getElementById('progressFill');

    // enhancer controls (may be null if HTML not inserted)
    const brightnessEl = document.getElementById('brightness');
    const contrastEl = document.getElementById('contrast');
    const saturationEl = document.getElementById('saturation');
    const sharpnessEl = document.getElementById('sharpness');
    const previewToggle = document.getElementById('previewToggle');
    const applyToSent = document.getElementById('applyToSent');
    const resetEnhancer = document.getElementById('resetEnhancer');

    const brightnessVal = document.getElementById('brightnessVal');
    const contrastVal = document.getElementById('contrastVal');
    const saturationVal = document.getElementById('saturationVal');
    const sharpnessVal = document.getElementById('sharpnessVal');

    // Students UI
    const thumbsContainer = document.getElementById('thumbs');

    // state
    let stream = null;
    let capturing = false;
    let sessionId = null;
    let savedCount = 0;
    let duplicates = 0;
    let captureLoopRunning = false;
    let recognizeInterval = null;

    // enhancer canvas & contexts
    let procCanvas = null;
    let procCtx = null;
    // processing resolution (match hiddenCanvas width/height)
    function getProcSize() {
      const w = hiddenCanvas ? hiddenCanvas.width : 320;
      const h = hiddenCanvas ? hiddenCanvas.height : 240;
      return { w, h };
    }

    // utility
    function setMsg(t, warn = false) {
      if (!message) return;
      message.textContent = t;
      message.style.color = warn ? 'tomato' : '';
    }
    function updateUI() {
      if (savedCountSpan) savedCountSpan.textContent = savedCount;
      if (targetDisplay) targetDisplay.textContent = document.getElementById('targetCount') ? document.getElementById('targetCount').value : '';
      const sessEl = document.getElementById('sessionId');
      if (sessEl) sessEl.textContent = sessionId || '—';
    }

    // ---------- FIXED: robust canvas sizing & DPR handling ----------
    function isVideoMirrored() {
      if (!video) return false;
      try {
        const cs = window.getComputedStyle(video).transform;
        if (!cs || cs === 'none') return false;
        if (cs.includes('scaleX(-1') || cs.includes('scale(-1')) return true;
        // matrix(a, b, c, d, e, f) — horizontal flip detected when a < 0
        const m = cs.match(/matrix\(([-0-9.,\s]+)\)/);
        if (m) {
          const parts = m[1].split(',').map(s => parseFloat(s.trim()));
          if (parts.length >= 1 && parts[0] < 0) return true;
        }
      } catch (e) {}
      return false;
    }

    function resizeCanvases() {
      if (!video || !overlay || !hiddenCanvas) return;
      const frameW = video.videoWidth || video.clientWidth || 640;
      const frameH = video.videoHeight || video.clientHeight || 480;
      const dpr = window.devicePixelRatio || 1;

      // Set overlay backing store to frame size * DPR for sharpness
      overlay.width = Math.max(1, Math.round(frameW * dpr));
      overlay.height = Math.max(1, Math.round(frameH * dpr));
      // Set CSS size to the displayed video size (clientWidth/clientHeight)
      overlay.style.width = (video.clientWidth || frameW) + 'px';
      overlay.style.height = (video.clientHeight || frameH) + 'px';

      // hidden canvas is lower-res for efficient upload/processing
      const maxW = 480;
      const hiddenW = Math.min(maxW, frameW);
      hiddenCanvas.width = Math.max(1, Math.round(hiddenW));
      hiddenCanvas.height = Math.max(1, Math.round(hiddenW * (frameH / Math.max(1, frameW))));

      // If procCanvas exists, match it to hiddenCanvas (processing resolution), and CSS to video display
      if (procCanvas) {
        procCanvas.width = hiddenCanvas.width;
        procCanvas.height = hiddenCanvas.height;
        procCanvas.style.width = (video.clientWidth || frameW) + 'px';
        procCanvas.style.height = (video.clientHeight || frameH) + 'px';
      }

      // Update overlay context after resizing
      if (overlay) {
        // keep 2d context (no context transform) — we'll scale coordinates when drawing
        // clear previous content
        const octx = overlay.getContext('2d');
        octx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }

    // utility to map server/frame coords to overlay backing-store coords
    function mapRectToOverlay(rect, srcW, srcH) {
      // rect: {x,y,w,h} in srcW x srcH coordinate space
      // overlay.width/height are backing store pixels (frameW * DPR)
      const ox = overlay.width / Math.max(1, srcW);
      const oy = overlay.height / Math.max(1, srcH);
      return {
        x: rect.x * ox,
        y: rect.y * oy,
        w: rect.w * ox,
        h: rect.h * oy
      };
    }

    // ---------- end sizing fixes ----------

    async function startCamera() {
      if (stream) return;
      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        setMsg('Camera API not available in this browser. Use https or localhost.', true);
        throw new Error('getUserMedia not supported');
      }
      try {
        const constraints = { video: { facingMode: 'user' }, audio: false };
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        video.srcObject = stream;
        video.play().catch(()=>{});
        initProcCanvasOnce();
        // Wait for metadata to ensure video.videoWidth/videoHeight are available before sizing
        if (video.readyState >= 1) {
          resizeCanvases();
        } else {
          video.addEventListener('loadedmetadata', function onMeta() { resizeCanvases(); video.removeEventListener('loadedmetadata', onMeta); });
        }
        setMsg('Camera started');
      } catch (e) {
        setMsg('Camera error: ' + (e.message || e), true);
        stream = null;
        throw e;
      }
    }

    function stopCamera() {
      if (recognizeInterval) { clearInterval(recognizeInterval); recognizeInterval = null; }
      if (stream) stream.getTracks().forEach(t => t.stop());
      stream = null;
      try { video.pause(); video.srcObject = null; } catch (e) {}
      if (ctxO) ctxO.clearRect(0,0,overlay.width, overlay.height);
      if (procCtx && procCanvas) procCtx.clearRect(0,0,procCanvas.width, procCanvas.height);
      setMsg('Camera stopped');
    }

    // --- Enhancer initialization ---
    function initProcCanvasOnce() {
      if (procCanvas) return;
      procCanvas = document.getElementById('procCanvas') || document.createElement('canvas');
      procCanvas.id = 'procCanvas';
      procCanvas.style.position = 'absolute';
      procCanvas.style.left = '0';
      procCanvas.style.top = '0';
      procCanvas.style.pointerEvents = 'none';
      const container = video?.closest('.video-container') || document.body;
      container.appendChild(procCanvas);
      procCtx = procCanvas.getContext('2d');
      resizeCanvases();
      requestAnimationFrame(processLoop);
      // ensure visible when previewToggle checked
      if (previewToggle && !previewToggle.checked) procCanvas.style.display = 'none';
      if (previewToggle) previewToggle.addEventListener('change', ()=> {
        procCanvas.style.display = previewToggle.checked ? 'block' : 'none';
      });
    }

    // --- Image processing helpers ---
    function clamp(v, a=0, b=255){ return v < a ? a : (v > b ? b : v); }
    function applyColorTransforms(data, bright, contrast, sat) {
      const brightOffset = bright * 2.55;
      const c = contrast;
      const factor = (259 * (c + 255)) / (255 * (259 - c || 0.0001));
      const s = sat;
      for (let i = 0, n = data.length; i < n; i += 4) {
        let r = data[i], g = data[i+1], b = data[i+2];
        r += brightOffset; g += brightOffset; b += brightOffset;
        r = factor * (r - 128) + 128;
        g = factor * (g - 128) + 128;
        b = factor * (b - 128) + 128;
        const gray = 0.299*r + 0.587*g + 0.114*b;
        r = gray + (r - gray) * s;
        g = gray + (g - gray) * s;
        b = gray + (b - gray) * s;
        data[i] = clamp(Math.round(r));
        data[i+1] = clamp(Math.round(g));
        data[i+2] = clamp(Math.round(b));
      }
    }

    function convolveSharpen(srcData, w, h, amount) {
      if (amount <= 0) return srcData;
      const dst = new Uint8ClampedArray(srcData.length);
      const center = 1 + 4 * amount;
      const neigh = -amount;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let r = 0, g = 0, b = 0, a = 0;
          const ci = (y * w + x) * 4;
          r += srcData[ci] * center;
          g += srcData[ci+1] * center;
          b += srcData[ci+2] * center;
          a += srcData[ci+3] * center;
          if (y > 0) { const i = ((y-1)*w + x)*4; r += srcData[i] * neigh; g += srcData[i+1] * neigh; b += srcData[i+2] * neigh; a += srcData[i+3] * neigh; }
          if (y < h-1) { const i = ((y+1)*w + x)*4; r += srcData[i] * neigh; g += srcData[i+1] * neigh; b += srcData[i+2] * neigh; a += srcData[i+3] * neigh; }
          if (x > 0) { const i = (y*w + (x-1))*4; r += srcData[i] * neigh; g += srcData[i+1] * neigh; b += srcData[i+2] * neigh; a += srcData[i+3] * neigh; }
          if (x < w-1) { const i = (y*w + (x+1))*4; r += srcData[i] * neigh; g += srcData[i+1] * neigh; b += srcData[i+2] * neigh; a += srcData[i+3] * neigh; }
          const di = ci;
          dst[di] = clamp(Math.round(r));
          dst[di+1] = clamp(Math.round(g));
          dst[di+2] = clamp(Math.round(b));
          dst[di+3] = clamp(Math.round(a));
        }
      }
      return dst;
    }

    // --- Processing loop (draws processed preview into procCanvas) ---
    function processLoop() {
      requestAnimationFrame(processLoop);
      if (!procCtx || !video || video.paused || video.ended) return;
      const showPreview = previewToggle ? previewToggle.checked : false;
      const sendProcessed = applyToSent ? applyToSent.checked : false;
      if (!showPreview && !sendProcessed) {
        procCtx.clearRect(0,0,procCanvas.width, procCanvas.height);
        return;
      }
      const { w, h } = getProcSize();
      try {
        procCtx.drawImage(video, 0, 0, w, h);
        let img = procCtx.getImageData(0, 0, w, h);
        const data = img.data;
        const bright = brightnessEl ? Number(brightnessEl.value) : 0;
        const contrast = contrastEl ? Number(contrastEl.value) : 0;
        const sat = saturationEl ? (Number(saturationEl.value) / 100.0) : 1.0;
        const sharp = sharpnessEl ? (Number(sharpnessEl.value) / 100.0) : 0.0;
        if (brightnessVal) brightnessVal.textContent = bright;
        if (contrastVal) contrastVal.textContent = contrast;
        if (saturationVal) saturationVal.textContent = Math.round(sat*100);
        if (sharpnessVal) sharpnessVal.textContent = Math.round(sharp*100);
        if (bright !== 0 || contrast !== 0 || sat !== 1.0) {
          applyColorTransforms(data, bright, contrast, sat);
        }
        if (sharp > 0) {
          const conv = convolveSharpen(data, w, h, sharp);
          img.data.set(conv);
        } else {
          img.data.set(data);
        }
        procCtx.putImageData(img, 0, 0);
      } catch (e) {
        // ignore processing errors
      }
    }

    function resetEnhancerValues() {
      if (brightnessEl) brightnessEl.value = 0;
      if (contrastEl) contrastEl.value = 0;
      if (saturationEl) saturationEl.value = 100;
      if (sharpnessEl) sharpnessEl.value = 0;
      if (brightnessVal) brightnessVal.textContent = '0';
      if (contrastVal) contrastVal.textContent = '0';
      if (saturationVal) saturationVal.textContent = '100';
      if (sharpnessVal) sharpnessVal.textContent = '0';
    }
    if (resetEnhancer) resetEnhancer.addEventListener('click', resetEnhancerValues);

    // --- Enrollment & capture pipeline ---
    async function startEnrollment() {
      try { await startCamera(); } catch (e) { setMsg('Camera start failed', true); return; }
      const name = (studentName && studentName.value) ? studentName.value.trim() : '';
      if (!name) { setMsg('Enter a name', true); return; }
      let uid = (studentUID && studentUID.value) ? studentUID.value.trim() : '';
      if (!uid) { uid = `${name.split(/\s+/)[0].toUpperCase()}_${Date.now().toString(36).slice(-6)}`; if (studentUID) studentUID.value = uid; }
      const target = parseInt(document.getElementById('targetCount') ? document.getElementById('targetCount').value : 100, 10) || 100;

      let password = prompt("Set a password to protect this student's images (plaintext will be stored):");
      if (!password) { setMsg('Enrollment cancelled: password required', true); return; }

      try {
        const createRes = await fetch('/api/students/create', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ name, uid, password })
        });
        const createJson = await createRes.json();
        if (!createJson.ok) throw new Error(createJson.error || 'create_failed');

        const r = await fetch('/api/enroll/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, uid, target }) });
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || 'start failed');
        sessionId = j.session_id;
        savedCount = j.count || 0; duplicates = 0;
        updateUI();
        capturing = true;
        captureLoop();
        setMsg('Enrollment started (folder created and password set)');
        await loadStudentsList();
      } catch (e) {
        setMsg('Enroll start failed: ' + (e.message || e), true);
      }
    }

    async function stopEnrollment() {
      capturing = false;
      if (sessionId) {
        try { await fetch('/api/enroll/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId }) }); } catch (e) {}
        sessionId = null;
      }
      if (ctxO) ctxO.clearRect(0,0,overlay.width, overlay.height);
      setMsg('Enrollment stopped');
      updateUI();
      await loadStudentsList();
    }

    async function captureLoop() {
      if (captureLoopRunning) return;
      captureLoopRunning = true;
      const fpsVal = Math.max(1, parseInt(document.getElementById('captureFPS') ? document.getElementById('captureFPS').value : 2, 10));
      const intervalMs = Math.max(200, Math.round(1000 / fpsVal));
      while (capturing) {
        const startTs = performance.now();
        try { await captureAndSend(); } catch (err) { console.warn('captureAndSend error', err); }
        const elapsed = performance.now() - startTs;
        const delay = Math.max(0, intervalMs - elapsed);
        await new Promise(res => setTimeout(res, delay));
      }
      captureLoopRunning = false;
    }

    async function captureAndSend() {
      if (!stream || !capturing || !sessionId || !hiddenCanvas || !ctxH) return;
      resizeCanvases();
      const w = hiddenCanvas.width, h = hiddenCanvas.height;
      const sendProcessed = (applyToSent && applyToSent.checked);
      if (sendProcessed && procCanvas) {
        try {
          ctxH.drawImage(procCanvas, 0, 0, procCanvas.width, procCanvas.height, 0, 0, w, h);
        } catch (e) {
          ctxH.drawImage(video, 0, 0, w, h);
        }
      } else {
        ctxH.drawImage(video, 0, 0, w, h);
      }

      const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.8);
      const res = await fetch('/api/enroll/frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId, image: dataUrl }) });
      const j = await res.json().catch(() => ({ ok: false, error: 'invalid_json' }));

      if (j.count !== undefined) savedCount = j.count;
      const target = j.target || parseInt(document.getElementById('targetCount') ? document.getElementById('targetCount').value : 100, 10) || 100;
      if (progressFill) progressFill.style.width = Math.min(100, Math.round((savedCount / Math.max(1, target)) * 100)) + '%';
      if (targetDisplay) targetDisplay.textContent = target;
      if (j.boxes && Array.isArray(j.boxes)) {
        // boxes from enroll/frame use hiddenCanvas dimensions: map accordingly
        // pass srcW/srcH equal to hiddenCanvas.width/height so drawOverlay maps correctly
        drawOverlay(j.boxes, hiddenCanvas.width || video.videoWidth || 640, hiddenCanvas.height || video.videoHeight || 480);
      }

      if (j.duplicate === true) duplicates++;
      if (j.message === 'no_face') setMsg('No face detected', true);
      else if (j.error) setMsg('Server error: ' + j.error, true);
      else if (j.saved) setMsg(`Saved ${savedCount}/${j.target || ''}`);
      if (j.done) { setMsg('Target reached'); await stopEnrollment(); }
      updateUI();
    }

    // Recognition
    recognizeBtn && recognizeBtn.addEventListener('click', () => {
      if (recognizeInterval) {
        clearInterval(recognizeInterval);
        recognizeInterval = null;
        if (recognizeBtn) recognizeBtn.textContent = 'Recognize Now';
        setMsg('Live recognize stopped');
        if (ctxO) ctxO.clearRect(0,0,overlay.width, overlay.height);
        return;
      }
      recognizeInterval = setInterval(liveRecognize, 800);
      if (recognizeBtn) recognizeBtn.textContent = 'Stop Recognize';
    });

    async function liveRecognize() {
      if (!stream) { try { await startCamera(); } catch (e) { return; } }
      const w = video.videoWidth || 640, h = video.videoHeight || 480;
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      const cc = c.getContext('2d'); cc.drawImage(video, 0, 0, w, h);
      const dataUrl = c.toDataURL('image/jpeg', 0.9);
      try {
        const res = await fetch('/api/recognize/frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: dataUrl }) });
        const j = await res.json();
        // pass the server's frame dims so mapping is accurate
        drawOverlay(j.faces || [], j.width || w, j.height || h);
      } catch (e) { console.warn('recognize err', e); }
    }

    function drawOverlay(faces, srcW, srcH) {
      if (!overlay) return;
      const octx = overlay.getContext('2d');
      // Clear backing-store canvas (use backing-store coords)
      octx.clearRect(0, 0, overlay.width, overlay.height);
      if (!faces || faces.length === 0) return;

      // Determine if preview is mirrored (CSS transform)
      const mirrored = isVideoMirrored();

      // scale factors from source frame to overlay backing store
      const sx = overlay.width / Math.max(1, srcW);
      const sy = overlay.height / Math.max(1, srcH);

      faces.forEach(f => {
        // server returns x,y,x2,y2 and/or w,h; prefer x,y,w,h
        let x = (f.x !== undefined) ? f.x : 0;
        let y = (f.y !== undefined) ? f.y : 0;
        let w = (f.w !== undefined) ? f.w : ((f.x2 !== undefined && f.x !== undefined) ? (f.x2 - f.x) : 0);
        let h = (f.h !== undefined) ? f.h : ((f.y2 !== undefined && f.y !== undefined) ? (f.y2 - f.y) : 0);

        // map to overlay backing-store coordinates
        let rx = x * sx;
        let ry = y * sy;
        let rw = Math.max(0, w * sx);
        let rh = Math.max(0, h * sy);

        if (mirrored) {
          // flip horizontally: new x = backing_width - (rx + rw)
          rx = overlay.width - (rx + rw);
        }

        octx.lineWidth = Math.max(2, Math.round(3 * (window.devicePixelRatio || 1)));
        octx.strokeStyle = f.matched ? 'rgba(45,212,191,0.95)' : 'rgba(255,120,120,0.9)';
        octx.strokeRect(rx, ry, rw, rh);

        const txt = (f.label || 'Unknown') + (f.prob ? ` ${(Math.round(f.prob * 100))}%` : '');
        octx.font = `${Math.max(12, 14 * (window.devicePixelRatio || 1))}px system-ui, Arial`;
        octx.fillStyle = 'rgba(0,0,0,0.6)';
        const tw = (octx.measureText(txt).width || 60) + 12;
        const ty = Math.max(ry - 22, 0);
        octx.fillRect(rx, ty, tw, 20);
        octx.fillStyle = 'white';
        octx.fillText(txt, rx + 6, ty + 14);

        // draw landmarks if present (array of points with x,y in source coords)
        if (f.landmarks && Array.isArray(f.landmarks)) {
          octx.fillStyle = 'red';
          f.landmarks.forEach(pt => {
            let lx = pt.x * sx;
            let ly = pt.y * sy;
            if (mirrored) {
              lx = overlay.width - lx;
            }
            octx.beginPath();
            octx.arc(lx, ly, Math.max(2, 3 * (window.devicePixelRatio || 1)), 0, Math.PI * 2);
            octx.fill();
          });
        }
      });
    }

    // --- Students panel logic (unchanged) ---
    async function loadStudentsList() {
      try {
        const res = await fetch('/api/students');
        const j = await res.json();
        if (!j.ok) { console.warn('failed to load students', j); return; }
        renderStudents(j.students || []);
      } catch (e) {
        console.warn('loadStudentsList err', e);
      }
    }
    function renderStudents(list) {
      if (!thumbsContainer) return;
      thumbsContainer.innerHTML = '';
      list.forEach(s => {
        const el = document.createElement('div');
        el.className = 'thumb-item';
        const placeholder = document.createElement('div');
        placeholder.style.width = '100%';
        placeholder.style.height = '70px';
        placeholder.style.display = 'flex';
        placeholder.style.alignItems = 'center';
        placeholder.style.justifyContent = 'center';
        placeholder.style.background = 'linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))';
        placeholder.innerHTML = `<div style="text-align:center;color:var(--muted);font-size:12px">
          <i class="fas fa-lock"></i><br>${s.key}<br><small>${s.count} images</small></div>`;
        el.appendChild(placeholder);
        const btnBar = document.createElement('div');
        btnBar.style.display = 'flex';
        btnBar.style.gap = '6px';
        btnBar.style.padding = '8px';
        btnBar.style.justifyContent = 'center';
        const openBtn = document.createElement('button');
        openBtn.className = 'btn btn-accent';
        openBtn.textContent = 'Open';
        openBtn.addEventListener('click', () => openStudentPrompt(s.key));
        btnBar.appendChild(openBtn);
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-secondary';
        downloadBtn.textContent = 'Download (auth)';
        downloadBtn.addEventListener('click', () => downloadStudentZip(s.key));
        btnBar.appendChild(downloadBtn);
        el.appendChild(btnBar);
        thumbsContainer.appendChild(el);
      });
      const countEl = document.getElementById('thumbCount');
      if (countEl) countEl.textContent = list.length;
    }
    async function openStudentPrompt(key) {
      const pwd = prompt(`Enter password for ${key}:`);
      if (!pwd) return;
      try {
        const r = await fetch(`/api/students/${encodeURIComponent(key)}/auth`, {
          method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ password: pwd })
        });
        const j = await r.json();
        if (!j.ok) { alert('Auth failed'); return; }
        const token = j.token;
        const L = await fetch(`/api/students/${encodeURIComponent(key)}/list?token=${encodeURIComponent(token)}`);
        const LJ = await L.json();
        if (!LJ.ok) { alert('Could not list images'); return; }
        showStudentImages(key, token, LJ.images || []);
      } catch (e) { console.warn('openStudentPrompt err', e); alert('Error authenticating'); }
    }
    function showStudentImages(key, token, images) {
      if (!thumbsContainer) return;
      thumbsContainer.innerHTML = '';
      const header = document.createElement('div');
      header.style.marginBottom = '10px';
      header.innerHTML = `<strong>${key}</strong> — authenticated (token valid briefly).`;
      thumbsContainer.appendChild(header);
      const grid = document.createElement('div');
      grid.className = 'thumb-grid';
      images.forEach(fn => {
        const item = document.createElement('div');
        item.className = 'thumb-item';
        const img = document.createElement('img');
        img.src = `/api/students/${encodeURIComponent(key)}/image/${encodeURIComponent(fn)}?token=${encodeURIComponent(token)}`;
        img.alt = fn;
        item.appendChild(img);
        grid.appendChild(item);
      });
      thumbsContainer.appendChild(grid);
      const backBtn = document.createElement('button');
      backBtn.className = 'btn btn-secondary';
      backBtn.textContent = 'Back to list';
      backBtn.style.marginTop = '10px';
      backBtn.addEventListener('click', () => loadStudentsList());
      thumbsContainer.appendChild(backBtn);
    }
    async function downloadStudentZip(key) {
      const pwd = prompt(`Enter password to download images for ${key}:`);
      if (!pwd) return;
      try {
        const r = await fetch(`/api/students/${encodeURIComponent(key)}/auth`, {
          method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ password: pwd })
        });
        const j = await r.json();
        if (!j.ok) { alert('Auth failed'); return; }
        const token = j.token;
        const listRes = await fetch(`/api/students/${encodeURIComponent(key)}/list?token=${encodeURIComponent(token)}`);
        const lj = await listRes.json();
        if (!lj.ok) { alert('List failed'); return; }
        const images = lj.images || [];
        if (images.length === 0) { alert('No images'); return; }
        for (const fn of images) {
          const blobRes = await fetch(`/api/students/${encodeURIComponent(key)}/image/${encodeURIComponent(fn)}?token=${encodeURIComponent(token)}`);
          if (blobRes.ok) {
            const b = await blobRes.blob();
            const url = URL.createObjectURL(b);
            const a = document.createElement('a');
            a.href = url; a.download = `${key}_${fn}`; a.click();
            URL.revokeObjectURL(url);
          }
        }
        setMsg('Images downloaded (separate files)', false);
      } catch (e) { console.warn('downloadStudentZip err', e); alert('Download error'); }
    }

    // --- Events wiring ---
    startCam && startCam.addEventListener('click', () => startCamera().catch(() => {}));
    stopCam && stopCam.addEventListener('click', () => { stopEnrollment(); stopCamera(); });
    startEnroll && startEnroll.addEventListener('click', startEnrollment);
    stopEnroll && stopEnroll.addEventListener('click', stopEnrollment);

    // slider live update (visual labels)
    [brightnessEl, contrastEl, saturationEl, sharpnessEl].forEach(el => {
      if (!el) return;
      el.addEventListener('input', () => {
        if (brightnessEl && brightnessVal) brightnessVal.textContent = brightnessEl.value;
        if (contrastEl && contrastVal) contrastVal.textContent = contrastEl.value;
        if (saturationEl && saturationVal) saturationVal.textContent = saturationEl.value;
        if (sharpnessEl && sharpnessVal) sharpnessVal.textContent = sharpnessEl.value;
      });
    });

    downloadBtn && downloadBtn.addEventListener('click', async () => {
      setMsg('Downloading dataset...');
      try {
        const r = await fetch('/api/dataset/download');
        const b = await r.blob();
        const url = URL.createObjectURL(b);
        const a = document.createElement('a');
        a.href = url; a.download = 'dataset.zip'; a.click();
        URL.revokeObjectURL(url);
        setMsg('Download started');
      } catch (e) { setMsg('Download failed', true); }
    });

    video && video.addEventListener('loadeddata', resizeCanvases);
    window.addEventListener('resize', resizeCanvases);

    setMsg('Frontend ready');
    updateUI();
    loadStudentsList();
    window.loadStudentsList = loadStudentsList;

  })();
});
