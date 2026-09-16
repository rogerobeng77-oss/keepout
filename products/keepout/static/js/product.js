// Keepout control room. No framework, no build step.
//
// Two sources of truth, deliberately:
//   * the BAKED run (/api/demo), pre-computed at image build time, so a judge
//     opening a cold container sees a real alert with its evidence immediately;
//   * a LIVE run (POST /api/jobs + the SSE stream), which executes the same
//     pipeline on the same clip so nobody has to take the baked one on trust.
//
// The live view is the real clip playing in a <video>, with the zone, the tracks
// and the status hatch drawn over it on a <canvas>, synchronised to the
// per-frame state the pipeline returned. Nothing on the overlay is invented:
// every box comes from a frame result.

const $ = (id) => document.getElementById(id);

const ui = {
  hatch: $('hatch'), hatchLabel: $('hatch-label'), hatchDetail: $('hatch-detail'),
  hatchAck: $('hatch-ack'),
  video: $('video'), overlay: $('overlay'), stage: $('stage'),
  unusable: $('unusable'), unusableTag: $('unusable-tag'),
  unusableText: $('unusable-text'), unusableNext: $('unusable-next'),
  play: $('play'), scrub: $('scrub'), time: $('time'), readout: $('readout'),
  clipSelect: $('clip-select'), clipNote: $('clip-note'),
  runLive: $('run-live'),
  progress: $('progress'), progressBar: $('progress-bar'),
  progressMessage: $('progress-message'), log: $('log'),
  rows: $('incident-rows'), incidentSummary: $('incident-summary'),
  evidence: $('evidence'), evidenceNote: $('evidence-note'),
  counts: {
    state: $('count-state'), incidents: $('count-incidents'),
    evidence: $('count-evidence'), zone: $('count-zone'),
  },
  kpi: {
    tta: $('kpi-tta'), ttaNote: $('kpi-tta-note'),
    detect: $('kpi-detect'), detectNote: $('kpi-detect-note'),
    usable: $('kpi-usable'), speed: $('kpi-speed'),
    machine: $('kpi-machine'), machineNote: $('kpi-machine-note'),
  },
  stat: {
    opencv: $('stat-opencv'), frames: $('stat-frames'),
    msframe: $('stat-msframe'), sha: $('stat-sha'),
  },
  chipOpenCV: $('chip-opencv'), chipSource: $('chip-source'), statCost: $('stat-cost'),
  headTitle: $('head-title'), headLine: $('head-line'),
  privacyLine: $('privacy-line'), quote: $('quote'), quoteSource: $('quote-source'),
  toggleZone: $('toggle-zone'), toggleBoxes: $('toggle-boxes'), toggleBlur: $('toggle-blur'),
  zoneCanvas: $('zone-canvas'), zoneStatus: $('zone-status'),
  zonePoints: $('zone-points'), zoneArea: $('zone-area'),
  zonePropose: $('zone-propose'), zoneClear: $('zone-clear'),
  zoneDefault: $('zone-default'), zoneApply: $('zone-apply'),
  machinePoints: $('machine-points'), zoneMeta: $('zone-meta'),
  uploadInput: $('upload-input'), uploadNote: $('upload-note'),
  refTime: $('ref-time'), refTimeLabel: $('ref-time-label'), refTimeOut: $('ref-time-out'),
  assumeRunning: $('assume-running'), tiledDetection: $('tiled-detection'),
};

const state = {
  clip: null,            // current sample name
  samples: [],
  run: null,             // { incidents, live, config, summary, timing }
  frames: [],            // per-frame results, ascending by timestamp
  jobId: null,           // set when the run came from a live job
  bakedClips: [],
  zonePoints: [],        // danger zone points, in reference-frame pixels
  machinePoints: [],     // machine region points, same frame
  layer: 'danger',       // which of the two the editor is drawing
  referenceSize: [960, 540],
  referenceImage: null,
  upload: null,          // { file, url, name } when the operator brought their own clip
  workSize: null,        // the analysis resolution of the current run, for box scaling
  // Bumped whenever the operator switches clip or uploads one. A slow load that
  // finishes after the switch sees a stale token and leaves the page alone: without
  // it, the baked demo loading at boot painted itself over a clip uploaded in the
  // first second.
  loadToken: 0,
};

const WORK_MAX_SIDE = 960;  // the service's default max_side; zones are drawn at this size

const LEVEL_LABEL = {
  note: 'Note', guard: 'Guard open', alert: 'Alert', critical: 'Person down',
};
const LEVEL_DETAIL = {
  clear: 'Nobody in the danger zone.',
  unusable: 'Keepout has stopped answering rather than guess.',
  note: 'Somebody is in the zone, but the machine is stopped.',
  guard: 'The fixed guard no longer matches its reference.',
  alert: 'Somebody is in the danger zone while the machine is running.',
  critical: 'Somebody is down, or has stopped being visible inside the zone.',
};

// ---------------------------------------------------------------- helpers

const fmt = (n, digits = 0) =>
  (n === null || n === undefined || Number.isNaN(n)) ? '—' : Number(n).toFixed(digits);

const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.error?.message || `${response.status} ${response.statusText}`);
  }
  return body;
}

function appendLog(line) {
  ui.log.textContent += `${line}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

// ---------------------------------------------------------------- overlay

function frameAt(ms) {
  const frames = state.frames;
  if (!frames.length) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].timestamp_ms < ms) lo = mid + 1; else hi = mid;
  }
  // Prefer the frame at or just before the requested time.
  if (lo > 0 && frames[lo].timestamp_ms > ms) lo -= 1;
  return frames[lo];
}

function zonesFromRun() {
  const config = state.run?.config ?? {};
  return [
    ['machine', config.machine_zone, 'rgba(200,160,60,0.95)'],
    ['guard', config.guard_zone, 'rgba(111,208,140,0.95)'],
    ['danger', config.danger_zone, 'rgba(245,197,24,0.95)'],
  ].filter(([, zone]) => zone);
}

function drawOverlay(frame) {
  const canvas = ui.overlay;
  const video = ui.video;
  const width = video.videoWidth || 960;
  const height = video.videoHeight || 540;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  // Lines and labels are sized for a 960 px picture; a 1920 px upload scales them up
  // so they read the same on screen. Bundled clips are 960 or smaller: k is 1.
  const k = Math.max(1, width / 960);
  // Before a run there are no frame results, but the zones are still worth seeing
  // over the picture: that is how the operator checks them before spending a run.
  if (!frame && !zonesFromRun().length) return;

  const occupied = (frame?.occupants ?? []).length > 0;
  const level = frame?.highest_level;

  // --- zones ---------------------------------------------------------------
  const labelled = [];  // label rectangles already placed, so they cannot collide
  if (ui.toggleZone.checked) {
    for (const [kind, zone, colour] of zonesFromRun()) {
      const ref = zone.reference_size ?? [width, height];
      const sx = width / ref[0];
      const sy = height / ref[1];
      const pts = zone.points.map(([x, y]) => [x * sx, y * sy]);
      const hot = kind === 'danger' && occupied;

      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
      ctx.closePath();
      ctx.fillStyle = hot ? 'rgba(255,107,53,0.15)' : 'rgba(245,197,24,0.09)';
      if (kind !== 'danger') ctx.fillStyle = 'rgba(255,255,255,0.045)';
      ctx.fill();
      ctx.lineWidth = (kind === 'danger' ? 3 : 2) * k;
      ctx.setLineDash(kind === 'danger' ? [] : [7 * k, 5 * k]);
      ctx.strokeStyle = hot ? '#FF6B35' : colour;
      ctx.stroke();
      ctx.setLineDash([]);

      // Three zones can share a top edge, so labels are pushed down until they
      // stop overlapping and are set on a chip so they read over any frame.
      const top = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
      ctx.font = `600 ${15 * k}px "Saira Condensed", sans-serif`;
      const textW = ctx.measureText(zone.name).width;
      let lx = Math.min(Math.max(2 * k, top[0]), width - textW - 12 * k);
      let ly = Math.max(17 * k, top[1] - 7 * k);
      const overlaps = (y) => labelled.some(
        (r) => Math.abs(r.y - y) < 17 * k && lx < r.x + r.w + 8 * k
          && lx + textW + 10 * k > r.x - 8 * k);
      let guard = 0;
      while (overlaps(ly) && guard < 8) { ly += 19 * k; guard += 1; }
      labelled.push({ x: lx, y: ly, w: textW + 10 * k });

      ctx.fillStyle = 'rgba(23,25,28,0.78)';
      ctx.fillRect(lx - 4 * k, ly - 14 * k, textW + 10 * k, 18 * k);
      ctx.fillStyle = hot ? '#FF6B35' : colour;
      ctx.fillText(zone.name, lx + 1, ly);
    }
  }

  if (!frame) return;

  // --- people ---------------------------------------------------------------
  // Boxes are in analysis pixels; an uploaded 1920 px clip is analysed at 960.
  const boxScale = state.workSize ? width / state.workSize[0] : 1;
  if (ui.toggleBoxes.checked) {
    const inZone = new Set((frame.occupants ?? []).map((o) => o.track_id));
    for (const track of frame.tracks ?? []) {
      const [x1, y1, x2, y2] = track.bbox.map((v) => v * boxScale);
      const hot = inZone.has(track.track_id);
      ctx.lineWidth = (hot ? 3 : 2) * k;
      ctx.strokeStyle = hot ? '#FF6B35' : '#F2F4F5';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // A blur patch over the head, mirroring what is written into evidence.
      if (ui.toggleBlur.checked) {
        const bw = x2 - x1;
        const bh = y2 - y1;
        const headH = Math.max(bh * 0.28, bw > bh ? bw * 0.5 : bh * 0.28);
        ctx.save();
        ctx.filter = `blur(${6 * k}px)`;
        ctx.drawImage(ui.video, x1, y1, bw, headH, x1, y1, bw, headH);
        ctx.restore();
      }

      const label = hot ? `#${track.track_id} in zone` : `#${track.track_id}`;
      ctx.font = `600 ${14 * k}px "Saira Condensed", sans-serif`;
      const tw = ctx.measureText(label).width + 10 * k;
      ctx.fillStyle = hot ? '#FF6B35' : 'rgba(23,25,28,0.85)';
      ctx.fillRect(x1, Math.max(0, y1 - 20 * k), tw, 19 * k);
      ctx.fillStyle = hot ? '#17191C' : '#F2F4F5';
      ctx.fillText(label, x1 + 5 * k, Math.max(13 * k, y1 - 6 * k));
    }
  }

  // --- the hatch, drawn on the frame itself ---------------------------------
  const band = Math.max(8, Math.round(height * 0.018));
  const colour = level === 'alert' || level === 'critical' ? '#FF6B35'
    : level === 'note' ? '#7FB3D5'
      : level === 'guard' ? '#F5C518' : '#6FD08C';
  for (let x = -band * 2; x < width + band * 2; x += band * 2) {
    ctx.beginPath();
    ctx.moveTo(x, height);
    ctx.lineTo(x + band, height - band);
    ctx.lineTo(x + band * 2, height - band);
    ctx.lineTo(x + band, height);
    ctx.closePath();
    ctx.fillStyle = colour;
    ctx.fill();
  }
}

function renderFrame(ms) {
  const frame = frameAt(ms);
  drawOverlay(frame);
  ui.time.textContent = seconds(ms);
  if (!frame) return;

  const view = frame.view ?? {};
  if (view.usable === false) {
    ui.unusable.hidden = false;
    ui.unusableTag.textContent = (view.problems ?? ['view unusable'])
      .map((p) => p.replace(/_/g, ' ')).join(' and ');
    ui.unusableText.textContent = (view.messages ?? []).join(' ');
    ui.unusableNext.textContent = view.problems?.includes('camera_moved')
      ? 'Nothing below this line is being measured. Draw the zone again.'
      : 'Nothing below this line is being measured until the view comes back.';
    setHatch('unusable', 'Watching paused',
      'The view cannot be trusted, so Keepout has stopped answering rather than guess.');
  } else {
    ui.unusable.hidden = true;
    const level = frame.highest_level ?? 'clear';
    setHatch(level, level === 'clear' ? 'Zone clear' : LEVEL_LABEL[level] ?? level,
      LEVEL_DETAIL[level] ?? '');
  }

  const machine = frame.machine ?? {};
  const guard = frame.guard ?? {};
  ui.readout.replaceChildren(
    readoutItem('Machine', machine.effective_running ? 'running' : 'stopped'),
    readoutItem('Motion score', fmt(machine.score, 2)),
    readoutItem('Guard', guard.status ?? 'not watched'),
    readoutItem('People tracked', String((frame.tracks ?? []).length)),
    readoutItem('In the zone', String((frame.occupants ?? []).length)),
    readoutItem('Detector', `${fmt(frame.detect_ms, 1)} ms`),
  );
}

function readoutItem(label, value) {
  const node = el('span', 'readout__item');
  node.append(`${label} `, el('b', null, value));
  return node;
}

function setHatch(level, label, detail) {
  ui.hatch.dataset.level = level;
  ui.hatchLabel.textContent = label;
  ui.hatchDetail.textContent = detail;
  ui.counts.state.textContent = level === 'clear' ? 'clear' : level;
}

// ---------------------------------------------------------------- render

function clipUrl() {
  return state.upload ? state.upload.url : `/api/samples/${encodeURIComponent(state.clip)}/clip`;
}

function renderRun(run, { source }) {
  state.run = run;
  state.workSize = run.work_size ?? null;
  state.frames = (run.live ?? []).slice().sort((a, b) => a.timestamp_ms - b.timestamp_ms);
  ui.chipSource.textContent = source;

  const incidents = run.incidents ?? [];
  renderIncidents(incidents);
  renderEvidence(incidents);
  renderKpis(run, incidents);

  ui.video.src = clipUrl();
  ui.video.load();
  ui.scrub.value = '0';
  // Open on the worst thing this clip contains, at the moment it was raised. A
  // judge landing cold should see what the product is for, not an empty floor.
  // This is a real frame at a real timestamp, not a still chosen by hand: the
  // clip that ends in a person on the ground opens on the person on the ground.
  const first = (run.incidents ?? [])
    .slice()
    .sort((a, b) => (b.level_rank - a.level_rank) || (a.started_ms - b.started_ms))[0];
  ui.video.addEventListener('loadedmetadata', () => {
    const at = first
      ? Math.min(first.started_ms + 400, Math.max(0, (ui.video.duration - 0.2) * 1000))
      : 0;
    ui.video.currentTime = at / 1000;
    renderFrame(at);
  }, { once: true });
}

function resetKpis() {
  for (const node of [ui.kpi.tta, ui.kpi.detect, ui.kpi.usable, ui.kpi.speed, ui.kpi.machine]) {
    node.textContent = '—';
  }
  ui.kpi.ttaNote.textContent = 'not run yet';
  ui.kpi.detectNote.textContent = 'not run yet';
  ui.kpi.machineNote.textContent = 'not run yet';
  ui.stat.frames.textContent = '0';
  ui.stat.msframe.textContent = '—';
  ui.counts.incidents.textContent = '0';
  ui.readout.replaceChildren();
}

function renderKpis(run, incidents) {
  const summary = run.summary ?? {};
  const timing = run.timing ?? {};
  const truth = run.ground_truth_events ?? [];

  const entry = truth.find((e) => e.event === 'zone_entry');
  const firstAlert = incidents
    .filter((i) => i.kind === 'zone_entry')
    .sort((a, b) => a.started_ms - b.started_ms)[0];
  if (entry && firstAlert) {
    ui.kpi.tta.textContent = fmt(Math.max(0, firstAlert.started_ms - entry.ts_ms), 0);
    ui.kpi.ttaNote.textContent =
      `labelled entry at ${seconds(entry.ts_ms)}, raised at ${seconds(firstAlert.started_ms)}`;
  } else if (firstAlert) {
    ui.kpi.tta.textContent = fmt(0, 0);
    ui.kpi.ttaNote.textContent = `first raised at ${seconds(firstAlert.started_ms)}`;
  } else {
    ui.kpi.tta.textContent = '—';
    ui.kpi.ttaNote.textContent = 'nothing entered the zone';
  }

  const occupiedFrames = state.frames.filter((f) => (f.occupants ?? []).length > 0).length;
  const usableFrames = state.frames.filter((f) => f.view?.usable !== false).length;
  ui.kpi.detect.textContent = state.frames.length
    ? fmt((occupiedFrames / Math.max(1, usableFrames)) * 100, 0) : '—';
  ui.kpi.detectNote.textContent =
    `${occupiedFrames} of ${usableFrames} usable frames had somebody in the zone`;

  const usable = summary.usable_fraction;
  ui.kpi.usable.textContent = usable === undefined ? '—' : fmt(usable * 100, 0);
  ui.kpi.speed.textContent = fmt(timing.ms_per_frame, 1);
  ui.kpi.machine.textContent = summary.machine?.running ? 'running' : 'stopped';
  ui.kpi.machineNote.textContent =
    `motion score ${fmt(summary.machine?.score, 2)} against a ${fmt(
      run.config?.machine?.running_threshold, 2)} threshold`;

  ui.stat.frames.textContent = String(timing.frames ?? 0);
  ui.stat.msframe.textContent = `${fmt(timing.ms_per_frame, 1)} ms`;
  ui.counts.incidents.textContent = String((run.incidents ?? []).length);
}

function renderIncidents(incidents) {
  ui.rows.replaceChildren();
  if (!incidents.length) {
    const row = el('tr');
    const cell = el('td', 'empty');
    cell.colSpan = 6;
    cell.append(el('p', 'empty__line', 'No entries. Nobody went into the zone.'));
    cell.append(el('blockquote', 'empty__quote',
      '"In an effort to raise the alarm, he repeatedly waved at a CCTV camera in '
      + 'the hope that someone monitoring the system would spot him and come to his '
      + 'aid, nobody did."'));
    cell.append(el('p', 'empty__source',
      'HSE, 9 September 2026, Factory Services UK Limited. '
      + 'This is the only place Keepout argues for itself, and an empty list is '
      + 'where the argument belongs.'));
    row.append(cell);
    ui.rows.append(row);
    ui.incidentSummary.textContent = 'nothing raised';
    setHatch('clear', 'Zone clear', 'Nobody entered the danger zone in this clip.');
    return;
  }

  let latched = 0;
  for (const incident of incidents) {
    const row = el('tr');
    row.dataset.incident = incident.incident_id;

    const level = el('td');
    level.append(el('span', `level level--${incident.level}`, incident.level_label));
    row.append(level);

    row.append(el('td', 'when', seconds(incident.started_ms)));
    row.append(el('td', 'num', seconds(incident.duration_ms)));

    const what = el('td');
    what.append(el('div', null, incident.reason));
    const detail = Object.entries(incident.detail ?? {})
      .map(([key, value]) => {
        const label = key.replace(/_ms$/, '').replace(/_px$/, '').replace(/_/g, ' ');
        if (key.endsWith('_ms')) return `${label} ${seconds(value)}`;
        if (key.endsWith('_px')) return `${label} ${fmt(value, 0)} px`;
        return `${label} ${typeof value === 'number' ? fmt(value, 2) : value}`;
      })
      .join(' · ');
    if (detail) what.append(el('div', 'state', detail));
    row.append(what);

    row.append(el('td', null, incident.machine_running ? 'running' : 'stopped'));

    const stateCell = el('td');
    if (incident.acknowledged_ms) {
      stateCell.append(el('span', 'state',
        `acknowledged by ${incident.acknowledged_by ?? 'operator'}`));
    } else if (incident.latched) {
      latched += 1;
      stateCell.append(el('span', 'state state--held', 'held until acknowledged'));
      const button = el('button', 'ack-button', 'Acknowledge');
      button.type = 'button';
      button.addEventListener('click', () => acknowledge(incident.incident_id));
      stateCell.append(button);
    } else if (incident.open) {
      stateCell.append(el('span', 'state state--open', 'open'));
    } else {
      stateCell.append(el('span', 'state', 'cleared when they left'));
    }
    row.append(stateCell);
    ui.rows.append(row);
  }

  const worst = incidents[0];
  ui.incidentSummary.textContent =
    `${incidents.length} raised · ${latched} held for a human`;
  setHatch(worst.level, worst.level_label, worst.reason);
  ui.hatchAck.hidden = latched === 0;
  ui.hatchAck.onclick = () => {
    const held = incidents.find((i) => i.latched && !i.acknowledged_ms);
    if (held) acknowledge(held.incident_id);
  };
}

function renderEvidence(incidents) {
  const frames = [];
  for (const incident of incidents) {
    for (const item of incident.evidence ?? []) {
      frames.push({ ...item, incident });
    }
  }
  ui.counts.evidence.textContent = String(frames.length);
  ui.evidence.replaceChildren();
  if (!frames.length) {
    ui.evidence.append(el('p', 'empty',
      'No evidence frames, because nothing was raised in this clip.'));
    ui.evidenceNote.textContent = '';
    return;
  }
  ui.evidenceNote.textContent = 'every detected person\'s head blurred before the frame was written to disk';
  for (const frame of frames) {
    const figure = el('figure');
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = `${frame.incident.level_label} at ${seconds(frame.timestamp_ms)}`;
    img.src = state.jobId
      ? frame.uri
      : `/api/demo/${encodeURIComponent(state.clip)}/evidence/${frame.uri.split('/').pop()}`;
    img.addEventListener('click', () => window.open(img.src, '_blank', 'noopener'));
    figure.append(img);
    const caption = el('figcaption');
    caption.append(el('b', null,
      `${frame.incident.level_label} · ${seconds(frame.timestamp_ms)}`));
    caption.append(frame.caption || frame.incident.reason);
    if (frame.redacted) caption.append(el('span', 'badge-blur', 'faces blurred'));
    figure.append(caption);
    ui.evidence.append(figure);
  }
}

async function acknowledge(incidentId) {
  const incident = (state.run?.incidents ?? []).find((i) => i.incident_id === incidentId);
  if (!incident) return;
  if (state.jobId) {
    try {
      await getJson(
        `/api/jobs/${state.jobId}/incidents/${incidentId}/acknowledge`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ by: 'control room' }) },
      );
    } catch (error) {
      appendLog(`could not acknowledge: ${error.message}`);
    }
  }
  // The baked run is a static artefact on disk, so acknowledging it is local to
  // this browser. A live job is acknowledged on the server, above.
  incident.acknowledged_ms = Date.now();
  incident.acknowledged_by = 'control room';
  incident.open = false;
  renderIncidents(state.run.incidents);
}

// ---------------------------------------------------------------- live run

function drawnZones() {
  const zones = {};
  if (state.zonePoints.length >= 3) {
    zones.danger_zone = {
      name: 'danger zone',
      points: state.zonePoints,
      reference_size: state.referenceSize,
      kind: 'danger',
      contact: 'feet',
    };
  }
  if (state.machinePoints.length >= 3) {
    zones.machine_zone = {
      name: 'machine',
      points: state.machinePoints,
      reference_size: state.referenceSize,
      kind: 'machine',
      contact: 'centroid',
    };
  }
  return zones;
}

async function runLive() {
  if (state.upload && state.zonePoints.length < 3) {
    ui.zoneStatus.textContent =
      'Draw the danger zone on your clip first. Keepout will not guess where the floor is.';
    document.getElementById('panel-zones').scrollIntoView({ behavior: 'smooth' });
    return;
  }
  ui.runLive.disabled = true;
  ui.zoneApply.disabled = true;
  ui.progress.hidden = false;
  ui.log.textContent = '';
  ui.progressBar.style.width = '0%';
  ui.progressMessage.textContent = 'uploading the clip';

  try {
    const body = new FormData();
    if (state.upload) {
      body.append('file', state.upload.file, state.upload.name);
    } else {
      const clipResponse = await fetch(clipUrl());
      const blob = await clipResponse.blob();
      body.append('file', new File([blob], `${state.clip}.mp4`, { type: 'video/mp4' }));
    }
    const params = { ...drawnZones() };
    if (ui.assumeRunning.checked) params.assume_machine_running = true;
    if (ui.tiledDetection.checked) params.tiled_detection = true;
    body.append('params', JSON.stringify(params));

    const { job_id: jobId } = await getJson('/api/jobs', { method: 'POST', body });
    appendLog(`job ${jobId} accepted`);
    await follow(jobId);
  } catch (error) {
    ui.progressMessage.textContent = `failed: ${error.message}`;
    appendLog(`error: ${error.message}`);
  } finally {
    ui.runLive.disabled = false;
    ui.zoneApply.disabled = false;
  }
}

function follow(jobId) {
  return new Promise((resolve) => {
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    const close = () => { source.close(); resolve(); };

    source.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data);
      ui.progressBar.style.width = `${data.percent ?? 0}%`;
      ui.progressMessage.textContent = data.message ?? '';
    });
    source.addEventListener('note', (event) => {
      appendLog(JSON.parse(event.data).message ?? '');
    });
    source.addEventListener('status', async (event) => {
      const data = JSON.parse(event.data);
      if (data.status !== 'done' && data.status !== 'failed') return;
      if (data.status === 'failed') {
        ui.progressMessage.textContent = data.error?.message ?? 'the run failed';
        close();
        return;
      }
      const job = await getJson(`/api/jobs/${jobId}`);
      const record = job.result;
      state.jobId = jobId;
      ui.progressBar.style.width = '100%';
      const m = record.metrics;
      ui.progressMessage.textContent =
        `analysed ${m.frames_analysed} frames in ${m.wall_seconds} s on this instance: `
        + `${m.coverage ?? 'the whole clip'}`;
      for (const warning of record.warnings ?? []) {
        const line = el('span', 'log--warn', `warning: ${warning}\n`);
        ui.log.append(line);
      }
      renderRun({
        incidents: record.results,
        live: record.params.live,
        config: record.params.config,
        work_size: record.params.work_size,
        summary: {
          usable_fraction: record.metrics.usable_fraction,
          machine: { running: record.metrics.machine_running_at_end,
                     score: record.metrics.by_level ? undefined : undefined },
        },
        timing: {
          frames: record.metrics.frames_analysed,
          ms_per_frame: record.metrics.ms_per_frame,
        },
        ground_truth_events: state.run?.ground_truth_events ?? [],
      }, { source: 'live run on this instance' });
      close();
    });
    source.addEventListener('end', close);
  });
}

// ---------------------------------------------------------------- zones

function drawPolygon(ctx, pts, { stroke, fill, dashed, active }) {
  if (!pts.length) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
  if (pts.length >= 3) {
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.setLineDash(dashed ? [8, 6] : []);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.setLineDash([]);
  if (!active) return;
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = stroke;
    ctx.fill();
    ctx.strokeStyle = '#17191C';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawEditor() {
  const canvas = ui.zoneCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.referenceImage) ctx.drawImage(state.referenceImage, 0, 0, canvas.width, canvas.height);

  drawPolygon(ctx, state.machinePoints, {
    stroke: 'rgba(200,160,60,0.95)', fill: 'rgba(255,255,255,0.08)', dashed: true,
    active: state.layer === 'machine',
  });
  drawPolygon(ctx, state.zonePoints, {
    stroke: '#F5C518', fill: 'rgba(245,197,24,0.18)', dashed: false,
    active: state.layer === 'danger',
  });

  const pts = state.zonePoints;
  ui.zonePoints.textContent = String(pts.length);
  ui.machinePoints.textContent = String(state.machinePoints.length);
  ui.zoneArea.textContent = pts.length >= 3 ? `${Math.round(polygonArea(pts)).toLocaleString()} px` : '—';
  ui.counts.zone.textContent = pts.length >= 3 ? 'drawn' : (state.upload ? 'needed' : 'set');
  previewZones();
}

// Show what is drawn over the clip itself, in the live view, before any run: the
// operator should see the zones on the picture they are about to spend a run on.
function previewZones() {
  if (state.frames.length) return;  // a finished run owns the overlay
  const zones = drawnZones();
  const config = state.upload ? zones : { ...(state.run?.config ?? {}), ...zones };
  state.run = { ...(state.run ?? {}), config };
  drawOverlay(null);
}

function polygonArea(points) {
  let total = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    total += x1 * y2 - x2 * y1;
  }
  return Math.abs(total / 2);
}

async function loadReference(token = state.loadToken) {
  try {
    const data = await getJson(
      `/api/samples/${encodeURIComponent(state.clip)}/frame?at_ms=0`);
    if (token !== state.loadToken) return;
    state.referenceSize = [data.width, data.height];
    ui.zoneCanvas.width = data.width;
    ui.zoneCanvas.height = data.height;
    const image = new Image();
    image.onload = () => {
      if (token !== state.loadToken) return;
      state.referenceImage = image;
      drawEditor();
    };
    image.src = data.image;
    ui.zoneStatus.textContent =
      `Reference frame at 0.0 s, ${data.width} by ${data.height}. `
      + 'Click to place corners; three or more makes a zone.';
  } catch (error) {
    ui.zoneStatus.textContent = `Could not load a reference frame: ${error.message}`;
  }
}

function currentLayerPoints() {
  return state.layer === 'machine' ? state.machinePoints : state.zonePoints;
}

// Grab a frame of the uploaded clip in the browser, at the analysis size, to draw on.
function captureUploadFrame(atSeconds) {
  return new Promise((resolve, reject) => {
    const video = ui.video;
    const done = () => {
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) { reject(new Error('the browser decoded no picture')); return; }
      const scale = Math.min(1, WORK_MAX_SIDE / Math.max(w, h));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      resolve(canvas);
    };
    video.addEventListener('seeked', done, { once: true });
    video.currentTime = Math.max(0, Math.min(atSeconds, (video.duration || 0) - 0.05));
  });
}

async function setUploadReference(atSeconds) {
  const token = state.loadToken;
  try {
    const frame = await captureUploadFrame(atSeconds);
    if (token !== state.loadToken) return;
    const [oldW, oldH] = state.referenceSize;
    ui.zoneCanvas.width = frame.width;
    ui.zoneCanvas.height = frame.height;
    if (oldW !== frame.width || oldH !== frame.height) {
      state.zonePoints = [];
      state.machinePoints = [];
    }
    state.referenceSize = [frame.width, frame.height];
    state.referenceImage = frame;
    ui.refTimeOut.textContent = seconds(atSeconds * 1000);
    ui.zoneStatus.textContent =
      `Frame at ${seconds(atSeconds * 1000)}, analysed at ${frame.width} by ${frame.height}. `
      + 'Click to place the corners of the danger zone, then switch to the machine region. '
      + 'If this frame is black or blurred, pick a later one: Keepout skips frames like that '
      + 'too.';
    drawEditor();
    renderFrame(atSeconds * 1000);
  } catch (error) {
    ui.zoneStatus.textContent = `Could not read a frame from this file: ${error.message}. `
      + 'The browser has to be able to play it; MP4 (H.264) or WebM work.';
  }
}

async function loadUpload(file) {
  state.loadToken += 1;
  if (state.upload) URL.revokeObjectURL(state.upload.url);
  state.upload = { file, url: URL.createObjectURL(file), name: file.name };
  state.jobId = null;
  state.frames = [];
  state.workSize = null;
  state.zonePoints = [];
  state.machinePoints = [];
  state.run = { config: {}, incidents: [], live: [] };
  ui.clipSelect.value = '';
  ui.headTitle.textContent = file.name;
  ui.headLine.textContent =
    'Your clip. Draw the danger zone and the machine region below, check them over the '
    + 'picture, then run it on this instance.';
  ui.clipNote.textContent = '';
  ui.uploadNote.textContent = `${file.name}, ${(file.size / 1048576).toFixed(1)} MB. `
    + 'Nothing is sent until you run it.';
  ui.chipSource.textContent = 'your clip, not analysed yet';
  ui.runLive.textContent = 'Run your clip';
  ui.zoneMeta.textContent = 'Drawn on a frame of your clip, at the size it is analysed.';
  renderIncidents([]);
  renderEvidence([]);
  resetKpis();
  let option = ui.clipSelect.querySelector('option[data-upload]');
  if (!option) {
    option = document.createElement('option');
    option.dataset.upload = '1';
    option.value = '';
    ui.clipSelect.append(option);
  }
  option.textContent = `Your clip: ${file.name}`;
  ui.clipSelect.value = '';
  setHatch('clear', 'Not watched yet', 'Draw the zones, then run the clip.');
  ui.progress.hidden = true;
  ui.refTime.hidden = false;
  ui.refTimeLabel.hidden = false;
  ui.refTime.value = '0';

  ui.video.src = state.upload.url;
  ui.video.load();
  ui.video.addEventListener('loadeddata', () => setUploadReference(0), { once: true });
  ui.video.addEventListener('error', () => {
    ui.zoneStatus.textContent = 'This browser cannot play that file, so there is no frame '
      + 'to draw on. Convert it to MP4 (H.264) and try again.';
  }, { once: true });
  document.getElementById('panel-zones').scrollIntoView({ behavior: 'smooth' });
}

function useDefaultZone() {
  const zone = state.run?.config?.danger_zone;
  if (!zone) return;
  const ref = zone.reference_size ?? state.referenceSize;
  const sx = ui.zoneCanvas.width / ref[0];
  const sy = ui.zoneCanvas.height / ref[1];
  state.zonePoints = zone.points.map(([x, y]) => [x * sx, y * sy]);
  state.referenceSize = [ui.zoneCanvas.width, ui.zoneCanvas.height];
  ui.zoneStatus.textContent = 'Loaded the zone this run used. Drag-free: click to start again.';
  drawEditor();
}

async function proposeZone() {
  ui.zonePropose.disabled = true;
  ui.zoneStatus.textContent = 'Watching where the scene moves …';
  try {
    const data = await getJson('/api/zones/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample: state.clip, standoff_px: 45 }),
    });
    const proposal = data.proposal;
    if (!proposal.zone) {
      ui.zoneStatus.textContent =
        `No zone proposed: ${proposal.reason}. Draw it by hand instead.`;
      return;
    }
    const ref = proposal.zone.reference_size;
    const sx = ui.zoneCanvas.width / ref[0];
    const sy = ui.zoneCanvas.height / ref[1];
    state.zonePoints = proposal.zone.points.map(([x, y]) => [x * sx, y * sy]);
    state.referenceSize = [ui.zoneCanvas.width, ui.zoneCanvas.height];
    ui.zoneStatus.textContent =
      `${proposal.reason}. Confidence ${fmt(proposal.confidence * 100, 0)}%. ${data.advice}`;
    drawEditor();
  } catch (error) {
    ui.zoneStatus.textContent = `Proposal failed: ${error.message}`;
  } finally {
    ui.zonePropose.disabled = false;
  }
}

// ---------------------------------------------------------------- boot

function wire() {
  ui.play.addEventListener('click', () => {
    if (ui.video.paused) { ui.video.play(); ui.play.textContent = 'Pause'; }
    else { ui.video.pause(); ui.play.textContent = 'Play'; }
  });

  ui.video.addEventListener('timeupdate', () => {
    const duration = ui.video.duration || 1;
    ui.scrub.value = String(Math.round((ui.video.currentTime / duration) * 1000));
    renderFrame(ui.video.currentTime * 1000);
  });
  ui.video.addEventListener('ended', () => { ui.play.textContent = 'Play'; });

  ui.scrub.addEventListener('input', () => {
    const duration = ui.video.duration || 0;
    ui.video.currentTime = (Number(ui.scrub.value) / 1000) * duration;
  });

  for (const toggle of [ui.toggleZone, ui.toggleBoxes, ui.toggleBlur]) {
    toggle.addEventListener('change', () => renderFrame(ui.video.currentTime * 1000));
  }

  ui.runLive.addEventListener('click', runLive);

  ui.clipSelect.addEventListener('change', async () => {
    if (!ui.clipSelect.value) return;
    state.clip = ui.clipSelect.value;
    state.jobId = null;
    if (state.upload) URL.revokeObjectURL(state.upload.url);
    state.upload = null;
    ui.clipSelect.querySelector('option[data-upload]')?.remove();
    ui.runLive.textContent = 'Run this clip live';
    ui.refTime.hidden = true;
    ui.refTimeLabel.hidden = true;
    await loadClip();
  });

  ui.uploadInput.addEventListener('change', () => {
    const file = ui.uploadInput.files?.[0];
    if (file) loadUpload(file);
    ui.uploadInput.value = '';
  });

  ui.refTime.addEventListener('change', () => {
    const duration = ui.video.duration || 0;
    setUploadReference((Number(ui.refTime.value) / 1000) * duration);
  });

  for (const radio of document.querySelectorAll('input[name="zone-layer"]')) {
    radio.addEventListener('change', () => {
      state.layer = radio.value;
      ui.zoneStatus.textContent = state.layer === 'machine'
        ? 'Drawing the machine region: where the moving parts are. Motion here decides '
          + 'whether the machine is running.'
        : 'Drawing the danger zone: the floor a person must not stand on while it runs.';
      drawEditor();
    });
  }

  ui.zoneCanvas.addEventListener('click', (event) => {
    const rect = ui.zoneCanvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * ui.zoneCanvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * ui.zoneCanvas.height;
    const pts = currentLayerPoints();
    pts.push([Math.round(x), Math.round(y)]);
    state.referenceSize = [ui.zoneCanvas.width, ui.zoneCanvas.height];
    const what = state.layer === 'machine' ? 'Machine region' : 'Danger zone';
    ui.zoneStatus.textContent = pts.length < 3
      ? `${3 - pts.length} more point(s) needed for the ${what.toLowerCase()}.`
      : `${what} ready. It is shown over the clip in the live view. Save and run to use it.`;
    drawEditor();
  });

  ui.zoneClear.addEventListener('click', () => {
    if (state.layer === 'machine') state.machinePoints = []; else state.zonePoints = [];
    ui.zoneStatus.textContent = 'Cleared. Click the frame to place corners.';
    drawEditor();
  });
  ui.zoneDefault.addEventListener('click', useDefaultZone);
  ui.zonePropose.addEventListener('click', proposeZone);
  ui.zoneApply.addEventListener('click', () => {
    if (state.zonePoints.length < 3) {
      ui.zoneStatus.textContent = 'Place at least three points first.';
      return;
    }
    document.getElementById('panel-live').scrollIntoView({ behavior: 'smooth' });
    runLive();
  });

  for (const link of document.querySelectorAll('.nav')) {
    link.addEventListener('click', () => {
      for (const other of document.querySelectorAll('.nav')) other.classList.remove('is-active');
      link.classList.add('is-active');
    });
  }
}

async function loadClip() {
  const token = ++state.loadToken;
  state.workSize = null;
  state.machinePoints = [];
  const sample = state.samples.find((s) => s.name === state.clip);
  ui.clipNote.textContent = sample?.description ?? '';
  ui.headTitle.textContent = titleFor(state.clip);

  // Every bundled clip is baked at image build time, so switching clips is
  // instant. "Run this clip live" re-does the same analysis on the instance.
  if (state.bakedClips.includes(state.clip)) {
    const demo = await getJson(`/api/demo?clip=${encodeURIComponent(state.clip)}`);
    if (token !== state.loadToken) return;
    renderRun(demo, { source: 'pre-computed at image build' });
  } else {
    state.run = { config: state.run?.config ?? {}, incidents: [], live: [] };
    state.frames = [];
    renderIncidents([]);
    renderEvidence([]);
    ui.video.src = `/api/samples/${encodeURIComponent(state.clip)}/clip`;
    ui.video.load();
    ui.chipSource.textContent = 'not analysed yet';
    ui.progressMessage.textContent = 'Press "Run this clip live" to analyse this one.';
    ui.progress.hidden = false;
  }
  await loadReference(token);
  if (token !== state.loadToken) return;
  state.zonePoints = [];
  drawEditor();
}

function titleFor(name) {
  const titles = {
    'cell-alert': 'Conveyor cell, camera 1',
    'cell-stopped': 'Conveyor cell, belt stopped',
    'cell-guard': 'Conveyor cell, drive-end guard',
    'cell-down': 'Conveyor cell, worker alone',
    'cell-unusable': 'Conveyor cell, degraded view',
    'cell-quiet': 'Conveyor cell, empty room',
    courtyard: 'Walkway, real footage',
  };
  return titles[name] ?? name;
}

(async function boot() {
  wire();
  try {
    const [info, version, samples] = await Promise.all([
      getJson('/api/keepout'), getJson('/version'), getJson('/api/samples'),
    ]);

    ui.chipOpenCV.textContent = `OpenCV ${version.opencv_version}`;
    ui.stat.opencv.textContent = version.opencv_version;
    ui.stat.sha.textContent = version.git_sha || 'local';
    ui.privacyLine.textContent = info.privacy;
    if (info.instance) ui.statCost.textContent = info.instance;
    ui.quote.textContent = `"${info.quote}"`;
    ui.quoteSource.textContent = info.quote_source;

    state.samples = samples.samples ?? [];
    const baked = await getJson('/api/demo/index').catch(() => ({ clips: [], default: null }));
    state.bakedClips = (baked.clips ?? []).map((c) => c.name);
    state.bakedClip = baked.default ?? 'cell-alert';
    for (const sample of state.samples) {
      const option = document.createElement('option');
      option.value = sample.name;
      option.textContent = titleFor(sample.name);
      ui.clipSelect.append(option);
    }
    state.clip = state.samples.some((s) => s.name === state.bakedClip)
      ? state.bakedClip : state.samples[0]?.name;
    if (!state.clip) throw new Error('no sample clips are bundled in this image');
    ui.clipSelect.value = state.clip;
    await loadClip();
  } catch (error) {
    setHatch('unusable', 'Service not ready', error.message);
    console.error(error);
  }
})();
