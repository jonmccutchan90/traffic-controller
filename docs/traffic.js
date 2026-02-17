/* ================================================================
   Adaptive Traffic Light Controller — Browser Simulation
   ================================================================
   A complete port of the Python controller to JavaScript / Canvas.
   Runs entirely client-side.  No server needed.
   ================================================================ */

"use strict";

/* ── Constants ─────────────────────────────────────────────────── */

const TIMING = {
  minGreen:        7,
  maxGreen:       60,
  yellowClearance:  4,
  allRedClearance:  2.5,
  startupLost:      2,
  minProtLeftGreen: 8,
  maxProtLeftGreen: 25,
  leftQueueThresh:  3,
  minWalk:          7,
  pedClearance:    13.7,   // 48 ft / 3.5 ft/s
  minCycle:        45,
  maxCycle:       150,
  defaultCycle:    90,
};

const FLOW = { through: 1800, leftTurn: 1600 };

const DIRS = ["N", "S", "E", "W"];

const CONFLICTS = [["N","E"],["N","W"],["S","E"],["S","W"]];

/* ── Enums ─────────────────────────────────────────────────────── */

const SIG = {
  RED:"RED", GREEN:"GREEN", YELLOW:"YELLOW",
  GREEN_ARROW:"GREEN_ARROW", YELLOW_ARROW:"YELLOW_ARROW",
  FLASHING_YELLOW:"FLASHING_YELLOW", ALL_RED:"ALL_RED", DARK:"DARK",
  WALK:"WALK", PED_CLEARANCE:"PED_CLEARANCE", DONT_WALK:"DONT_WALK",
};

const STEP = { GREEN:0, YELLOW:1, ALL_RED:2 };

/* ── Data Model ────────────────────────────────────────────────── */

function makeLane(satFlow) {
  return { queue: 0, satFlow, arrivalRate: 0 };
}

function makeApproach(dir) {
  return {
    dir,
    through: makeLane(FLOW.through),
    left:    makeLane(FLOW.leftTurn),
  };
}

function makeIntersection() {
  const app = {};
  for (const d of DIRS) app[d] = makeApproach(d);
  return app;
}

/* ── Phase Ring ────────────────────────────────────────────────── */

function makePhaseRing() {
  return [
    { id:1, dirs:["N","S"], isLeft:true,  green: TIMING.minProtLeftGreen, yellow: TIMING.yellowClearance, allRed: TIMING.allRedClearance, protLeft:false },
    { id:2, dirs:["N","S"], isLeft:false, green: TIMING.minGreen,         yellow: TIMING.yellowClearance, allRed: TIMING.allRedClearance, protLeft:false },
    { id:3, dirs:["E","W"], isLeft:true,  green: TIMING.minProtLeftGreen, yellow: TIMING.yellowClearance, allRed: TIMING.allRedClearance, protLeft:false },
    { id:4, dirs:["E","W"], isLeft:false, green: TIMING.minGreen,         yellow: TIMING.yellowClearance, allRed: TIMING.allRedClearance, protLeft:false },
  ];
}

/* ── Signal Controller ─────────────────────────────────────────── */

function makeSignalController(ring) {
  const heads = {};
  for (const d of DIRS) heads[d] = { veh: SIG.RED, lt: SIG.RED, ped: SIG.DONT_WALK };

  const ctrl = {
    ring,
    phaseIdx: 0,
    step: STEP.GREEN,
    stepStart: 0,      // sim-seconds
    cycle: 0,
    heads,
    preempted: false,
    preemptDir: null,
    faultMode: false,
  };

  applySignals(ctrl);
  return ctrl;
}

function applySignals(c) {
  // Reset all to red
  for (const d of DIRS) { c.heads[d].veh = SIG.RED; c.heads[d].lt = SIG.RED; c.heads[d].ped = SIG.DONT_WALK; }

  if (c.faultMode) { for (const d of DIRS) { c.heads[d].veh = SIG.ALL_RED; c.heads[d].lt = SIG.ALL_RED; } return; }
  if (c.step === STEP.ALL_RED) return;

  const ph = c.ring[c.phaseIdx];

  for (const dir of ph.dirs) {
    const h = c.heads[dir];
    if (c.step === STEP.GREEN) {
      if (ph.isLeft) {
        h.lt = ph.protLeft ? SIG.GREEN_ARROW : SIG.FLASHING_YELLOW;
      } else {
        h.veh = SIG.GREEN;
        h.lt  = SIG.FLASHING_YELLOW;
        h.ped = SIG.WALK;
      }
    } else if (c.step === STEP.YELLOW) {
      if (ph.isLeft) { h.lt = SIG.YELLOW_ARROW; }
      else { h.veh = SIG.YELLOW; h.ped = SIG.PED_CLEARANCE; }
    }
  }
}

function stepDuration(c) {
  if (c.preempted && c.step === STEP.GREEN) return Infinity;
  const ph = c.ring[c.phaseIdx];
  if (c.step === STEP.GREEN)   return ph.green;
  if (c.step === STEP.YELLOW)  return ph.yellow;
  if (c.step === STEP.ALL_RED) return ph.allRed;
  return 0;
}

function tickSignal(c, now) {
  if (c.faultMode) return;
  const elapsed = now - c.stepStart;
  if (elapsed >= stepDuration(c)) advanceStep(c, now);
}

function advanceStep(c, now) {
  if (c.step === STEP.GREEN) {
    c.step = STEP.YELLOW;
  } else if (c.step === STEP.YELLOW) {
    c.step = STEP.ALL_RED;
  } else if (c.step === STEP.ALL_RED) {
    if (c.preempted && c.preemptDir) {
      enterPreemptionGreen(c, now);
      return;
    }
    const prev = c.phaseIdx;
    c.phaseIdx = (c.phaseIdx + 1) % c.ring.length;
    c.step = STEP.GREEN;
    if (c.phaseIdx === 0 && prev !== 0) {
      c.cycle++;
      onCycleComplete();
    }
  }
  c.stepStart = now;
  applySignals(c);
}

function enterPreemptionGreen(c, now) {
  for (const d of DIRS) { c.heads[d].veh = SIG.RED; c.heads[d].lt = SIG.RED; c.heads[d].ped = SIG.DONT_WALK; }
  if (c.preemptDir) c.heads[c.preemptDir].veh = SIG.GREEN;
  c.step = STEP.GREEN;
  c.stepStart = now;
}

function requestPreemption(c, dir) {
  if (c.preempted) return;
  c.preempted = true;
  c.preemptDir = dir;
  if (c.step === STEP.GREEN) { c.step = STEP.YELLOW; c.stepStart = simTime; applySignals(c); }
}

function clearPreemptionOnCtrl(c) {
  c.preempted = false;
  c.preemptDir = null;
}

function checkConflicts(c) {
  const greens = new Set();
  for (const d of DIRS) {
    if (c.heads[d].veh === SIG.GREEN || c.heads[d].veh === SIG.YELLOW) greens.add(d);
    if (c.heads[d].lt === SIG.GREEN_ARROW) greens.add(d);
  }
  for (const [a,b] of CONFLICTS) {
    if (greens.has(a) && greens.has(b)) {
      c.faultMode = true;
      applySignals(c);
      return true;
    }
  }
  return false;
}

/* ── Adaptive Timing ───────────────────────────────────────────── */

let prevDS = {};

function computeAdaptivePlan(ring, intersection) {
  const demands = ring.map(ph => {
    let totalQ = 0, maxIdeal = 0, leftQ = 0;
    for (const d of ph.dirs) {
      const app = intersection[d];
      const lane = ph.isLeft ? app.left : app.through;
      totalQ += lane.queue;
      if (ph.isLeft) leftQ += lane.queue;
      const sfps = lane.satFlow / 3600;
      const ideal = lane.queue > 0 ? (lane.queue / sfps) + TIMING.startupLost : 0;
      maxIdeal = Math.max(maxIdeal, ideal);
    }
    // DS
    const refApp = intersection[ph.dirs[0]];
    const refLane = ph.isLeft ? refApp.left : refApp.through;
    const sfps = refLane.satFlow / 3600;
    let ds = ph.green > 0 ? totalQ / (ph.green * sfps) : 0;
    const prev = prevDS[ph.id] ?? ds;
    ds = 0.6 * ds + 0.4 * prev;
    prevDS[ph.id] = ds;

    return { id: ph.id, totalQ, idealGreen: maxIdeal, ds, protLeft: ph.isLeft && leftQ >= TIMING.leftQueueThresh };
  });

  // Cycle length (Webster's simplified)
  const totalLost = ring.length * (TIMING.yellowClearance + TIMING.allRedClearance);
  const avgDS = demands.reduce((s,d) => s + d.ds, 0) / demands.length;
  const y = Math.min(0.90, avgDS);
  let cycle = y < 0.05 ? TIMING.minCycle : (1.5 * totalLost + 5) / (1 - y);
  cycle = Math.max(TIMING.minCycle, Math.min(TIMING.maxCycle, cycle));

  // Green splits
  const totalFixed = ring.length * (TIMING.yellowClearance + TIMING.allRedClearance);
  const availGreen = Math.max(0, cycle - totalFixed);
  const weights = demands.map(d => {
    const ph = ring.find(p => p.id === d.id);
    if (d.totalQ === 0) return TIMING.minGreen;
    if (ph.isLeft && !d.protLeft) return TIMING.minProtLeftGreen * 0.5;
    return Math.max(TIMING.minGreen, d.idealGreen);
  });
  const totalW = weights.reduce((s,w) => s + w, 0) || 1;
  const greens = {};
  weights.forEach((w, i) => { greens[demands[i].id] = (w / totalW) * availGreen; });

  return { cycle, demands, greens };
}

function applyPlan(plan, ring) {
  for (const ph of ring) {
    if (plan.greens[ph.id] !== undefined) ph.green = plan.greens[ph.id];
    const dem = plan.demands.find(d => d.id === ph.id);
    if (dem) ph.protLeft = dem.protLeft;
    enforcePhase(ph);
  }
}

function enforcePhase(ph) {
  const minG = ph.isLeft ? TIMING.minProtLeftGreen : TIMING.minGreen;
  const maxG = ph.isLeft ? TIMING.maxProtLeftGreen : TIMING.maxGreen;
  ph.green = Math.max(minG, Math.min(maxG, ph.green));
  ph.yellow = TIMING.yellowClearance;
  ph.allRed = TIMING.allRedClearance;
  if (!ph.isLeft) {
    const minPed = TIMING.minWalk + TIMING.pedClearance;
    if (ph.green < minPed) ph.green = minPed;
  }
}

/* ── Mock Traffic Generator ────────────────────────────────────── */

const MAX_QUEUE = 25;  // Realistic cap — a long block's worth of cars

function tickMockTraffic(intersection, dt) {
  for (const d of DIRS) {
    const app = intersection[d];
    const head = signalCtrl.heads[d];

    // Time-varying arrival rate (sinusoidal rush-hour pattern)
    const timeMult = 1 + 1.0 * Math.max(0, Math.sin((simTime % 120) / 120 * Math.PI));
    const baseRate = 0.3 * timeMult * (surgeDir === d ? 3 : 1);

    // --- Through lane ---
    // Arrivals: ~0.3–0.9 cars/sec depending on time, Poisson-ish
    const tArr = Math.random() < baseRate * dt ? 1 : 0;

    // Departures: discharge at saturation flow ONLY when signal is green
    let tDep = 0;
    if (head.veh === SIG.GREEN) {
      // Saturation flow = 1800 veh/hr = 0.5 veh/sec
      tDep = Math.random() < (0.5 * dt) ? 1 : 0;
    } else if (head.veh === SIG.YELLOW) {
      // Some cars still clear during yellow
      tDep = Math.random() < (0.3 * dt) ? 1 : 0;
    }

    app.through.queue = Math.min(MAX_QUEUE, Math.max(0, app.through.queue + tArr - tDep));

    // --- Left-turn lane ---
    // Arrivals: ~15% of through rate
    const lArr = Math.random() < baseRate * 0.15 * dt ? 1 : 0;

    let lDep = 0;
    if (head.lt === SIG.GREEN_ARROW) {
      // Protected left: saturation flow = 1600 veh/hr = 0.44 veh/sec
      lDep = Math.random() < (0.44 * dt) ? 1 : 0;
    } else if (head.lt === SIG.FLASHING_YELLOW) {
      // Permissive yield: some turn when gaps appear (~30% of protected rate)
      lDep = Math.random() < (0.15 * dt) ? 1 : 0;
    } else if (head.lt === SIG.YELLOW_ARROW) {
      lDep = Math.random() < (0.2 * dt) ? 1 : 0;
    }

    app.left.queue = Math.min(MAX_QUEUE, Math.max(0, app.left.queue + lArr - lDep));
  }
}

/* ── Drawing ───────────────────────────────────────────────────── */

const canvas = document.getElementById("intersection");
const ctx = canvas.getContext("2d");

const CX = 350, CY = 350, ROAD_W = 90, HALF = 45;

const COLORS = {
  grass: "#32783c",
  road: "#3c3c3c",
  laneMarking: "#c8c864",
  stopLine: "#e0e0e0",
  crosswalk: "#e0e0e0",
  red: "#dc1e1e",
  yellow: "#f0dc28",
  green: "#1ec832",
  darkRed: "#501010",
  darkYellow: "#504b0f",
  darkGreen: "#0a410e",
  flashYellow: "#ffc800",
  orange: "#f08c1e",
  white: "#ffffff",
  darkGray: "#282828",
  gray: "#646464",
  lightGray: "#b4b4b4",
};

function sigColor(state) {
  const flash = Math.floor(simTime * 5) % 2 === 0;
  const m = {
    RED: COLORS.red, GREEN: COLORS.green, YELLOW: COLORS.yellow,
    GREEN_ARROW: COLORS.green, YELLOW_ARROW: COLORS.yellow,
    FLASHING_YELLOW: flash ? COLORS.flashYellow : COLORS.darkYellow,
    ALL_RED: COLORS.red, DARK: COLORS.darkGray,
    WALK: COLORS.white, PED_CLEARANCE: flash ? COLORS.orange : COLORS.darkGray,
    DONT_WALK: COLORS.red,
  };
  return m[state] || COLORS.gray;
}

function drawIntersection() {
  ctx.fillStyle = COLORS.grass;
  ctx.fillRect(0, 0, 700, 700);

  // Roads
  ctx.fillStyle = COLORS.road;
  ctx.fillRect(0, CY - HALF, 700, ROAD_W);
  ctx.fillRect(CX - HALF, 0, ROAD_W, 700);

  // Lane markings (dashed center lines)
  ctx.strokeStyle = COLORS.laneMarking;
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 10]);
  // Vertical
  for (let y = 0; y < 700; y += 20) {
    if (y > CY - HALF - 5 && y < CY + HALF + 5) continue;
    ctx.beginPath(); ctx.moveTo(CX, y); ctx.lineTo(CX, y + 10); ctx.stroke();
  }
  // Horizontal
  for (let x = 0; x < 700; x += 20) {
    if (x > CX - HALF - 5 && x < CX + HALF + 5) continue;
    ctx.beginPath(); ctx.moveTo(x, CY); ctx.lineTo(x + 10, CY); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Stop lines
  ctx.strokeStyle = COLORS.stopLine;
  ctx.lineWidth = 3;
  const so = HALF + 3, sl = HALF - 2;
  [[CX-sl, CY-so, CX+sl, CY-so], [CX-sl, CY+so, CX+sl, CY+so],
   [CX+so, CY-sl, CX+so, CY+sl], [CX-so, CY-sl, CX-so, CY+sl]].forEach(([x1,y1,x2,y2]) => {
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  });

  // Crosswalks
  ctx.fillStyle = COLORS.crosswalk;
  for (let off = -HALF + 5; off < HALF - 5; off += 12) {
    ctx.fillRect(CX + off - 3, CY - HALF - 15, 6, 12);
    ctx.fillRect(CX + off - 3, CY + HALF + 3, 6, 12);
    ctx.fillRect(CX + HALF + 3, CY + off - 3, 12, 6);
    ctx.fillRect(CX - HALF - 15, CY + off - 3, 12, 6);
  }
}

function drawSignalHead(x, y, dir, heads) {
  const h = heads[dir];

  // Background
  ctx.fillStyle = COLORS.darkGray;
  roundRect(x - 16, y - 36, 32, 72, 5);
  ctx.strokeStyle = COLORS.gray;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Red
  ctx.fillStyle = h.veh === SIG.RED || h.veh === SIG.ALL_RED ? COLORS.red : COLORS.darkRed;
  ctx.beginPath(); ctx.arc(x, y - 20, 10, 0, Math.PI*2); ctx.fill();
  // Yellow
  ctx.fillStyle = h.veh === SIG.YELLOW ? COLORS.yellow : COLORS.darkYellow;
  ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI*2); ctx.fill();
  // Green
  ctx.fillStyle = h.veh === SIG.GREEN ? COLORS.green : COLORS.darkGreen;
  ctx.beginPath(); ctx.arc(x, y + 20, 10, 0, Math.PI*2); ctx.fill();

  // Left-turn indicator
  ctx.fillStyle = sigColor(h.lt);
  ctx.beginPath(); ctx.arc(x + 22, y, 6, 0, Math.PI*2); ctx.fill();

  // Direction label
  ctx.fillStyle = COLORS.white;
  ctx.font = "bold 11px monospace";
  ctx.textAlign = "center";
  ctx.fillText(dir, x, y + 48);

  // Pedestrian
  ctx.fillStyle = COLORS.darkGray;
  roundRect(x - 8, y + 52, 16, 16, 3);
  ctx.fillStyle = sigColor(h.ped);
  roundRect(x - 5, y + 55, 10, 10, 2);
}

function drawSignals() {
  const pos = { N:[CX+58, CY-72], S:[CX-58, CY+72], E:[CX+72, CY+58], W:[CX-72, CY-58] };
  for (const d of DIRS) drawSignalHead(pos[d][0], pos[d][1], d, signalCtrl.heads);
}

function drawQueues() {
  const maxBar = 200, maxQ = 20;
  const cfg = {
    N: { x: CX-32, y: 30,       hor: false, lx: CX-55, ly: 12 },
    S: { x: CX+12, y: CY+120,   hor: false, lx: CX-5,  ly: CY+102 },
    E: { x: CX+120, y: CY-32,   hor: true,  lx: CX+100,ly: CY-50 },
    W: { x: 30,     y: CY+12,   hor: true,  lx: 10,    ly: CY-10 },
  };

  for (const d of DIRS) {
    const c = cfg[d], app = intersection[d];
    const tQ = app.through.queue, lQ = app.left.queue;
    const tLen = Math.min(maxBar, (tQ / maxQ) * maxBar);
    const lLen = Math.min(maxBar, (lQ / maxQ) * maxBar);
    const tCol = tQ < 5 ? COLORS.green : tQ < 12 ? COLORS.yellow : COLORS.red;
    const lCol = lQ < 3 ? COLORS.green : lQ < 6 ? COLORS.yellow : COLORS.red;

    if (c.hor) {
      ctx.fillStyle = "#1a1a1a"; roundRect(c.x, c.y, maxBar, 15, 3);
      ctx.fillStyle = tCol;       roundRect(c.x, c.y, tLen, 15, 3);
      ctx.fillStyle = "#1a1a1a"; roundRect(c.x, c.y+20, maxBar, 10, 3);
      ctx.fillStyle = lCol;       roundRect(c.x, c.y+20, lLen, 10, 3);
    } else {
      ctx.fillStyle = "#1a1a1a"; roundRect(c.x, c.y, 15, maxBar, 3);
      ctx.fillStyle = tCol;       roundRect(c.x, c.y + maxBar - tLen, 15, tLen, 3);
      ctx.fillStyle = "#1a1a1a"; roundRect(c.x+20, c.y, 10, maxBar, 3);
      ctx.fillStyle = lCol;       roundRect(c.x+20, c.y + maxBar - lLen, 10, lLen, 3);
    }

    ctx.fillStyle = COLORS.white;
    ctx.font = "11px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${d}: T=${tQ} L=${lQ}`, c.lx, c.ly);
  }
}

function drawCars() {
  // Draw small car rectangles queued on each approach
  const carW = 10, carH = 16, gap = 3;
  for (const d of DIRS) {
    const app = intersection[d];
    for (let i = 0; i < Math.min(app.through.queue, 15); i++) {
      let cx, cy;
      if (d === "N") { cx = CX + 14; cy = CY - HALF - 25 - i * (carH + gap); }
      else if (d === "S") { cx = CX - 14 - carW; cy = CY + HALF + 25 + i * (carH + gap); }
      else if (d === "E") { cx = CX + HALF + 25 + i * (carH + gap); cy = CY + 14; }
      else { cx = CX - HALF - 25 - i * (carH + gap); cy = CY - 14 - carW; }

      const w = (d === "E" || d === "W") ? carH : carW;
      const h = (d === "E" || d === "W") ? carW : carH;
      ctx.fillStyle = carColors[i % carColors.length];
      roundRect(cx, cy, w, h, 2);
    }
    // Left-turn cars (offset to inner lane)
    for (let i = 0; i < Math.min(app.left.queue, 8); i++) {
      let cx, cy;
      if (d === "N") { cx = CX - 4 - carW; cy = CY - HALF - 25 - i * (carH + gap); }
      else if (d === "S") { cx = CX + 4; cy = CY + HALF + 25 + i * (carH + gap); }
      else if (d === "E") { cx = CX + HALF + 25 + i * (carH + gap); cy = CY - 4 - carW; }
      else { cx = CX - HALF - 25 - i * (carH + gap); cy = CY + 4; }

      const w = (d === "E" || d === "W") ? carH : carW;
      const h = (d === "E" || d === "W") ? carW : carH;
      ctx.fillStyle = "#6688aa";
      roundRect(cx, cy, w, h, 2);
    }
  }
}

const carColors = ["#c0392b","#2980b9","#8e44ad","#27ae60","#f39c12","#1abc9c","#e74c3c","#3498db","#9b59b6","#2ecc71","#e67e22","#16a085","#d35400","#2c3e50","#7f8c8d"];

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}

/* ── Info Panel Updates ────────────────────────────────────────── */

function updatePanel() {
  const ph = signalCtrl.ring[signalCtrl.phaseIdx];
  const stepNames = ["GREEN","YELLOW","ALL_RED"];
  const elapsed = simTime - signalCtrl.stepStart;
  const dur = stepDuration(signalCtrl);
  const remain = dur === Infinity ? "∞" : Math.max(0, dur - elapsed).toFixed(1) + "s";

  setInfo("signal-info", [
    ["Phase", `${ph.id} (${ph.isLeft ? "LEFT" : "THRU"})`, "blue"],
    ["Step", stepNames[signalCtrl.step], signalCtrl.step===0?"green":signalCtrl.step===1?"yellow":"red"],
    ["Remaining", remain, "white"],
    ["Cycle #", signalCtrl.cycle, "blue"],
  ]);

  const totalCycle = signalCtrl.ring.reduce((s,p) => s + p.green + p.yellow + p.allRed, 0);
  const demands = lastPlan ? lastPlan.demands : [];
  const timingRows = [["Cycle Length", totalCycle.toFixed(0) + "s", "blue"]];
  for (const d of demands) {
    const dsColor = d.ds < 0.5 ? "green" : d.ds < 0.85 ? "yellow" : "red";
    timingRows.push([`P${d.id} DS`, d.ds.toFixed(2), dsColor]);
    timingRows.push([`P${d.id} Queue`, d.totalQ, "white"]);
  }
  setInfo("timing-info", timingRows);

  const qRows = [];
  let total = 0;
  for (const d of DIRS) {
    const a = intersection[d];
    qRows.push([`${d} Through`, a.through.queue, a.through.queue < 5 ? "green" : a.through.queue < 12 ? "yellow" : "red"]);
    qRows.push([`${d} Left`, a.left.queue, a.left.queue < 3 ? "green" : a.left.queue < 6 ? "yellow" : "red"]);
    total += a.through.queue + a.left.queue;
  }
  qRows.push(["Total", total, "white"]);
  setInfo("queue-info", qRows);

  const safetyRows = [];
  if (signalCtrl.faultMode) {
    safetyRows.push(["Conflict", "FAULT!", "red"]);
  } else {
    safetyRows.push(["Conflict", "OK", "green"]);
  }
  if (signalCtrl.preempted) {
    safetyRows.push(["Preemption", signalCtrl.preemptDir, "red"]);
  } else {
    safetyRows.push(["Preemption", "None", "green"]);
  }
  setInfo("safety-info", safetyRows);
}

function setInfo(elId, rows) {
  const el = document.getElementById(elId);
  el.innerHTML = rows.map(([label, value, color]) =>
    `<span class="label">${label}</span><span class="value ${color||''}">${value}</span>`
  ).join("");
}

/* ── Global State ──────────────────────────────────────────────── */

let intersection = makeIntersection();
let phaseRing    = makePhaseRing();
let signalCtrl   = makeSignalController(phaseRing);
let simTime      = 0;
let speed        = 1;
let paused       = false;
let surgeDir     = null;
let surgeEnd     = 0;
let lastPlan     = null;

/* ── Cycle-Complete Hook ───────────────────────────────────────── */

function onCycleComplete() {
  if (signalCtrl.preempted) return;
  lastPlan = computeAdaptivePlan(phaseRing, intersection);
  applyPlan(lastPlan, phaseRing);
}

/* ── Public Controls ───────────────────────────────────────────── */

function triggerPreemption(dir) {
  requestPreemption(signalCtrl, dir);
}

function clearPreemption() {
  clearPreemptionOnCtrl(signalCtrl);
}

function togglePause() {
  paused = !paused;
  document.getElementById("btn-pause").textContent = paused ? "▶ Resume" : "⏸ Pause";
}

function setSpeed(v) {
  speed = parseFloat(v);
  document.getElementById("speed-label").textContent = speed + "x";
}

function injectSurge(dir) {
  surgeDir = dir;
  surgeEnd = simTime + 10; // 10 sim-seconds of surge
}

/* ── Keyboard ──────────────────────────────────────────────────── */

document.addEventListener("keydown", e => {
  const k = e.key.toUpperCase();
  if (DIRS.includes(k)) triggerPreemption(k);
  else if (k === "C") clearPreemption();
  else if (k === " ") { e.preventDefault(); togglePause(); }
  else if (k >= "1" && k <= "5") { setSpeed(parseInt(k)); document.getElementById("speed").value = k; }
});

/* ── Main Loop ─────────────────────────────────────────────────── */

let lastFrameTime = performance.now();

// Do initial timing computation
lastPlan = computeAdaptivePlan(phaseRing, intersection);
applyPlan(lastPlan, phaseRing);

function frame(now) {
  requestAnimationFrame(frame);

  const wallDt = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  if (paused) { updatePanel(); return; }

  const simDt = Math.min(wallDt, 0.1) * speed;
  simTime += simDt;

  // Clear surge
  if (surgeDir && simTime > surgeEnd) surgeDir = null;

  // Tick subsystems
  tickMockTraffic(intersection, simDt);
  tickSignal(signalCtrl, simTime);
  checkConflicts(signalCtrl);

  // Draw
  drawIntersection();
  drawCars();
  drawSignals();
  drawQueues();
  updatePanel();
}

requestAnimationFrame(frame);
