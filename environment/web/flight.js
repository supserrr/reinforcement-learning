// Flight plans that fly like the real aircraft. From the hub: a catapult
// launch off the rail and a 10° climb to about 120 m above the ground. Then a
// terrain-following cruise that climbs over each ridge before it gets there.
// At a health centre: a long shallow descent to 35 m over the drop zone, the
// parachute release, and a climb back to the 90 m loiter. Back at the hub: a
// circuit that lines up with the recovery wire, a 4.5° final approach and
// the catch.
//
// Distances and heights are kilometres (world units); speeds are playback
// metres per second at 1x. Take-off, the drop and the landing play at 2.5x
// real time; the cruise between them is a time-lapse (up to about 40x), and
// sampleFlight reports the warp so the page can say so. app.js maps the path
// back onto the env's shift clock.
import * as THREE from "three";

export const AGL = { cruise: 0.12, drop: 0.035, loiter: 0.09, floor: 0.03 };
export const AIRSPEED = 33.3; // m/s, 120 km/h
export const SLOW = 2.5;      // playback warp around take-off, drops and landings
export const TURN_R = 0.22;   // km: a 27° banked turn at 120 km/h
const STEP = 0.02;            // km between path samples
const DEG = Math.PI / 180;
const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const H = (x, z) => new THREE.Vector3(x, 0, z);

/**
 * o.launch  { start, end } (Vector3) the catapult rail, when leaving the hub
 * o.start   { pos, heading } where the drone is now, when leaving a centre
 * o.kind    "drop" | "visit" | "land"
 * o.target  { x, z } the drop zone, the centre, or the recovery wire
 * o.axis    "land": horizontal unit vector the drone flies along into the wire
 * o.finalY  drop height, loiter height, or wire height (world y)
 * o.exitY   "drop": the loiter height to climb back to
 * o.crashAt fraction of the leg where the battery dies (0 = right here), or null
 * ground(x, z) -> world y of the terrain
 */
export function planFlight(o, ground) {
  // ---------------------------------------------------- the ground track
  // Straight legs joined by constant-radius banked turns (a Dubins path), the
  // way a fixed-wing autopilot flies: a turn onto course after the climb-out,
  // the cruise, and a turn onto a straight final through the drop zone or
  // into the wire.
  if (o.crashAt === 0) return glideDown(o.start.pos, o.start.heading, ground);
  const pts2 = [];
  const line = (a, b) => { const n = Math.max(1, Math.ceil(a.distanceTo(b) / STEP)); for (let k = pts2.length ? 1 : 0; k <= n; k++) pts2.push(a.clone().lerp(b, k / n)); };
  let railLen = 0, startY, S, Sdir;
  if (o.launch) {
    const a = H(o.launch.start.x, o.launch.start.z), b = H(o.launch.end.x, o.launch.end.z);
    Sdir = b.clone().sub(a);
    railLen = Sdir.length();
    Sdir.normalize();
    S = b.clone().addScaledVector(Sdir, 0.45);
    line(a, b);
    line(b, S);
    startY = o.launch.start.y;
  } else {
    S = H(o.start.pos.x, o.start.pos.z);
    Sdir = H(o.start.heading.x, o.start.heading.z).normalize();
    pts2.push(S.clone());
    startY = o.start.pos.y;
  }
  const T = H(o.target.x, o.target.z);
  let ax, E;
  if (o.kind === "land") {
    ax = H(o.axis.x, o.axis.z).normalize();
    E = T.clone().addScaledVector(ax, -2.0);
  } else {
    ax = T.clone().sub(S).normalize();
    E = T.clone().addScaledVector(ax, -Math.min(o.kind === "drop" ? 1.3 : 0.6, S.distanceTo(T) * 0.4));
  }
  for (const q of dubins(S, Sdir, E, ax, TURN_R).slice(1)) pts2.push(q);
  line(pts2[pts2.length - 1].clone(), T);
  const iT = pts2.length - 1;
  line(T.clone(), T.clone().addScaledVector(ax, o.kind === "land" ? 0.014 : 0.45));
  const track = pts2, N = track.length - 1, ds = STEP;
  const sT = iT * ds;
  const L = N * ds;

  // ---------------------------------------------------- the height profile
  const g = track.map((p) => ground(p.x, p.z));
  const w1 = Math.round(0.6 / ds), w2 = Math.round(0.4 / ds);
  const peak = g.map((_, i) => { let m = -Infinity; for (let k = Math.max(0, i - w1); k <= Math.min(N, i + w1); k++) m = Math.max(m, g[k]); return m + AGL.cruise; });
  const cruise = peak.map((_, i) => { let s = 0, c = 0; for (let k = Math.max(0, i - w2); k <= Math.min(N, i + w2); k++) { s += peak[k]; c++; } return Math.max(s / c, g[i] + 0.06); });
  const alt = new Array(N + 1);
  for (let i = 0; i <= N; i++) {
    const s = i * ds;
    let a = cruise[i];
    if (o.launch) {
      if (s <= railLen) a = startY + (o.launch.end.y - startY) * (s / railLen);
      else a = Math.min(a, o.launch.end.y + Math.tan(10 * DEG) * (s - railLen));
    } else {
      a = Math.min(a, startY + Math.tan(9 * DEG) * s);
      a = Math.max(a, startY - Math.tan(5 * DEG) * s);
    }
    if (o.kind === "drop") {
      if (s <= sT) a = Math.min(a, o.finalY + Math.tan(4 * DEG) * (sT - s));
      else a = Math.min(o.finalY + Math.tan(9 * DEG) * (s - sT), o.exitY);
    } else if (o.kind === "visit") {
      a = s <= sT ? Math.min(a, o.finalY + Math.tan(4 * DEG) * (sT - s)) : o.finalY;
    } else {
      a = s <= sT ? Math.min(a, o.finalY + Math.tan(4.5 * DEG) * (sT - s)) : o.finalY - 0.0016 * smooth(sT, L + 1e-6, s);
    }
    const onRail = o.launch && s < railLen + 0.06;
    const onFinal = o.kind === "land" && s > sT - 0.08; // the last 80 m are over the hub's own yard
    const overZone = o.kind === "drop" && Math.abs(s - sT) < 0.25;
    if (!onRail && !onFinal) a = Math.max(a, g[i] + (overZone ? 0.025 : AGL.floor));
    alt[i] = a;
  }
  let pts = track.map((p, i) => new THREE.Vector3(p.x, alt[i], p.z));

  // ---------------------------------------------------- a flat battery: glide until it meets the ground
  let crashed = false, sCrash = 0;
  if (o.crashAt != null) {
    const ic = Math.max(2, Math.min(N - 1, Math.round(o.crashAt * iT)));
    sCrash = ic * ds;
    const head = pts[ic].clone().sub(pts[ic - 1]).setY(0).normalize();
    pts = pts.slice(0, ic + 1).concat(glidePoints(pts[ic], head, ground));
    crashed = true;
  }

  // ---------------------------------------------------- speeds and timing
  const sEnd = (pts.length - 1) * ds;
  const vSlow = AIRSPEED * SLOW;
  const zones = [];
  if (o.launch) zones.push([0, railLen + 0.8]);
  if (crashed) zones.push([sCrash - 0.5, Infinity]);
  else if (o.kind === "drop") zones.push([sT - 1.1, Infinity]);
  else if (o.kind === "visit") zones.push([sT - 0.7, Infinity]);
  else zones.push([sT - 1.3, Infinity]);
  const outside = (s) => zones.reduce((m, [a, b]) => Math.min(m, s < a ? a - s : s > b ? s - b : 0), Infinity);
  let slowLen = 0;
  for (let i = 0; i < pts.length; i++) if (outside(i * ds) === 0) slowLen += ds;
  const cruiseLen = Math.max(sEnd - slowLen, 0);
  const vCruise = Math.max(vSlow, (cruiseLen * 1000) / Math.min(Math.max(cruiseLen * 0.9, 4), 16));
  const v = pts.map((_, i) => {
    const s = i * ds;
    if (o.launch && s <= railLen) return 5 + 45 * (s / railLen);                       // the catapult stroke
    if (!crashed && o.kind === "land" && s > sT) return Math.max(3, 30 * (1 - (s - sT) / 0.014)); // the wire pays out
    return vSlow + (vCruise - vSlow) * smooth(0, 0.5, outside(s));
  });
  const t = new Float64Array(pts.length);
  for (let i = 1; i < pts.length; i++) t[i] = t[i - 1] + (pts[i].distanceTo(pts[i - 1]) * 1000) / ((v[i] + v[i - 1]) / 2) * 1000;
  const at = (s) => { const k = Math.min(s / ds, pts.length - 1), i = Math.floor(k), f = k - i; return i >= pts.length - 1 ? t[pts.length - 1] : t[i] + (t[i + 1] - t[i]) * f; };

  // bank from the path's curvature at the real airspeed: tan(bank) = v² κ / g, rolled in and out smoothly.
  // turn > 0 is a right turn (x east, z south); a positive bank lowers the right wing.
  const bank = new Float32Array(pts.length);
  for (let i = 1; i < pts.length - 1; i++) {
    const ax1 = pts[i].x - pts[i - 1].x, az1 = pts[i].z - pts[i - 1].z, ax2 = pts[i + 1].x - pts[i].x, az2 = pts[i + 1].z - pts[i].z;
    const turn = Math.atan2(ax1 * az2 - az1 * ax2, ax1 * ax2 + az1 * az2);
    const run = Math.hypot(ax2, az2) * 1000 || 1;
    bank[i] = Math.atan((AIRSPEED * AIRSPEED * (turn / run)) / 9.81);
  }
  const bankS = bank.map((_, i) => { let sum = 0, c = 0; for (let k = Math.max(0, i - 5); k <= Math.min(pts.length - 1, i + 5); k++) { sum += bank[k]; c++; } return Math.max(-0.6, Math.min(0.6, sum / c)); });

  const events = [];
  if (o.launch) events.push({ t: 0, type: "launch" }, { t: at(railLen + 0.4), type: "climbed" });
  if (crashed) events.push({ t: t[pts.length - 1], type: "crash" });
  else if (o.kind === "drop") events.push({ t: at(sT - 0.4), type: "approach" }, { t: at(sT - 0.012), type: "drop" });
  else if (o.kind === "land") events.push({ t: at(sT - 0.45), type: "approach" }, { t: at(sT), type: "capture" });
  events.sort((a, b) => a.t - b.t);
  const f = finish(pts, t, events, o.kind, crashed);
  f.bank = bankS;
  // playback speed over real speed, at 1x: the catapult stroke and the wire's pay-out play as they happen
  f.warp = Float32Array.from(v, (x, i) => (o.launch && i * ds <= railLen) || (!crashed && o.kind === "land" && i * ds > sT) ? 1 : x / AIRSPEED);
  f.railLen = railLen;
  f.tCapture = events.find((e) => e.type === "capture")?.t ?? null;
  return f;
}

// Shortest path from pose (p0, h0) to pose (p1, h1) made of a turn, a straight
// and a turn, all turns at radius R (Dubins CSC). Points every STEP km.
function dubins(p0, h0, p1, h1, R) {
  const mod = (a) => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const dx = p1.x - p0.x, dz = p1.z - p0.z, D = Math.hypot(dx, dz), d = D / R;
  const th = Math.atan2(dz, dx), phi0 = Math.atan2(h0.z, h0.x), phi1 = Math.atan2(h1.z, h1.x);
  const a = mod(phi0 - th), b = mod(phi1 - th), sa = Math.sin(a), sb = Math.sin(b), ca = Math.cos(a), cb = Math.cos(b), cab = Math.cos(a - b);
  const opts = [];
  { const p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb); if (p2 >= 0) { const t1 = Math.atan2(cb - ca, d + sa - sb); opts.push(["LSL", mod(-a + t1), Math.sqrt(p2), mod(b - t1)]); } }
  { const p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa); if (p2 >= 0) { const t1 = Math.atan2(ca - cb, d - sa + sb); opts.push(["RSR", mod(a - t1), Math.sqrt(p2), mod(-b + t1)]); } }
  { const p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb); if (p2 >= 0) { const p = Math.sqrt(p2), t2 = Math.atan2(-ca - cb, d + sa + sb) - Math.atan2(-2, p); opts.push(["LSR", mod(-a + t2), p, mod(-mod(b) + t2)]); } }
  { const p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb); if (p2 >= 0) { const p = Math.sqrt(p2), t2 = Math.atan2(ca + cb, d - sa - sb) - Math.atan2(2, p); opts.push(["RSL", mod(a - t2), p, mod(b - t2)]); } }
  const out = [p0.clone()];
  if (!opts.length) { out.push(p1.clone()); return out; }
  const [type, t, p, q] = opts.reduce((m, o) => (o[1] + o[2] + o[3] < m[1] + m[2] + m[3] ? o : m));
  let x = p0.x, z = p0.z, phi = phi0;
  for (const [seg, len] of [[type[0], t * R], ["S", p * R], [type[2], q * R]]) {
    const n = Math.max(1, Math.ceil(len / STEP)), step = len / n, turn = seg === "L" ? 1 : seg === "R" ? -1 : 0;
    for (let k = 0; k < n; k++) {
      const mid = phi + (turn * step) / R / 2;
      x += Math.cos(mid) * step; z += Math.sin(mid) * step;
      phi += (turn * step) / R;
      out.push(new THREE.Vector3(x, 0, z));
    }
  }
  return out;
}

// the battery died where the drone is: it glides on down its heading until it hits the ground
function glideDown(pos, heading, ground) {
  const pts = [pos.clone()].concat(glidePoints(pos, H(heading.x, heading.z).normalize(), ground));
  const t = new Float64Array(pts.length);
  for (let i = 1; i < pts.length; i++) t[i] = t[i - 1] + (pts[i].distanceTo(pts[i - 1]) * 1e6) / (AIRSPEED * SLOW);
  return finish(pts, t, [{ t: t[pts.length - 1], type: "crash" }], "crash", true);
}
function glidePoints(from, head, ground) {
  const out = [];
  const p = from.clone();
  for (let k = 0; k < 400; k++) {
    p.addScaledVector(head, STEP * 0.5);
    p.y -= Math.tan((10 + k * 0.6) * DEG) * STEP * 0.5; // steepens as it stalls
    const gy = ground(p.x, p.z);
    if (p.y <= gy) { p.y = gy; out.push(p.clone()); break; }
    out.push(p.clone());
  }
  return out;
}

function finish(pts, t, events, kind, crashed) {
  return { pts, t, dur: t[t.length - 1], events, kind, crashed, lastT: -1,
    route: pts.filter((_, i) => i % 4 === 0 || i === pts.length - 1) };
}

/** Where the drone is tMs into the flight (playback ms at 1x): its position,
 * a point a little ahead for its heading, how far along the leg it is (0..1,
 * for the shift clock) and the events passed since the last call. */
export function sampleFlight(f, tMs, pos, ahead) {
  const { pts, t } = f;
  const n = pts.length - 1;
  const tc = Math.min(Math.max(tMs, 0), t[n]);
  let lo = 0, hi = n;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= tc) lo = m; else hi = m; }
  const k = t[hi] > t[lo] ? (tc - t[lo]) / (t[hi] - t[lo]) : 0;
  pos.lerpVectors(pts[lo], pts[hi], k);
  const j = Math.min(lo + 3, n);
  if (j > lo + 1) ahead.copy(pts[j]); else ahead.copy(pos).add(pts[n].clone().sub(pts[Math.max(n - 3, 0)]));
  const events = f.events.filter((e) => e.t > f.lastT && e.t <= tc).map((e) => e.type);
  f.lastT = tc;
  const bank = f.bank ? f.bank[lo] + (f.bank[hi] - f.bank[lo]) * k : 0;
  const warp = f.warp ? f.warp[lo] + (f.warp[hi] - f.warp[lo]) * k : SLOW;
  return { frac: (lo + k) / n, events, done: tc >= t[n], bank, warp, onRail: f.railLen > 0 && lo * STEP < f.railLen, railK: f.railLen > 0 ? Math.min((lo + k) * STEP / f.railLen, 1) : 1 };
}
