// Streams one JSON message per environment step from the Python server and
// plays it back in the 3D world (world.js), with the shift card, the
// decision log and toasts on top.
import { createWorld } from "./world.js";

const world = createWorld(document.getElementById("scene"));
window.__world = world; // for debugging from the console
let SITES = [];

// ---------------------------------------------------------------- HUD
const $ = (id) => document.getElementById(id);
function clock(t) {
  const m = 8 * 60 + Math.round(t);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}
// kind: "good" | "bad" | "emergency" | "" (the dot always sits beside words)
function toast(text, kind = "") {
  const d = document.createElement("div");
  d.className = "toast";
  const dot = document.createElement("span");
  dot.className = `dot ${kind}`;
  const t = document.createElement("span");
  t.textContent = text;
  d.append(dot, t);
  $("toasts").appendChild(d);
  setTimeout(() => d.remove(), 2700);
  while ($("toasts").children.length > 3) $("toasts").firstChild.remove();
}
const unit = (el, value, u) => { el.textContent = value; const s = document.createElement("small"); s.textContent = u; el.append(s); };
const signed = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`;
const COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
// the env stores the heading the wind blows towards; people name where it comes from
const windFrom = (rad) => COMPASS[Math.round(((((rad + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (Math.PI / 4)) % 8];

function setBattery(b) {
  unit($("battery"), Math.max(0, b).toFixed(0), "%");
  const f = $("battery-fill");
  f.style.width = `${Math.max(0, b)}%`;
  f.style.background = b > 50 ? "var(--bar-ok)" : b > 25 ? "var(--bar-low)" : "var(--red)";
}
function setClock(t, shift) {
  $("clock").textContent = clock(t);
  $("timeline-fill").style.width = `${(100 * t) / shift}%`;
  world.setTime(t);
}
function setHud(snap) {
  setClock(snap.t, snap.shift_minutes);
  setBattery(snap.battery);
  $("cargo").textContent = `${snap.cargo} of ${snap.cargo_capacity}`;
  unit($("wind"), snap.wind.speed_kmh.toFixed(0), `km/h from ${windFrom(snap.wind.dir_rad)}`);
  $("score").textContent = signed(snap.total_reward);
  $("d-e").textContent = snap.stats.delivered_emergency;
  $("d-r").textContent = snap.stats.delivered_routine;
  $("m-e").textContent = snap.stats.missed_emergency;
  $("m-r").textContent = snap.stats.missed_routine;
  $("m-e").classList.toggle("missed", snap.stats.missed_emergency > 0);
  $("m-r").classList.toggle("missed", snap.stats.missed_routine > 0);
  world.setWind(snap.wind.speed_kmh, snap.wind.dir_rad);
  world.setCargo(snap.cargo);
}
function setRequests(snap) {
  for (const s of SITES) {
    if (s.hub) continue;
    const r = snap.requests[s.id];
    world.setRequest(s.id, r);
    const line = s.labelEl.querySelector(".s");
    line.textContent = "";
    if (r) {
      const dot = document.createElement("span");
      dot.className = `dot ${r.emergency ? "emergency" : "routine"}`;
      const t = document.createElement("span");
      t.textContent = `${r.emergency ? "Emergency" : "Routine"}, ${Math.max(0, r.minutes_left).toFixed(0)} min`;
      line.append(dot, t);
    }
  }
}
function logDecision(msg) {
  const li = document.createElement("li");
  const r = msg.reward;
  li.innerHTML = `<span class="t"></span><span class="a"></span><span class="r ${r < 0 ? "neg" : r === 0 ? "zero" : ""}"></span>`;
  li.querySelector(".t").textContent = clock(msg.snapshot.t);
  li.querySelector(".a").textContent = msg.action_name;
  li.querySelector(".r").textContent = r === 0 ? "0" : signed(r);
  $("log").prepend(li);
  while ($("log").children.length > 16) $("log").lastChild.remove();
}

// ---------------------------------------------------------------- playback
let anim = null;
const queue = [];
let prevBattery = 100, prevT = 0;
let follow = false;
window.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, [contenteditable]")) return;
  if (e.key === "f" || e.key === "F") follow = !follow;
  if (e.key === "o" || e.key === "O") world.toggleOrbit();
});

function startStep(msg) {
  const snap = msg.snapshot;
  if (msg._reset) {
    $("episode").textContent = `Shift ${msg.episode}, Seed ${msg.seed}`;
    $("log").innerHTML = "";
    world.placeDrone(0);
    setHud(snap);
    setRequests(snap);
    prevBattery = snap.battery;
    prevT = snap.t;
    anim = { msg, t0: performance.now(), dur: 900, reset: true };
    return;
  }
  const dur = Math.max(msg.anim_ms, 120) * (queue.length > 4 ? 0.3 : 1);
  const ev = snap.events || [];
  const flight = ev.find((e) => e.type === "flight");
  const crash = ev.find((e) => e.type === "crash");
  anim = { msg, t0: performance.now(), dur, curve: null, loiter: null, b0: prevBattery, b1: snap.battery, t0sim: prevT, t1sim: snap.t };
  if (flight) anim.curve = world.flightCurve(flight.from, flight.to);
  else if (crash) anim.curve = world.flightCurve(crash.from, crash.to, Math.max(crash.fraction, 0.02));
  else if (ev.some((e) => e.type === "hold") && snap.loc !== 0) anim.loiter = world.sitePoint(snap.loc);
  if (ev.some((e) => e.type === "recharge")) world.setCharging(performance.now() + dur);
  if (ev.some((e) => e.type === "load")) world.loadParcel();
  for (const e of ev) {
    if (e.type === "request" && e.emergency) toast(`Emergency request at ${SITES[e.at].name}`, "emergency");
    if (e.type === "missed") toast(`Missed ${e.emergency ? "emergency" : "routine"} request at ${SITES[e.at].name}`, "bad");
  }
  logDecision(msg);
}

function finishStep() {
  const { msg } = anim;
  if (anim.reset) { anim = null; return; }
  const snap = msg.snapshot;
  for (const e of snap.events || []) {
    if (e.type === "delivered") { world.dropParcel(e.at, e.emergency); toast(`Delivered to ${SITES[e.at].name}`, "good"); }
    if (e.type === "wasted") toast(`Wasted trip, ${e.reason}`);
    if (e.type === "crash") { world.smoke(world.drone.position.clone()); toast("Battery empty, drone down", "bad"); }
    if (e.type === "stranded") toast("Shift ended away from the hub", "bad");
    if (e.type === "shift_end") toast(`Shift over, return ${signed(snap.total_reward)}`, "good");
  }
  if (!anim.curve && !anim.loiter && !(snap.events || []).some((e) => e.type === "crash")) world.placeDrone(snap.loc);
  setHud(snap);
  setRequests(snap);
  prevBattery = snap.battery;
  prevT = snap.t;
  anim = null;
}

function tickStep(now, dt) {
  if (!anim && queue.length) startStep(queue.shift());
  if (!anim) return;
  const k = Math.min((now - anim.t0) / anim.dur, 1);
  if (!anim.reset) {
    setBattery(anim.b0 + (anim.b1 - anim.b0) * k);
    setClock(anim.t0sim + (anim.t1sim - anim.t0sim) * k, anim.msg.snapshot.shift_minutes);
  }
  if (anim.curve) {
    const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
    world.flyDrone(anim.curve.getPointAt(e), anim.curve.getPointAt(Math.min(e + 0.01, 1)), dt);
  } else if (anim.loiter) {
    const a = (now / 1000) * 1.2, L = anim.loiter;
    const p = { x: L.x + Math.cos(a) * 1.5, y: L.y + 0.6, z: L.z + Math.sin(a) * 1.5 };
    const q = { x: L.x + Math.cos(a + 0.15) * 1.5, y: L.y + 0.6, z: L.z + Math.sin(a + 0.15) * 1.5 };
    world.flyDrone(new world.drone.position.constructor(p.x, p.y, p.z), new world.drone.position.constructor(q.x, q.y, q.z), dt);
  }
  if (k >= 1) finishStep();
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  tickStep(now, dt);
  world.update(now, dt, follow);
}

// ---------------------------------------------------------------- data
let built = false;
function onScene(sc) {
  if (!built) {
    SITES = sc.sites.map((s) => {
      const el = document.createElement("div");
      el.className = "site-label" + (s.hub ? " hub" : "");
      const name = document.createElement("div");
      name.textContent = s.name;
      const line = document.createElement("div");
      line.className = "s";
      el.append(name, line);
      return { ...s, labelEl: el };
    });
    world.build(SITES);
    built = true;
  }
  $("agent").textContent = sc.agent || "Agent";
}

const es = new EventSource("/events");
let connected = false;
es.addEventListener("open", () => { if (connected === null) toast("Reconnected", "good"); connected = true; });
es.addEventListener("error", () => { if (connected) toast("Disconnected from the agent", "bad"); connected = null; });
es.addEventListener("scene", (e) => onScene(JSON.parse(e.data)));
es.addEventListener("episode_start", (e) => queue.push({ ...JSON.parse(e.data), _reset: true }));
es.addEventListener("step", (e) => queue.push(JSON.parse(e.data)));

fetch("/scene").then((r) => r.json()).then((sc) => { if (sc.sites) onScene(sc); });
requestAnimationFrame(frame);
