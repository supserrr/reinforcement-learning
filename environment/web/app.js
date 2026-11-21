// Streams one JSON message per environment step from the Python server and
// plays it back in the 3D world (world.js), with the HUD on top: the title and
// camera tools, the agent tabs, the dock of cards, the destination card over
// the map and toasts.
import { createWorld } from "./world.js";
import { liquidGlass } from "./glass.js";

const world = createWorld(document.getElementById("scene"));
window.__world = world; // for debugging from the console
document.querySelectorAll(".card, .tray, .search, .popup-card").forEach(liquidGlass);
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
const signed = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`;
const COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
// the env stores the heading the wind blows towards; people name where it comes from
const windFrom = (rad) => COMPASS[Math.round(((((rad + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (Math.PI / 4)) % 8];
const km = (a, b) => Math.hypot(SITES[a].x - SITES[b].x, SITES[a].y - SITES[b].y);

function setBattery(b) {
  const el = $("battery");
  el.textContent = `${Math.max(0, b).toFixed(0)}%`;
  el.classList.toggle("low", b <= 50 && b > 25);
  el.classList.toggle("critical", b <= 25);
}
function setClock(t, shift) {
  $("clock").textContent = clock(t);
  const p = Math.min(Math.max(t / shift, 0), 1);
  $("pie").style.setProperty("--p", `${(p * 100).toFixed(1)}%`);
  $("pie").setAttribute("aria-label", `Shift ${Math.round(p * 100)}% done`);
  world.setTime(t);
}
function setHud(snap) {
  setClock(snap.t, snap.shift_minutes);
  setBattery(snap.battery);
  $("cargo").textContent = `${snap.cargo} of ${snap.cargo_capacity}`;
  $("wind").textContent = `${snap.wind.speed_kmh.toFixed(0)} km/h ${windFrom(snap.wind.dir_rad)}`;
  $("score").textContent = signed(snap.total_reward);
  const st = snap.stats;
  $("d-e").textContent = st.delivered_emergency;
  $("d-r").textContent = st.delivered_routine;
  $("m-e").textContent = st.missed_emergency;
  $("m-r").textContent = st.missed_routine;
  $("m-e").classList.toggle("bad", st.missed_emergency > 0);
  $("m-r").classList.toggle("bad", st.missed_routine > 0);
  $("f-km").textContent = `${Math.round(st.km_flown)} km`;
  setPosition(snap.loc);
  world.setWind(snap.wind.speed_kmh, snap.wind.dir_rad);
  world.setCargo(snap.cargo);
}
function setPosition(i) {
  $("f-pos").textContent = SITES[i].name;
  $("f-elev").textContent = `${SITES[i].elevation_m.toLocaleString("en")} m`;
  $("radar-site").textContent = SITES[i].name;
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
  while ($("log").children.length > 12) $("log").lastChild.remove();
  // one tick per decision on the return card, up for reward, down for penalty
  const tick = document.createElement("span");
  tick.className = r > 0 ? "pos" : r < 0 ? "neg" : "zero";
  if (r) tick.style.height = `${Math.min(3 + Math.abs(r) * 1.1, 13)}px`;
  $("ticks").append(tick);
  while ($("ticks").children.length > 28) $("ticks").firstChild.remove();
}
function setState(word, action) {
  $("state").textContent = word;
  $("f-action").textContent = action;
}
// the card over the destination: its request, the flight, the reward on delivery
function showPopup(to, from, snapBefore) {
  const r = snapBefore?.requests?.[to];
  const d = km(from, to);
  $("pp-name").textContent = SITES[to].name;
  $("pp-kind").textContent = "Deliver";
  $("pp-req").textContent = r ? `${r.emergency ? "Emergency" : "Routine"}, ${Math.max(0, r.minutes_left).toFixed(0)} min left` : "None open";
  $("pp-flight").textContent = `${d.toFixed(0)} km, ${Math.round(d / 2)} min`;
  $("pp-reward").textContent = r ? (r.emergency ? "+10.0" : "+4.0") : "−1.0";
  world.showPopup(to);
}

// ---------------------------------------------------------------- controls
let follow = false;
function setFollow(on) { follow = on; $("btn-follow").setAttribute("aria-pressed", String(on)); }
function toggleOrbit() { $("btn-orbit").setAttribute("aria-pressed", String(world.toggleOrbit())); }
window.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, [contenteditable]")) return;
  if (e.key === "f" || e.key === "F") setFollow(!follow);
  if (e.key === "o" || e.key === "O") toggleOrbit();
});
$("btn-follow").addEventListener("click", () => setFollow(!follow));
$("btn-orbit").addEventListener("click", toggleOrbit);
$("btn-hub").addEventListener("click", () => { setFollow(false); world.focusSite(0); });
$("btn-labels").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  document.body.classList.toggle("labels-off", !on);
});
$("overview").addEventListener("click", () => { setFollow(false); world.resetView(); });
$("zoom-in").addEventListener("click", () => world.zoomBy(0.7));
$("zoom-out").addEventListener("click", () => world.zoomBy(1.4));
$("find").addEventListener("change", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const s = SITES.find((x) => x.name.toLowerCase() === q) || SITES.find((x) => x.name.toLowerCase().includes(q));
  if (!q || !s) return;
  setFollow(false);
  world.focusSite(s.id);
  e.target.value = "";
  e.target.blur();
});

// ---------------------------------------------------------------- playback
let anim = null;
const queue = [];
let prevBattery = 100, prevT = 0, prevSnap = null;

function startStep(msg) {
  const snap = msg.snapshot;
  $("shift-no").textContent = `#${msg.episode}`;
  if (msg._reset) {
    $("f-seed").textContent = msg.seed;
    $("log").innerHTML = "";
    $("ticks").innerHTML = "";
    world.placeDrone(0);
    world.showPopup(null);
    setHud(snap);
    setRequests(snap);
    setState("At hub", "Waiting");
    prevBattery = snap.battery;
    prevT = snap.t;
    prevSnap = snap;
    anim = { msg, t0: performance.now(), dur: 900, reset: true };
    return;
  }
  const dur = Math.max(msg.anim_ms, 120) * (queue.length > 4 ? 0.3 : 1);
  const ev = snap.events || [];
  const flight = ev.find((e) => e.type === "flight");
  const crash = ev.find((e) => e.type === "crash");
  anim = { msg, t0: performance.now(), dur, curve: null, loiter: null, b0: prevBattery, b1: snap.battery, t0sim: prevT, t1sim: snap.t };
  if (flight) {
    anim.curve = world.flightCurve(flight.from, flight.to);
    world.showRoute(world.flightCurve(flight.from, flight.to), flight.to);
    if (!SITES[flight.to].hub) showPopup(flight.to, flight.from, prevSnap);
    setState("Flying", msg.action_name);
    $("f-pos").textContent = `To ${SITES[flight.to].name}`;
  } else if (crash) {
    anim.curve = world.flightCurve(crash.from, crash.to, Math.max(crash.fraction, 0.02));
    world.showRoute(world.flightCurve(crash.from, crash.to), crash.to);
    setState("Falling", msg.action_name);
  } else if (ev.some((e) => e.type === "hold")) {
    if (snap.loc !== 0) anim.loiter = world.sitePoint(snap.loc);
    setState(snap.loc !== 0 ? "Circling" : "Waiting", msg.action_name);
  } else if (ev.some((e) => e.type === "recharge")) {
    world.setCharging(performance.now() + dur);
    setState("Charging", msg.action_name);
  } else if (ev.some((e) => e.type === "load")) {
    world.loadParcel();
    setState("Loading", msg.action_name);
  } else {
    setState(snap.loc === 0 ? "At hub" : "On site", msg.action_name);
  }
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
  world.showRoute(null);
  world.showPopup(null);
  for (const e of snap.events || []) {
    if (e.type === "delivered") { world.dropParcel(e.at, e.emergency); toast(`Delivered to ${SITES[e.at].name}`, "good"); }
    if (e.type === "wasted") toast(`Wasted trip, ${e.reason}`);
    if (e.type === "crash") { world.smoke(world.drone.position.clone()); toast("Battery empty, drone down", "bad"); setState("Down", msg.action_name); }
    if (e.type === "stranded") toast("Shift ended away from the hub", "bad");
    if (e.type === "shift_end") { toast(`Shift over, return ${signed(snap.total_reward)}`, "good"); setState("Shift over", msg.action_name); }
  }
  if (!anim.curve && !anim.loiter && !(snap.events || []).some((e) => e.type === "crash")) world.placeDrone(snap.loc);
  if ($("state").textContent === "Flying") setState(snap.loc === 0 ? "At hub" : "On site", msg.action_name);
  setHud(snap);
  setRequests(snap);
  prevBattery = snap.battery;
  prevT = snap.t;
  prevSnap = snap;
  anim = null;
}

function tickStep(now, dt) {
  if (!world.isReady()) return; // the map is still loading
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

// ---------------------------------------------------------------- minimap
// the location card's thumbnail: routes from the hub, the sites, the drone's pin
const MAP_W = 220, MAP_H = 120, MAP_KM = 27;
const mx = (x) => MAP_W / 2 + (x / MAP_KM) * (MAP_H / 2) * 1.15;
const my = (y) => MAP_H / 2 - (y / MAP_KM) * (MAP_H / 2) * 1.15 + 8;
let mapPin = null;
function drawMinimap() {
  const svg = $("minimap");
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.append(e); return e; };
  el("rect", { width: MAP_W, height: MAP_H, fill: "rgba(255,255,255,0.02)" });
  for (let i = -4; i <= 4; i++) { // a faint street grid
    el("line", { x1: mx(i * 7) - 60, y1: 0, x2: mx(i * 7) + 60, y2: MAP_H, stroke: "rgba(0,0,0,0.35)", "stroke-width": 3 });
    el("line", { x1: 0, y1: my(i * 7) + 30, x2: MAP_W, y2: my(i * 7) - 30, stroke: "rgba(0,0,0,0.3)", "stroke-width": 2 });
  }
  const hub = SITES[0];
  for (const s of SITES.slice(1)) el("line", { x1: mx(hub.x), y1: my(hub.y), x2: mx(s.x), y2: my(s.y), stroke: "rgba(255,255,255,0.22)", "stroke-width": 1, "stroke-dasharray": "2 3" });
  for (const s of SITES) el("circle", { cx: mx(s.x), cy: my(s.y), r: s.hub ? 3 : 2.2, fill: "rgba(255,255,255,0.6)" });
  mapPin = document.createElementNS(NS, "g");
  mapPin.innerHTML = `<circle r="11" fill="rgba(255,255,255,0.14)" stroke="rgba(255,255,255,0.5)"/><circle r="5.5" fill="#f4f5f3"/><circle r="2.4" fill="#121414"/>`;
  svg.append(mapPin);
}
function updateMinimap() {
  if (!mapPin) return;
  const p = world.drone.position; // world x east, z south
  mapPin.setAttribute("transform", `translate(${mx(p.x).toFixed(1)} ${my(-p.z).toFixed(1)})`);
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  tickStep(now, dt);
  world.update(now, dt, follow);
  updateMinimap();
}

// ---------------------------------------------------------------- data
let built = false;
function onScene(sc) {
  if (!built) {
    SITES = sc.sites.map((s) => {
      // a map pin with a glass tag beside it
      const el = document.createElement("div");
      el.className = "site-label" + (s.hub ? " hub" : "");
      const pin = document.createElement("span");
      pin.className = "pin";
      const tag = document.createElement("div");
      tag.className = "tag";
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = s.name;
      const line = document.createElement("div");
      line.className = "s";
      tag.append(name, line);
      el.append(pin, tag);
      return { ...s, labelEl: el };
    });
    $("site-list").append(...SITES.map((s) => Object.assign(document.createElement("option"), { value: s.name })));
    world.attachPopup($("popup"));
    world.build(SITES);
    drawMinimap();
    setPosition(0);
    built = true;
  }
  const agent = sc.agent || "";
  for (const t of document.querySelectorAll(".tabs span")) {
    const on = agent.startsWith(t.dataset.agent);
    t.classList.toggle("on", on);
    if (on) t.title = sc.agent_note || agent;
  }
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
