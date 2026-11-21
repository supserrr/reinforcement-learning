// Streams one JSON message per environment step from the Python server and
// plays it back in the 3D world (world.js), with the HUD on top: the title and
// camera tools, the agent tabs, one panel with the shift, the destination card
// over the map and toasts.
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
const signed = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`;
const COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
// the env stores the heading the wind blows towards; people name where it comes from
const windFrom = (rad) => COMPASS[Math.round(((((rad + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (Math.PI / 4)) % 8];
const km = (a, b) => Math.hypot(SITES[a].x - SITES[b].x, SITES[a].y - SITES[b].y);

function setBattery(b) {
  const el = $("battery");
  el.textContent = `${Math.max(0, b).toFixed(0)}%`;
  for (const e of [el, $("battery-bar")]) {
    e.classList.toggle("low", b <= 50 && b > 25);
    e.classList.toggle("critical", b <= 25);
  }
  $("battery-bar").firstChild.style.width = `${Math.min(Math.max(b, 0), 100)}%`;
}
function setClock(t, shift) {
  $("clock").textContent = clock(t);
  const p = Math.min(Math.max(t / shift, 0), 1);
  $("shift-bar").firstChild.style.width = `${(p * 100).toFixed(1)}%`;
  $("shift-bar").setAttribute("aria-label", `Shift ${Math.round(p * 100)}% done`);
  world.setTime(t);
}
// how much faster than real life the page is playing: flights show their own
// time-lapse, ground steps the shift minutes they cover
let warpShown = "";
function setWarp(x) {
  const t = !isFinite(x) || x <= 0 ? "" : x < 1.5 ? "real time" : `${x < 10 ? x.toFixed(1) : Math.round(x)}× real time`;
  if (t !== warpShown) { warpShown = t; $("warp").textContent = t; }
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
  $("m-e").textContent = `${st.missed_emergency} missed`;
  $("m-r").textContent = `${st.missed_routine} missed`;
  $("m-e").classList.toggle("bad", st.missed_emergency > 0);
  $("m-r").classList.toggle("bad", st.missed_routine > 0);
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
  const r = msg.reward, el = $("f-reward");
  el.textContent = r === 0 ? "" : signed(r);
  el.className = `reward num ${r < 0 ? "neg" : r === 0 ? "zero" : ""}`;
}
function setState(word, action) {
  $("state").textContent = word;
  $("f-action").textContent = action;
  $("state-dot").className = `dot ${["Down", "Falling"].includes(word) ? "bad" : ["Flying", "Launching", "Landing", "Dropping", "Caught", "Circling", "Charging", "Loading"].includes(word) ? "live" : ""}`;
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
// Follow flies the camera down behind the drone (third person); turning it off,
// or any map tool, flies it back up to the map.
let follow = false;
function showFollow(on) { follow = on; $("btn-follow").setAttribute("aria-pressed", String(on)); document.body.classList.toggle("chase", on); }
function setFollow(on) { showFollow(on); world.setFollow(on); }
function toggleOrbit() { $("btn-orbit").setAttribute("aria-pressed", String(world.toggleOrbit())); }
// playback speed: the page sets the pace and the agent waits for it
const SPEEDS = [0.5, 1, 2, 4];
let speed = 1;
function setSpeed(v) {
  speed = v;
  for (const b of document.querySelectorAll("#speed button")) b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === v));
}
window.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, [contenteditable]")) return;
  if (e.key === "f" || e.key === "F") setFollow(!follow);
  if (e.key === "o" || e.key === "O") toggleOrbit();
  const i = SPEEDS.indexOf(speed);
  if (e.key === "[" || e.key === "-") setSpeed(SPEEDS[Math.max(i - 1, 0)]);
  if (e.key === "]" || e.key === "=" || e.key === "+") setSpeed(SPEEDS[Math.min(i + 1, SPEEDS.length - 1)]);
});
for (const b of document.querySelectorAll("#speed button")) b.addEventListener("click", () => setSpeed(Number(b.dataset.speed)));
$("btn-follow").addEventListener("click", () => setFollow(!follow));
$("btn-orbit").addEventListener("click", toggleOrbit);
$("btn-hub").addEventListener("click", () => { showFollow(false); world.focusSite(0); });
$("btn-labels").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  document.body.classList.toggle("labels-off", !on);
});
$("overview").addEventListener("click", () => { showFollow(false); world.resetView(); });
$("zoom-in").addEventListener("click", () => world.zoomBy(0.7));
$("zoom-out").addEventListener("click", () => world.zoomBy(1.4));
$("find").addEventListener("change", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const s = SITES.find((x) => x.name.toLowerCase() === q) || SITES.find((x) => x.name.toLowerCase().includes(q));
  if (!q || !s) return;
  showFollow(false);
  world.focusSite(s.id);
  e.target.value = "";
  e.target.blur();
});

// ---------------------------------------------------------------- playback
let anim = null;
const queue = [];
let prevBattery = 100, prevT = 0, prevSnap = null;

// How long a ground step plays at 1x. Flights take as long as their flight
// plan (world.planLeg): take-off, the drop and the landing at 2.5x real time,
// the cruise between them compressed.
function stepDuration(msg) {
  const ev = msg.snapshot.events || [];
  if (ev.some((e) => e.type === "hold")) return msg.snapshot.loc !== 0 ? 3600 : 1500;
  if (ev.some((e) => e.type === "recharge")) return 3200;
  if (ev.some((e) => e.type === "load")) return 2200;
  return 900;
}
const catchUp = () => (queue.length > 2 ? 3 : 1); // play faster if the page fell behind
function played(msg) {
  fetch("/api/ack", { method: "POST", body: JSON.stringify({ episode: msg.episode, step: msg.step }) }).catch(() => {});
}

function startStep(msg) {
  const snap = msg.snapshot;
  $("shift-no").textContent = `#${msg.episode}`;
  if (msg._reset) {
    logDecision({ reward: 0 });
    world.placeDrone();
    world.setCharging(false);
    world.showPopup(null);
    setHud(snap);
    setRequests(snap);
    setState("At hub", "Waiting");
    prevBattery = snap.battery;
    prevT = snap.t;
    prevSnap = snap;
    anim = { msg, el: 0, dur: 900, reset: true };
    return;
  }
  if (!prevSnap) { // a reloaded page joins mid-shift: start this step's clock and battery where it began
    const took = (snap.events || []).reduce((m, e) => m + (e.minutes || 0), 0);
    prevT = Math.max(snap.t - took, 0);
    prevBattery = snap.battery;
    $("shift-no").textContent = `#${msg.episode}`;
    setHud(snap);
    setRequests(snap);
    prevSnap = snap;
  }
  const ev = snap.events || [];
  const flight = ev.find((e) => e.type === "flight");
  const crash = ev.find((e) => e.type === "crash");
  anim = { msg, el: 0, dur: stepDuration(msg), flight: null, b0: prevBattery, b1: snap.battery, t0sim: prevT, t1sim: snap.t };
  const leg = flight || crash;
  if (leg) {
    const delivered = ev.find((e) => e.type === "delivered");
    anim.flight = world.planLeg({ from: leg.from, to: leg.to, drop: !!delivered, crash: crash ? Math.max(crash.fraction, crash.from === crash.to ? 0 : 0.02) : null });
    anim.dur = anim.flight.dur;
    anim.delivered = delivered;
    if (leg.from !== leg.to) world.showRoute(anim.flight, leg.to);
    if (flight && !SITES[flight.to].hub) showPopup(flight.to, flight.from, prevSnap);
    setState(crash ? "Falling" : SITES[leg.from].hub ? "Launching" : "Flying", msg.action_name);
  } else if (ev.some((e) => e.type === "hold")) {
    setState(snap.loc !== 0 ? "Circling" : "Waiting", msg.action_name);
  } else if (ev.some((e) => e.type === "recharge")) {
    world.setCharging(true);
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
  logDecision({ reward: 0 });
}

function finishStep() {
  const { msg } = anim;
  if (anim.reset) { anim = null; return; }
  const snap = msg.snapshot;
  world.showRoute(null);
  world.showPopup(null);
  world.setCharging(false);
  const down = (snap.events || []).some((e) => e.type === "crash");
  for (const e of snap.events || []) {
    if (e.type === "delivered" && !anim.toasted) toast(`Delivered to ${SITES[e.at].name}`, "good");
    if (e.type === "wasted") toast(`Wasted trip, ${e.reason}`);
    if (e.type === "crash") { toast("Battery empty, drone down", "bad"); setState("Down", msg.action_name); }
    if (e.type === "stranded") toast("Shift ended away from the hub", "bad");
    if (e.type === "shift_end") { toast(`Shift over, return ${signed(snap.total_reward)}`, "good"); setState("Shift over", msg.action_name); }
  }
  if (!down) world.idleAt(snap.loc); // circle the centre, or back into the charging dock
  if (["Flying", "Launching", "Landing", "Dropping", "Caught"].includes($("state").textContent)) setState(snap.loc === 0 ? "At hub" : "Circling", msg.action_name);
  setHud(snap);
  setRequests(snap);
  logDecision(msg); // the reward shows when it is earned, with the return
  prevBattery = snap.battery;
  prevT = snap.t;
  prevSnap = snap;
  anim = null;
  played(msg);
}

function tickStep(now, dt) {
  if (!world.isReady()) return; // the map is still loading
  if (!anim && queue.length) startStep(queue.shift());
  if (!anim) return;
  world.setRate(speed * catchUp());
  anim.el += dt * 1000 * speed * catchUp();
  let k = Math.min(anim.el / anim.dur, 1);
  if (anim.flight) {
    const st = world.followFlight(anim.flight, anim.el);
    for (const e of st.events) {
      if (e === "launch") setState("Flying", anim.msg.action_name);
      if (e === "approach") setState(anim.flight.kind === "land" ? "Landing" : "Dropping", anim.msg.action_name);
      if (e === "drop" && anim.delivered) { toast(`Delivered to ${SITES[anim.delivered.at].name}`, "good"); anim.toasted = true; }
      if (e === "capture") setState("Caught", anim.msg.action_name);
    }
    k = Math.min(k, 1);
    setWarp(st.warp * speed * catchUp());
    if (!anim.reset) { // the shift clock and battery follow the drone along its leg
      setBattery(anim.b0 + (anim.b1 - anim.b0) * st.frac);
      setClock(anim.t0sim + (anim.t1sim - anim.t0sim) * st.frac, anim.msg.snapshot.shift_minutes);
    }
  } else if (!anim.reset) {
    setWarp(((anim.t1sim - anim.t0sim) * 60000) / anim.dur * speed * catchUp());
    setBattery(anim.b0 + (anim.b1 - anim.b0) * k);
    setClock(anim.t0sim + (anim.t1sim - anim.t0sim) * k, anim.msg.snapshot.shift_minutes);
  }
  if (k >= 1) finishStep();
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  tickStep(now, dt);
  world.update(now, dt);
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
