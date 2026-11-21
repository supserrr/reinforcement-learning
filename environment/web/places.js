// The built world at real scale. The drone hub has a catapult launcher, a
// recovery wire between two masts, a hangar with a covered charging dock,
// battery and supply containers, a solar array and a windsock. Each health
// centre has a walled compound, a marked drop zone and a windsock. Houses and
// trees stream in around the camera: villages along the ridges, the town
// around the hub, banana groves by the houses and eucalyptus woodlots on the
// hills. Everything is modelled in metres and scaled to the map's kilometres.
import * as THREE from "three";
import { mergeGeometries, mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";
import { loadGLB } from "./glb.js";

// The models built in Blender (scripts/blender/build_assets.py). Any that fail
// to load fall back to the simpler shapes built in code below.
export async function loadModels(base = "models/") {
  const names = ["drone", "hub", "clinic", "houses", "trees"];
  const out = await Promise.all(names.map((n) => loadGLB(`${base}${n}.glb`).catch((e) => { console.warn(`model ${n}:`, e.message); return null; })));
  return Object.fromEntries(names.map((n, i) => [n, out[i]]));
}

const M = 0.001; // metres to km
const UP = new THREE.Vector3(0, 1, 0);
const mat = (color, roughness = 0.85, metalness = 0, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });

// window glass: dark, smooth and clear-coated, so it reflects the sky
const windowGlass = () => new THREE.MeshPhysicalMaterial({ color: "#1b252c", roughness: 0.06, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.6 });

// a gable roof, ridge along z: w across, h tall, d long
function gable(w, h, d) {
  const s = new THREE.Shape();
  s.moveTo(-w / 2, 0); s.lineTo(w / 2, 0); s.lineTo(0, h); s.closePath();
  return new THREE.ExtrudeGeometry(s, { depth: d, bevelEnabled: false }).translate(0, 0, -d / 2);
}

// collects transformed parts per material and merges them into one mesh each
function kit() {
  const lists = new Map();
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _e = new THREE.Euler(), _p = new THREE.Vector3(), ONE = new THREE.Vector3(1, 1, 1);
  function put(material, geo, x, y, z, ry = 0, rx = 0, rz = 0) {
    _e.set(rx, ry, rz, "YXZ");
    geo.applyMatrix4(_m.compose(_p.set(x, y, z), _q.setFromEuler(_e), ONE));
    if (!lists.has(material)) lists.set(material, []);
    lists.get(material).push(geo.index ? geo.toNonIndexed() : geo);
  }
  return {
    put,
    box: (m, w, h, d, x, y, z, ry = 0, rx = 0, rz = 0) => put(m, new THREE.BoxGeometry(w, h, d), x, y, z, ry, rx, rz),
    cyl: (m, r, h, x, y, z, seg = 10, rx = 0, rz = 0, ry = 0) => put(m, new THREE.CylinderGeometry(r, r, h, seg), x, y, z, ry, rx, rz),
    roof: (m, w, h, d, x, y, z, ry = 0) => put(m, gable(w, h, d), x, y, z, ry),
    // a building: walls, a gable roof with eaves and a dark band of windows on the long faces
    house(walls, roofM, glass, w, h, d, x, z, ry = 0, rh = 2) {
      put(walls, new THREE.BoxGeometry(w, h, d), x, h / 2, z, ry);
      const along = w >= d;
      put(roofM, gable((along ? d : w) + 1.2, rh, (along ? w : d) + 1.2), x, h, z, ry + (along ? Math.PI / 2 : 0));
      const c = Math.cos(ry), s = Math.sin(ry);
      for (const side of [1, -1]) {
        const off = (along ? d : w) / 2 + 0.03;
        const lx = along ? 0 : side * off, lz = along ? side * off : 0;
        put(glass, new THREE.BoxGeometry(along ? w * 0.8 : 0.06, 1.1, along ? 0.06 : d * 0.8), x + lx * c + lz * s, h * 0.55, z - lx * s + lz * c, ry);
      }
    },
    build() {
      const g = new THREE.Group();
      for (const [m, list] of lists) {
        const mesh = new THREE.Mesh(mergeGeometries(list), m);
        mesh.castShadow = !m.transparent;
        mesh.receiveShadow = true;
        g.add(mesh);
      }
      return g;
    },
  };
}

function place(group, x, y, z, yaw) {
  group.scale.setScalar(M);
  group.position.set(x, y, z);
  group.rotation.y = yaw;
  const origin = new THREE.Vector3(x, y, z);
  return (lx, ly, lz) => new THREE.Vector3(lx, ly, lz).applyAxisAngle(UP, yaw).multiplyScalar(M).add(origin);
}

function windsock(k, x, z) {
  k.cyl(mat("#c9ccd0", 0.5, 0.6), 0.07, 6, x, 3, z, 8);
  const sock = new THREE.Group();
  sock.position.set(x, 5.8, z);
  const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.18, 2.6, 12, 1, true).rotateZ(Math.PI / 2).translate(1.4, 0, 0),
    new THREE.MeshStandardMaterial({ color: "#e0662a", roughness: 0.8, side: THREE.DoubleSide }));
  cone.castShadow = true;
  sock.add(cone);
  return sock;
}

// ------------------------------------------------------------------ the hub
// Local axes: +z is the launch direction and the side the landings come in
// from (the open ground the site was chosen for); +x is to the right.
export function buildHub({ x, y, z, yaw }, model = null) {
  if (model) return hubFromModel(model, { x, y, z, yaw });
  const k = kit();
  const concrete = mat("#5a5753", 0.97, 0, { envMapIntensity: 0.25 }), slab = mat("#8a8883", 0.95), paint = mat("#e9e6df", 0.7), walls = mat("#d8d5cc"), roofM = mat("#5d646c", 0.55, 0.35);
  const glass = windowGlass(), steel = mat("#b8bcc1", 0.45, 0.65), dark = mat("#2b2f33", 0.6, 0.4);
  const yellow = mat("#d8a81f", 0.6), white = mat("#eceae4"), orange = mat("#cc6430", 0.8), panel = mat("#1a2637", 0.25, 0.45);
  const fence = mat("#3a3f42", 0.7, 0.3, { transparent: true, opacity: 0.45 });

  k.box(concrete, 120, 0.4, 100, 0, 0.2, 0);
  k.box(slab, 14, 0.04, 10, -30, 0.42, -16);   // concrete under the charging dock
  k.box(slab, 8, 0.04, 22, 26, 0.42, -1);      // and along the launcher
  for (const [bx, bz] of [[-30, 14], [-30, 22]]) k.box(paint, 22, 0.02, 0.4, bx, 0.41, bz); // the recovery lane markings
  // hangar, offices, containers
  k.house(walls, roofM, glass, 34, 9, 18, -30, -38, 0, 3);
  k.box(dark, 15, 7, 0.2, -30, 3.5, -28.9);
  k.house(walls, roofM, glass, 18, 5.5, 10, 6, -40, 0, 1.8);
  k.box(white, 12.2, 2.6, 2.44, 32, 1.7, -38);
  k.box(orange, 12.2, 2.6, 2.44, 32, 1.7, -34.5);
  k.box(white, 6, 2.6, 2.44, 32, 1.7, -31);
  // charging dock: a cradle on a plinth, a charger cabinet with its status light, and the cable
  k.box(dark, 5, 0.6, 3.6, -30, 0.7, -16);
  k.box(steel, 0.3, 0.5, 2.6, -31.6, 1.25, -16);
  k.box(steel, 0.3, 0.5, 2.6, -28.4, 1.25, -16);
  k.box(paint, 1.3, 2.1, 0.8, -25.5, 1.45, -19.5);
  k.cyl(dark, 0.05, 4.2, -27.6, 0.5, -18, 6, 0, Math.PI / 2, 0.6);
  const chargeLight = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.5, 0.05), new THREE.MeshBasicMaterial({ color: "#1d3326" }));
  chargeLight.position.set(-25.5, 1.9, -19.08);
  const dockStrip = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.08, 0.08), new THREE.MeshBasicMaterial({ color: "#1d3326" }));
  dockStrip.position.set(-30, 1.02, -14.15);
  // catapult launcher: a 15 m rail rising 12° toward +z on two trestles
  const LX = 26, rail0 = new THREE.Vector3(LX, 1.4, -8), up12 = 12 * Math.PI / 180, RL = 15;
  const rail1 = new THREE.Vector3(LX, rail0.y + Math.sin(up12) * RL, rail0.z + Math.cos(up12) * RL);
  k.box(dark, 4, 1, 5, LX, 0.9, -9);
  k.box(steel, 0.55, 0.35, RL, LX, (rail0.y + rail1.y) / 2, (rail0.z + rail1.z) / 2, 0, -up12);
  k.box(yellow, 0.9, 0.5, 0.5, LX, rail1.y + 0.1, rail1.z);
  const carriage = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.3, 1.1), mat("#2b2f33", 0.5, 0.5));
  carriage.rotation.x = -up12;
  carriage.castShadow = true;
  const rail0L = rail0.clone().setY(rail0.y + 0.3), rail1L = rail1.clone().setY(rail1.y + 0.3);
  carriage.position.copy(rail0L);
  for (const t of [0.45, 0.9]) { const h = rail0.y + Math.sin(up12) * RL * t; k.box(steel, 2.6, h, 0.25, LX, h / 2 + 0.4, rail0.z + Math.cos(up12) * RL * t); }
  // recovery: two 15 m masts and the wire between them, 12 m up
  const WZ = 24, WX = -32;
  for (const mx of [WX - 12, WX + 12]) { k.cyl(steel, 0.3, 15, mx, 7.9, WZ, 12); k.box(yellow, 0.9, 0.6, 0.9, mx, 15.6, WZ); }

  // solar array behind the buildings
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) k.box(panel, 9.6, 0.12, 3.4, -8 + c * 10.2, 1.4, -56 - r * 6, 0, -0.26);
  // perimeter fence
  for (const [fx, fz, fw, fd] of [[0, -62, 124, 0.08], [0, 52, 124, 0.08], [-62, -5, 0.08, 114], [62, -5, 0.08, 114]]) k.box(fence, fw, 2.2, fd, fx, 1.5, fz);
  const sock = windsock(k, 52, 42);

  const group = k.build();
  const wireObj = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 24, 6).rotateZ(Math.PI / 2), dark);
  wireObj.position.set(WX, 12, WZ);
  group.add(sock, chargeLight, dockStrip, carriage, wireObj);
  const W = place(group, x, y, z, yaw);
  const fwd = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  return {
    group, sock, chargeLight, dockStrip, carriage, rail0L, rail1L, wireObj,
    dock: W(-30, 1.0 + 0.22, -16),                     // the drone's belly on the dock cradle
    cradle: W(LX, rail0.y + 0.45, rail0.z),            // on the rail, ready to launch
    railEnd: W(LX, rail1.y + 0.45, rail1.z),
    launchPitch: up12,
    wire: W(WX, 11.4, WZ),                             // where the hook meets the wire
    axis: fwd.clone().negate(),                        // landings fly toward -z, into the wire
    heading: fwd,
    padY: y + 0.4 * M,
    cams: { launch: W(LX + 16, 2.6, -1), land: W(WX + 30, 2.2, WZ - 20) }, // beside and past the wire, facing the approach
    mastA: W(WX - 12, 12, WZ), mastB: W(WX + 12, 12, WZ),
    supply: W(32, 2, -33),
  };
}

// the hub as built in Blender: same layout and the same named parts
function hubFromModel(model, { x, y, z, yaw }) {
  const group = model.scene;
  const lamp = (name) => {
    const o = model.get(name);
    const m = o.isMesh ? o : o.children[0];
    m.material = new THREE.MeshBasicMaterial({ color: "#1d3326" }); // world.js lights it while charging
    return m;
  };
  const chargeLight = lamp("charge_light"), dockStrip = lamp("dock_strip");
  const carriage = model.get("carriage"), sock = model.get("windsock"), wireObj = model.get("wire");
  for (const o of [sock]) o.rotation.set(0, 0, 0);
  group.traverse((o) => { if (o.isMesh && /^(alpha|leaf)_/.test(o.material.name)) o.material = keepCoverage(o.material); });
  const W = place(group, x, y, z, yaw);
  const LX = 26, up12 = 12 * Math.PI / 180, RL = 15, r0 = new THREE.Vector3(LX, 1.4, -8);
  const r1 = new THREE.Vector3(LX, r0.y + Math.sin(up12) * RL, r0.z + Math.cos(up12) * RL);
  const WX = -32, WZ = 24;
  const fwd = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  return {
    group, sock, chargeLight, dockStrip, carriage, wireObj,
    rail0L: carriage.position.clone(), rail1L: carriage.position.clone().add(r1.clone().sub(r0)),
    dock: W(-30, 1.72, -16),                           // belly on the dock's rubber pads
    cradle: W(LX, r0.y + 0.66, r0.z),                  // on the carriage, ready to launch
    railEnd: W(LX, r1.y + 0.66, r1.z),
    launchPitch: up12,
    wire: W(WX, 11.4, WZ),
    axis: fwd.clone().negate(),
    heading: fwd,
    padY: y + 0.4 * M,
    cams: { launch: W(LX + 16, 2.6, -1), land: W(WX + 30, 2.2, WZ - 20) }, // beside and past the wire, facing the approach
    mastA: W(WX - 12, 12, WZ), mastB: W(WX + 12, 12, WZ),
    supply: W(32, 2, -33),
  };
}

// ------------------------------------------------------------------ a health centre
// Local axes: +z points toward the hub; the drop zone lies on that side.
export function buildClinic({ x, y, z, yaw, seed }, model = null) {
  if (model) {
    // each centre gets its own copy, with its own wall and roof colours
    const group = model.scene.clone(true);
    const r = (n) => { const v = Math.sin((seed + 1) * 12.9898 + n * 78.233) * 43758.5453; return v - Math.floor(v); };
    const tints = { clinic_roof: ["#d6dade", "#c97a5c", "#b9c2c8", "#d08a62"][Math.floor(r(2) * 4)], clinic_walls: ["#ffffff", "#f6ead6", "#fff4e4"][Math.floor(r(1) * 3)] };
    const done = new Map();
    group.traverse((o) => {
      if (!o.isMesh || !tints[o.material.name]) return;
      if (!done.has(o.material.name)) { const m = o.material.clone(); m.color.set(tints[o.material.name]); done.set(o.material.name, m); }
      o.material = done.get(o.material.name);
    });
    group.traverse((o) => {
      if (o.isMesh && /^(alpha|leaf)_/.test(o.material.name) && !o.material.userData.coverage) { o.material.userData.coverage = true; keepCoverage(o.material); }
    });
    const sock = group.getObjectByName("windsock");
    const W = place(group, x, y, z, yaw);
    return { group, sock, drop: W(0, 0, 46), cams: { drop: W(-34, 1.8, 28) } /* outside the compound wall, beside the drop zone */ };
  }
  const k = kit();
  const r = (n) => { const v = Math.sin((seed + 1) * 12.9898 + n * 78.233) * 43758.5453; return v - Math.floor(v); };
  const walls = mat(["#e8dfcd", "#ded5c2", "#efe9dc"][Math.floor(r(1) * 3)]);
  const roofM = mat(["#6e777f", "#8c4a34", "#7a8288", "#9a5a3d"][Math.floor(r(2) * 4)], 0.6, 0.3);
  const glass = windowGlass(), yard = mat("#a69a86", 0.95), wallM = mat("#cbbfa8"), white = mat("#f2f0ea", 0.6);
  const steel = mat("#9ea4a9", 0.5, 0.5), blue = mat("#3d5d7a", 0.6);

  k.box(yard, 64, 0.12, 44, 0, 0.06, -10);
  for (const [wx, wz, ww, wd] of [[0, -32, 64, 0.3], [-32, -10, 0.3, 44], [32, -10, 0.3, 44], [-18, 12, 28, 0.3], [18, 12, 28, 0.3]]) k.box(wallM, ww, 1.8, wd, wx, 0.9, wz);
  k.house(walls, roofM, glass, 30, 4.2, 10, -4, -22, 0, 2.2);
  k.house(walls, roofM, glass, 10, 4.2, 16, 18, -12, 0, 2);
  k.house(walls, roofM, glass, 16, 3.8, 8, -14, 2, 0, 1.8);
  k.house(walls, roofM, glass, 6, 3, 6, 22, 4, 0, 1.4);
  for (const [px, pz] of [[24, -2], [27, -2], [24, 1], [27, 1]]) k.cyl(steel, 0.08, 4, px, 2, pz, 6);
  k.cyl(blue, 1.6, 2.6, 25.5, 5.3, -0.5, 16);
  // the drop zone: a white ring and cross on cut grass, 46 m out toward the hub
  const ring = new THREE.RingGeometry(8.4, 9.2, 48).rotateX(-Math.PI / 2);
  k.put(white, ring, 0, 0.08, 46);
  k.box(white, 10, 0.04, 0.7, 0, 0.08, 46);
  k.box(white, 0.7, 0.04, 10, 0, 0.08, 46);
  const sock = windsock(k, 13, 46);

  const group = k.build();
  group.add(sock);
  for (const m of group.children) if (m.material === white) { m.material.polygonOffset = true; m.material.polygonOffsetFactor = -2; }
  const W = place(group, x, y, z, yaw);
  return { group, sock, drop: W(0, 0, 46), cams: { drop: W(-34, 1.8, 28) } /* outside the compound wall, beside the drop zone */ };
}

// ------------------------------------------------------------------ houses and trees
function hash(ix, iz, s = 0) {
  let h = Math.imul(ix | 0, 374761393) ^ Math.imul(iz | 0, 668265263) ^ Math.imul(s | 0, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x, z, s) {
  const ix = Math.floor(x), iz = Math.floor(z), fx = x - ix, fz = z - iz;
  const u = fx * fx * (3 - 2 * fx), v = fz * fz * (3 - 2 * fz);
  const a = hash(ix, iz, s), b = hash(ix + 1, iz, s), c = hash(ix, iz + 1, s), d = hash(ix + 1, iz + 1, s);
  return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v;
}
const fbm = (x, z, s) => vnoise(x, z, s) * 0.55 + vnoise(x * 2.1, z * 2.1, s + 1) * 0.3 + vnoise(x * 4.3, z * 4.3, s + 2) * 0.15;
function mulberry(seed) { return () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const sstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };


// ------------------------------------------------------------------ tree models
// A crown is a few lumpy lobes with a darker underside and core (cheap ambient
// occlusion) and blotchy leaf-clump shading baked into its vertex colours.
function crownGeometry(lobes, detail, seed) {
  const r = mulberry(seed), parts = [];
  for (let i = 0; i < lobes; i++) {
    const g = new THREE.IcosahedronGeometry(1, detail);
    const a = (i / lobes) * Math.PI * 2 + r() * 0.8, d = i === 0 ? 0 : 0.4 + r() * 0.2, sc = i === 0 ? 0.82 : 0.55 + r() * 0.2;
    g.scale(sc, sc * 0.85, sc).translate(Math.cos(a) * d, i === 0 ? 0.12 : -0.12 + r() * 0.32, Math.sin(a) * d);
    g.deleteAttribute("uv");
    parts.push(g);
  }
  const g = mergeVertices(mergeGeometries(parts), 1e-4);
  const pos = g.attributes.position, col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const k = 1 + (vnoise(x * 2.6 + 7, z * 2.6 + y * 1.9, 11) - 0.5) * 0.34; // lumpy, not a ball
    pos.setXYZ(i, x * k, y * k, z * k);
    const ao = 0.32 + 0.68 * sstep(-0.95, 0.75, y) * (0.7 + 0.3 * Math.min(Math.hypot(x, y, z), 1));
    const clump = 0.78 + 0.4 * vnoise(x * 7.3, z * 7.3 + y * 5.1, 13);
    col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = ao * clump;
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();
  return g;
}
// a banana plant: a pseudo-stem and seven long blades that arch and droop
function bananaGeometry() {
  const parts = [], stemCol = 0.55;
  const stem = new THREE.CylinderGeometry(0.11, 0.17, 1.7, 7).translate(0, 0.85, 0).toNonIndexed();
  stem.deleteAttribute("uv");
  stem.setAttribute("color", new THREE.BufferAttribute(new Float32Array(stem.attributes.position.count * 3).fill(stemCol), 3));
  parts.push(stem);
  const r = mulberry(21);
  for (let i = 0; i < 7; i++) {
    const leaf = new THREE.PlaneGeometry(0.55, 2.3, 1, 8).toNonIndexed();
    leaf.deleteAttribute("uv");
    const pos = leaf.attributes.position, col = new Float32Array(pos.count * 3);
    for (let v = 0; v < pos.count; v++) {
      const t = (pos.getY(v) + 1.15) / 2.3; // 0 at the stem, 1 at the tip
      const w = pos.getX(v) * (0.35 + Math.sin(Math.PI * Math.min(t * 1.1, 1)) * 0.9);
      pos.setXYZ(v, w, t * 2.3, -t * t * 1.3 - Math.abs(w) * 0.25); // arches out and droops, edges cupped
      const c = 0.6 + 0.4 * t;
      col[v * 3] = col[v * 3 + 1] = col[v * 3 + 2] = c;
    }
    leaf.setAttribute("color", new THREE.BufferAttribute(col, 3));
    leaf.rotateX(-0.55 - r() * 0.35).rotateY((i / 7) * Math.PI * 2 + r() * 0.5).translate(0, 1.55 + r() * 0.2, 0);
    parts.push(leaf);
  }
  const g = mergeGeometries(parts);
  g.computeVertexNormals();
  return g;
}

// Crowns and banana leaves bend downwind and sway in gusts: the tops move
// most, each tree on its own phase. world.js feeds the time and the wind.
export const sway = { uTime: { value: 0 }, uWind: { value: new THREE.Vector2() } };
function swaying(material, amount) {
  material.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, sway);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nuniform float uTime; uniform vec2 uWind;")
      .replace("#include <begin_vertex>", `#include <begin_vertex>
        #ifdef USE_INSTANCING
          vec2 ax = normalize(instanceMatrix[0].xz), az = normalize(instanceMatrix[2].xz);
          vec2 home = instanceMatrix[3].xz * 3000.;
          float gust = .55 + .45 * sin(uTime * .37 + home.x * .013) * sin(uTime * .23 + home.y * .017);
          float bend = max(transformed.y + 1., 0.) * ${amount.toFixed(3)} * gust;
          float flutter = sin(uTime * 2.1 + home.x * .7 + home.y * .5 + transformed.y * 2.) * .35 + .65;
          transformed.x += dot(uWind, ax) * bend * flutter;
          transformed.z += dot(uWind, az) * bend * flutter;
        #endif`);
  };
  material.customProgramCacheKey = () => `sway-${amount}`;
  return material;
}

// light the leaf cards as if the crown were a solid ball: normals point out from its centre
function foliageNormals(geo) {
  geo.computeBoundingBox();
  const c = geo.boundingBox.getCenter(new THREE.Vector3()), v = new THREE.Vector3(), n = new THREE.Vector3();
  const pos = geo.attributes.position, nor = geo.attributes.normal;
  for (let i = 0; i < pos.count; i++) {
    v.set(pos.getX(i) - c.x, (pos.getY(i) - c.y) * 1.3, pos.getZ(i) - c.z).normalize();
    n.set(nor.getX(i), nor.getY(i), nor.getZ(i));
    if (n.dot(v) < 0) n.negate();
    n.lerp(v, 0.7).normalize();
    nor.setXYZ(i, n.x, n.y, n.z);
  }
  return geo;
}

// Chains extra shader edits onto a material, keeping any it already has.
function patch(material, key, edit) {
  const prev = material.onBeforeCompile, prevKey = material.customProgramCacheKey;
  material.onBeforeCompile = (sh, r) => { prev.call(material, sh, r); edit(sh); };
  material.customProgramCacheKey = () => `${prevKey.call(material)}|${key}`;
  return material;
}

// Leaf cards and chain-link are cut out of their texture by its alpha. Far away
// the texture's smaller mip levels average the holes in, the alpha falls under
// the cut and whole crowns vanish (leaving bare trunks) or a fence turns into a
// grey sheet. Raising the alpha with the mip level keeps the coverage, and alpha
// to coverage on the multisampled target softens the cut edges.
export function keepCoverage(material) {
  material.alphaToCoverage = true;
  return patch(material, "coverage", (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace("#include <alphatest_fragment>", `
      #ifdef USE_MAP
        { vec2 ts = vec2(textureSize(map, 0)) * vMapUv;
          float mip = max(0., .5 * log2(max(dot(dFdx(ts), dFdx(ts)), dot(dFdy(ts), dFdy(ts)))));
          diffuseColor.a *= 1. + mip * .28; }
      #endif
      #include <alphatest_fragment>`);
  });
}

// Levels of detail cross-fade instead of popping: each instance's distance from
// the camera (and from the middle of the scattered area, for its outer rim)
// picks a share of its pixels by a fixed screen pattern, so the near and the far
// version of a tree each draw the pixels the other leaves out.
export const scatterU = { uFocus: { value: new THREE.Vector2() } };
const f3 = (v) => v.toFixed(3);
function lodFade(material, near, far, edge) {
  return patch(material, `lod-${near}-${far}-${edge}`, (sh) => {
    Object.assign(sh.uniforms, scatterU);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec2 vLod; uniform vec2 uFocus;")
      .replace("#include <project_vertex>", `#include <project_vertex>
        #ifdef USE_INSTANCING
          vec4 lodC = modelMatrix * instanceMatrix * vec4(0., 0., 0., 1.);
          vLod = vec2(distance(lodC.xz, cameraPosition.xz), distance(lodC.xz, uFocus));
        #else
          vLod = vec2(0.);
        #endif`);
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying vec2 vLod;")
      .replace("#include <clipping_planes_fragment>", `#include <clipping_planes_fragment>
        {
          float keep = 1.;
          ${near ? `keep *= 1. - smoothstep(${f3(near[0])}, ${f3(near[1])}, vLod.x);` : ""}
          ${far ? `keep *= smoothstep(${f3(far[0])}, ${f3(far[1])}, vLod.x);` : ""}
          ${edge ? `keep *= 1. - smoothstep(${f3(edge[0])}, ${f3(edge[1])}, vLod.y);` : ""}
          float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(.06711056, .00583715))));
          if (ign >= keep) discard;
        }`);
  });
}

const WALLS = ["#c79a6c", "#b5845b", "#d8ccb6", "#e4ddcd", "#a9734f", "#cdb999", "#bf8f64"].map((c) => new THREE.Color(c));
const ROOFS = ["#8a8f94", "#a2a7ab", "#7c4632", "#93533a", "#6e757a", "#b0694a", "#5f6a72", "#9aa0a4"].map((c) => new THREE.Color(c));
const PLINTH = new THREE.Color("#7b6d5e");

/**
 * Houses and trees streamed around what the camera looks at.
 * Houses are the real footprints from OpenStreetMap (map/buildings.bin) where
 * it maps them; elsewhere they are placed where the satellite image shows roofs.
 * Trees grow where the image is green and dark and where OpenStreetMap maps
 * woodland, never inside a house. Everything sits on groundAt(), the visible
 * ground; houses on a slope stand on a plinth down to the low side.
 */
export function createScatter({ scene, groundAt, colorAt, roofAt, woodsAt, half, keepOut, town, models = {}, buildings = null }) {
  const CH = 0.5, R_HOUSE = 5, R_TREE = 4, MARGIN = 0.35;
  const HOUSE_NEAR = [0.95, 1.25], TREE_NEAR = [0.55, 0.8], TREE_MID = [1.7, 2.1];
  const H_EDGE = [R_HOUSE - 0.8, R_HOUSE], T_EDGE = [R_TREE - 0.7, R_TREE];
  const tNear = models.trees ? TREE_NEAR : null;

  // ------------------------------------------------ meshes
  const unitBox = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
  { // walls darken toward the ground, where light is blocked and dirt splashes up
    const pos = unitBox.attributes.position, col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = 0.68 + 0.32 * pos.getY(i);
    unitBox.setAttribute("color", new THREE.BufferAttribute(col, 3));
  }
  const unitRoof = gable(1, 1, 1);
  const trunkGeo = new THREE.CylinderGeometry(0.3, 0.5, 1, 6).translate(0, 0.5, 0);
  const crownMid = crownGeometry(3, 1, 5), banana = bananaGeometry();
  const crownFar = new THREE.OctahedronGeometry(1, 0).scale(1, 0.8, 1);
  const all = [];
  const inst = (geo, max, m, shadow = true) => {
    const im = new THREE.InstancedMesh(geo, m, max);
    im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3);
    im.count = 0; im.castShadow = shadow; im.receiveShadow = true;
    scene.add(im);
    all.push(im);
    return im;
  };
  const far = (m) => lodFade(m, null, models.houses ? HOUSE_NEAR : null, H_EDGE);
  const walls = inst(unitBox, 50000, far(mat("#ffffff", 0.9, 0, { vertexColors: true })));
  const roofs = inst(unitRoof, 50000, far(mat("#ffffff", 0.5, 0.35)));
  const plinths = inst(unitBox, 30000, lodFade(mat("#ffffff", 0.95, 0, { vertexColors: true }), null, null, H_EDGE));
  const leafy = { vertexColors: true, envMapIntensity: 0.6 };
  const trunks = inst(trunkGeo, 20000, lodFade(mat("#ffffff", 0.95), TREE_MID, tNear, T_EDGE));
  const crownsM = inst(crownMid, 20000, lodFade(swaying(mat("#ffffff", 0.92, 0, leafy), 0.06), TREE_MID, tNear, T_EDGE));
  const crownsF = inst(crownFar, 40000, lodFade(mat("#ffffff", 0.95), null, TREE_MID, T_EDGE), false);
  const bananas = inst(banana, 20000, lodFade(swaying(mat("#ffffff", 0.75, 0, { ...leafy, side: THREE.DoubleSide }), 0.1), [0.9, 1.1], tNear, T_EDGE));
  // instanced copies of the Blender models: four house types and four tree species up close
  const fromModel = (node, max, nearBand, leaves = 0) => {
    const mesh = node.isMesh ? node : node.children[0];
    const geo = leaves ? foliageNormals(mesh.geometry.clone()) : mesh.geometry;
    let m = mesh.material.clone();
    if (/^leaf_/.test(m.name)) m.color.multiplyScalar(m.name === "leaf_banana" ? 0.7 : 0.55);
    if (/bark/.test(node.name)) m.color.multiplyScalar(0.7);
    if (/^(leaf|alpha)_/.test(m.name)) m = keepCoverage(m);
    if (leaves) m = swaying(m, leaves);
    const im = inst(geo, max, lodFade(m, nearBand, null, leaves || /bark/.test(node.name) ? T_EDGE : H_EDGE));
    if (leaves) im.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: m.map, alphaTest: 0.5 }); // shadows with holes, not squares
    return im;
  };
  const HV = models.houses ? ["a", "b", "c", "d"].map((v) => {
    const parts = ["walls", "roof", "trim"].map((p) => fromModel(models.houses.get(`house_${v}_${p}`), 6000, HOUSE_NEAR));
    const box = new THREE.Box3().setFromBufferAttribute(parts[0].geometry.attributes.position);
    const sx = box.max.x - box.min.x, sz = box.max.z - box.min.z;
    // fit footprints along the model's long side
    return { walls: parts[0], roof: parts[1], trim: parts[2], turn: sz > sx ? Math.PI / 2 : 0, len: Math.max(sx, sz), wid: Math.min(sx, sz), h: box.max.y };
  }) : null;
  const SP = models.trees ? Object.fromEntries(["euc", "broad", "cypress", "banana"].map((sp) => [sp, {
    bark: fromModel(models.trees.get(`${sp}_bark`), 6000, TREE_NEAR),
    leaves: fromModel(models.trees.get(`${sp}_leaves`), 6000, TREE_NEAR, sp === "banana" ? 0.03 : 0.012),
  }])) : null;
  const WHITE = new THREE.Color(1, 1, 1);
  const BARK = new THREE.Color("#4a3a2c"), EUC_BARK = new THREE.Color("#544b40");
  const EUC_LEAF = new THREE.Color("#2e4233"), BROAD_LEAF = new THREE.Color("#2b4521"), CYPRESS_LEAF = new THREE.Color("#22361f"), BANANA_LEAF = new THREE.Color("#4f7029");

  // ------------------------------------------------ the real buildings, by chunk
  const osm = new Map();
  if (buildings) {
    const dv = buildings.data, o = buildings.origin;
    for (let i = 0; i < buildings.count; i++) {
      const x = (dv.getUint16(i * 8, true) - o) / 1000, z = (dv.getUint16(i * 8 + 2, true) - o) / 1000;
      const key = Math.floor(x / CH) * 100000 + Math.floor(z / CH);
      let l = osm.get(key);
      if (!l) osm.set(key, (l = []));
      l.push(i);
    }
  }
  // OpenStreetMap is trusted where it maps buildings around a chunk; elsewhere it is blank, not empty
  const mapped = (ix, iz) => {
    let n = 0;
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) n += osm.get((ix + dx) * 100000 + iz + dz)?.length || 0;
    return n >= 12;
  };

  const slope = (x, z) => Math.hypot(groundAt(x + 0.015, z) - groundAt(x - 0.015, z), groundAt(x, z + 0.015) - groundAt(x, z - 0.015)) / 0.03;
  const blocked = (x, z, pad) => keepOut.some((p) => (x - p.x) ** 2 + (z - p.z) ** 2 < (p.r + pad) ** 2);
  const rgb = [0, 0, 0];

  function houseDensity(x, z) {
    colorAt(x, z, rgb);
    const green = rgb[1] - (rgb[0] + rgb[2]) / 2;
    const dt = Math.hypot(x - town.x, z - town.z);
    const near = keepOut.reduce((m, p) => Math.max(m, 1 - sstep(0.2, 1.3, Math.hypot(x - p.x, z - p.z))), 0);
    return Math.min(0.05 + 0.85 * roofAt(x, z) + 0.25 * sstep(0.5, 0.76, fbm(x * 0.8, z * 0.8, 3)) * sstep(0.04, 0, green)
      + 0.5 * (1 - sstep(0.3, town.r, dt)) + 0.2 * near, 1);
  }
  function woodDensity(x, z) {
    colorAt(x, z, rgb);
    const green = rgb[1] - (rgb[0] + rgb[2]) / 2, bright = (rgb[0] + rgb[1] + rgb[2]) / 3;
    return Math.min(0.04 + 0.35 * sstep(0.55, 0.78, fbm(x * 1.3 + 40, z * 1.3, 9)) + 0.6 * sstep(0.01, 0.05, green) * sstep(0.32, 0.14, bright)
      + 0.7 * woodsAt(x, z), 1) * sstep(0.45, 0.3, bright);
  }

  // a house from its footprint: base on the uphill corner, a plinth down to the low side
  function house(x, z, w, d, yaw, levels, kind, seed) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    let lo = Infinity, hi = -Infinity;
    for (const [a, b] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const g = groundAt(x + (a * w * c - b * d * s) * 0.0005, z + (a * w * s + b * d * c) * 0.0005);
      lo = Math.min(lo, g); hi = Math.max(hi, g);
    }
    const drop = (hi - lo) * 1000;
    const base = drop < 2.5 ? hi - 0.00005 : (hi + lo) / 2;
    const v = kind === 1 || kind === 2 || w * d > 110 ? 3 : kind === 3 || w / d > 1.7 ? 0 : [0, 0, 0, 1, 2, 2][Math.floor(seed * 6)];
    const tall = v === 3 ? (levels > 1 ? levels / 2 : 0.55) : 0.92 + seed * 0.16;
    return { x, z, y: base, lo, plinth: (base - lo) * 1000 > 0.2, w, d, yaw, v, tall, levels,
      flat: v === 3, wc: WALLS[Math.floor(seed * 977) % WALLS.length], rc: ROOFS[Math.floor(seed * 7919) % ROOFS.length] };
  }

  function makeChunk(ix, iz) {
    const r = mulberry(Math.floor(hash(ix, iz, 7) * 4294967296));
    const x0 = ix * CH, z0 = iz * CH, houses = [], trees = [];
    const tree = (x, z, kind) => {
      if (houses.some((h) => (h.x - x) ** 2 + (h.z - z) ** 2 < ((Math.max(h.w, h.d) * 0.5 + 2) * 0.001) ** 2)) return;
      const y = groundAt(x, z);
      colorAt(x, z, rgb);
      const c = new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
      const spin = r() * Math.PI * 2, lb = 0.8 + r() * 0.32;
      const lc = new THREE.Color(lb, lb * (0.97 + r() * 0.06), lb * (0.92 + r() * 0.1));
      if (kind === 0) {        // eucalyptus: tall trunk, a narrow open blue-green crown
        const th = 9 + r() * 9, cr = 2.6 + r() * 1.6;
        trees.push({ kind, sp: "euc", ms: (th + 6) / 21, lc, x, y, z, spin, tr: 0.22, th, cr, ch: 6 + r() * 4, bark: EUC_BARK, c: c.lerp(EUC_LEAF, 0.8).multiplyScalar(0.8 + r() * 0.3) });
      } else if (kind === 1) { // banana
        const sz = 1.3 + r() * 0.8;
        trees.push({ kind, sp: "banana", ms: sz / 1.6, lc, x, y, z, spin, s: sz, c: c.lerp(BANANA_LEAF, 0.75).multiplyScalar(0.85 + r() * 0.3) });
      } else {                 // broadleaf: avocado, mango, ficus; and cypress
        const cr = 2.4 + r() * 2.6, cyp = r() < 0.22;
        trees.push({ kind, sp: cyp ? "cypress" : "broad", ms: cyp ? 0.8 + r() * 0.5 : cr / 3.6, lc, x, y, z, spin, tr: 0.32, th: 2 + r() * 2, cr: cyp ? 1.8 : cr, ch: cyp ? 9 : cr * (0.8 + r() * 0.35), bark: BARK, c: c.lerp(cyp ? CYPRESS_LEAF : BROAD_LEAF, 0.75).multiplyScalar(0.85 + r() * 0.3) });
      }
    };
    const isMapped = mapped(ix, iz);
    if (isMapped) {
      const dv = buildings.data, o = buildings.origin;
      for (const i of osm.get(ix * 100000 + iz) || []) {
        const x = (dv.getUint16(i * 8, true) - o) / 1000, z = (dv.getUint16(i * 8 + 2, true) - o) / 1000;
        if (blocked(x, z, 0.005)) continue;
        const k = dv.getUint8(i * 8 + 7);
        houses.push(house(x, z, dv.getUint8(i * 8 + 4) / 2, dv.getUint8(i * 8 + 5) / 2, (dv.getUint8(i * 8 + 6) / 255) * Math.PI, k & 7, k >> 3, hash(i, 3, 11)));
      }
    }
    // Where OpenStreetMap has no buildings, place them where the photo shows roofs. Inside a
    // mapped area this only fills blocks it has not mapped yet: clear roofs, away from mapped houses.
    const roadYaw = vnoise(ix * 0.35, iz * 0.35, 5) * Math.PI;
    for (let n = 0; n < (isMapped ? 140 : 90); n++) {
      const x = x0 + r() * CH, z = z0 + r() * CH;
      if (Math.abs(x) > half - 0.5 || Math.abs(z) > half - 0.5 || blocked(x, z, 0.02)) continue;
      if (r() > (isMapped ? sstep(0.3, 0.7, roofAt(x, z)) : houseDensity(x, z))) continue;
      if (slope(x, z) > 0.42) continue;
      const w = 6.5 + r() * 5, d = 5 + r() * 2.5;
      const gap = isMapped ? 0.012 : 0;
      if (houses.some((h) => (h.x - x) ** 2 + (h.z - z) ** 2 < ((Math.max(h.w, h.d) + w) * 0.6 * 0.001 + gap) ** 2)) continue;
      houses.push(house(x, z, w, d, roadYaw + (r() < 0.5 ? 0 : Math.PI / 2) + (r() - 0.5) * 0.25, 1, 0, r()));
    }
    for (const h of houses) for (let b = 0, nb = Math.floor(r() * 4); b < nb; b++) { // bananas and fruit trees by the houses
      const a = r() * Math.PI * 2, dd = (Math.max(h.w, h.d) * 0.5 + 4 + r() * 12) * M;
      const bx = h.x + Math.cos(a) * dd, bz = h.z + Math.sin(a) * dd;
      if (!blocked(bx, bz, 0.01)) tree(bx, bz, r() < 0.75 ? 1 : 2);
    }
    for (let n = 0; n < 160; n++) {
      const x = x0 + r() * CH, z = z0 + r() * CH;
      if (Math.abs(x) > half - 0.5 || Math.abs(z) > half - 0.5 || blocked(x, z, 0.015)) continue;
      if (r() > woodDensity(x, z)) continue;
      if (slope(x, z) > 0.9) continue;
      tree(x, z, r() < 0.7 ? 0 : 2);
    }
    return { houses, trees };
  }
  const cache = new Map();
  const chunk = (ix, iz) => {
    const key = ix * 100000 + iz;
    let c = cache.get(key);
    if (!c) {
      c = makeChunk(ix, iz);
      cache.set(key, c);
      if (cache.size > 1400) cache.delete(cache.keys().next().value);
    }
    return c;
  };

  // ------------------------------------------------ filling the instances
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3();
  let last = null, shown = false, pending = false;
  function rebuild(cam, focus) {
    const n = { walls: 0, plinth: 0, trunks: 0, mid: 0, far: 0, banana: 0 }, hn = [0, 0, 0, 0], tn = { euc: 0, broad: 0, cypress: 0, banana: 0 };
    let made = 0;
    pending = false;
    const span = Math.ceil(R_HOUSE / CH);
    const icx = Math.floor(focus.x / CH), icz = Math.floor(focus.z / CH);
    const near = (d, band) => d < band[1] + MARGIN, beyond = (d, band) => d > band[0] - MARGIN;
    for (let iz = icz - span; iz <= icz + span; iz++) for (let ix = icx - span; ix <= icx + span; ix++) {
      if (Math.hypot((ix + 0.5) * CH - focus.x, (iz + 0.5) * CH - focus.z) > R_HOUSE + CH) continue;
      if (Math.abs((ix + 0.5) * CH) > half || Math.abs((iz + 0.5) * CH) > half) continue;
      const key = ix * 100000 + iz;
      if (!cache.has(key) && made >= 10) { pending = true; continue; }
      if (!cache.has(key)) made++;
      const c = chunk(ix, iz);
      for (const h of c.houses) {
        if (Math.hypot(h.x - focus.x, h.z - focus.z) > R_HOUSE) continue;
        const dc = Math.hypot(h.x - cam.x, h.z - cam.z);
        if (h.plinth && n.plinth < 30000) {
          _q.setFromAxisAngle(UP, -h.yaw);
          plinths.setMatrixAt(n.plinth, _m.compose(_p.set(h.x, h.lo - 0.0003, h.z), _q, _s.set((h.w + 0.3) * M, h.y - h.lo + 0.0003, (h.d + 0.3) * M)));
          plinths.setColorAt(n.plinth++, PLINTH);
        }
        if (HV && near(dc, HOUSE_NEAR)) {
          const V = HV[h.v], i = hn[h.v]++;
          if (i < 6000) {
            _q.setFromAxisAngle(UP, -h.yaw + V.turn);
            _m.compose(_p.set(h.x, h.y, h.z), _q, V.turn ? _s.set((h.d / V.wid) * M, h.tall * M, (h.w / V.len) * M) : _s.set((h.w / V.len) * M, h.tall * M, (h.d / V.wid) * M));
            V.walls.setMatrixAt(i, _m); V.walls.setColorAt(i, h.wc);
            V.roof.setMatrixAt(i, _m); V.roof.setColorAt(i, h.rc);
            V.trim.setMatrixAt(i, _m); V.trim.setColorAt(i, WHITE);
          }
        }
        if (!beyond(dc, HOUSE_NEAR) || n.walls >= 50000) continue;
        const wallH = (h.flat ? 3.2 * Math.max(h.levels, 1) : 2.9 * h.tall) * M;
        _q.setFromAxisAngle(UP, -h.yaw);
        walls.setMatrixAt(n.walls, _m.compose(_p.set(h.x, h.y, h.z), _q, _s.set(h.w * M, wallH, h.d * M)));
        walls.setColorAt(n.walls, h.wc);
        _q.setFromAxisAngle(UP, -h.yaw + Math.PI / 2); // the gable's ridge runs along the long side
        roofs.setMatrixAt(n.walls, _m.compose(_p.set(h.x, h.y + wallH, h.z), _q, _s.set((h.d + 0.8) * M, (h.flat ? 0.3 : 1.2 + h.d * 0.12) * M, (h.w + 0.8) * M)));
        roofs.setColorAt(n.walls++, h.rc);
      }
      for (const t of c.trees) {
        if (Math.hypot(t.x - focus.x, t.z - focus.z) > R_TREE) continue;
        const dc = Math.hypot(t.x - cam.x, t.z - cam.z);
        _q.setFromAxisAngle(UP, t.spin);
        if (SP && near(dc, TREE_NEAR)) {
          const S = SP[t.sp], i = tn[t.sp]++;
          if (i < 6000) {
            _m.compose(_p.set(t.x, t.y - 0.15 * M, t.z), _q, _s.setScalar(t.ms * M));
            S.bark.setMatrixAt(i, _m); S.bark.setColorAt(i, WHITE);
            S.leaves.setMatrixAt(i, _m); S.leaves.setColorAt(i, t.lc);
          }
        }
        if (t.kind === 1) { // banana plants: only close by
          if (dc < 1.1 + MARGIN && (!SP || beyond(dc, TREE_NEAR)) && n.banana < 20000) {
            bananas.setMatrixAt(n.banana, _m.compose(_p.set(t.x, t.y - 0.1 * M, t.z), _q, _s.setScalar(t.s * M)));
            bananas.setColorAt(n.banana++, t.c);
          }
          continue;
        }
        if ((!SP || beyond(dc, TREE_NEAR)) && near(dc, TREE_MID) && n.mid < 20000) {
          trunks.setMatrixAt(n.mid, _m.compose(_p.set(t.x, t.y - 0.3 * M, t.z), _q, _s.set(t.tr * 2 * M, (t.th + 0.3) * M, t.tr * 2 * M)));
          trunks.setColorAt(n.mid, t.bark);
          crownsM.setMatrixAt(n.mid, _m.compose(_p.set(t.x, t.y + (t.th + t.ch * 0.45) * M, t.z), _q, _s.set(t.cr * M, t.ch * M, t.cr * M)));
          crownsM.setColorAt(n.mid++, t.c);
        }
        if (beyond(dc, TREE_MID) && n.far < 40000) {
          crownsF.setMatrixAt(n.far, _m.compose(_p.set(t.x, t.y + (t.th + t.ch * 0.4) * M, t.z), _q, _s.set(t.cr * 1.05 * M, t.ch * 0.62 * M, t.cr * 1.05 * M)));
          crownsF.setColorAt(n.far++, t.c);
        }
      }
    }
    const counts = [[walls, n.walls], [roofs, n.walls], [plinths, n.plinth], [trunks, n.mid], [crownsM, n.mid], [crownsF, n.far], [bananas, n.banana]];
    if (HV) HV.forEach((V, v) => counts.push([V.walls, Math.min(hn[v], 6000)], [V.roof, Math.min(hn[v], 6000)], [V.trim, Math.min(hn[v], 6000)]));
    if (SP) for (const sp in SP) counts.push([SP[sp].bark, Math.min(tn[sp], 6000)], [SP[sp].leaves, Math.min(tn[sp], 6000)]);
    for (const [im, k] of counts) {
      im.count = Math.min(k, im.instanceMatrix.count);
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      if (im.count) im.computeBoundingSphere(); // so the camera and the sun's shadow camera can cull it
    }
  }
  return {
    update(cam, focus, on) {
      for (const im of all) im.visible = on && im.count > 0;
      if (!on) { shown = false; return; }
      scatterU.uFocus.value.set(focus.x, focus.z);
      if (shown && !pending && last && Math.hypot(cam.x - last.cx, cam.z - last.cz) < 0.2 && Math.hypot(focus.x - last.fx, focus.z - last.fz) < 0.3) return;
      last = { cx: cam.x, cz: cam.z, fx: focus.x, fz: focus.z };
      shown = true;
      rebuild(cam, focus);
      for (const im of all) im.visible = im.count > 0;
    },
  };
}
