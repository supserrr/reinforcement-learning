// The 3D world: Muhanga's hills under a sky whose sun follows the shift clock.
// Terraced fields, eucalyptus woodlots, banana groves, villages, red dirt
// roads, a river and lakes, valley mist, clouds, the hub, six health centres
// and the delivery drone. app.js drives it; nothing here knows about the RL
// messages beyond site positions.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { Sky } from "three/addons/objects/Sky.js";
import { ImprovedNoise } from "three/addons/math/ImprovedNoise.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const WATER_LEVEL = 0.5;
// uniforms shared by every wind-driven shader
const U = {
  uTime: { value: 0 },
  uWind: { value: new THREE.Vector2(1, 0) },   // world-space direction the wind blows towards
  uWindStrength: { value: 0.3 },               // 0 calm .. 1 = 40 km/h
  uCloudOffset: { value: new THREE.Vector2() },
};
const WIND_GLSL = /* glsl */ `uniform float uTime; uniform vec2 uWind; uniform float uWindStrength; uniform vec2 uCloudOffset;`;
const TERRAIN_SIZE = 240;
const DEG = Math.PI / 180;

// ------------------------------------------------------------ deterministic noise
const perlin = new ImprovedNoise();
function rng(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function fbm(x, z, oct = 5) {
  let a = 1, f = 1, s = 0, n = 0;
  for (let i = 0; i < oct; i++) { s += a * perlin.noise(x * f, 11.7 + i * 3.1, z * f); n += a; a *= 0.5; f *= 2.03; }
  return s / n;
}
const smoothstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const softMax = (a, b, k) => (a + b + Math.sqrt((a - b) ** 2 + k)) / 2;

// A river winding north-south through the western valleys, and two lakes.
const RIVER = [[-75, -55], [-46, -36], [-33, -18], [-35, -2], [-31, 12], [-37, 26], [-30, 44], [-42, 75]];
const LAKES = [{ x: 30, z: 27, r: 6.5 }, { x: -9, z: 36, r: 4.5 }, { x: 42, z: -30, r: 7 }];
function distToRiver(x, z) {
  let best = 1e9;
  for (let i = 0; i < RIVER.length - 1; i++) {
    const [ax, az] = RIVER[i], [bx, bz] = RIVER[i + 1];
    const dx = bx - ax, dz = bz - az;
    const t = Math.min(Math.max(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0), 1);
    best = Math.min(best, Math.hypot(x - ax - t * dx, z - az - t * dz));
  }
  return best + perlin.noise(x * 0.2, 2.2, z * 0.2) * 0.8;
}

// ------------------------------------------------------------ world
export function createWorld(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
  const dpr = Math.min(window.devicePixelRatio, 2);
  renderer.setPixelRatio(dpr);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.55;
  container.appendChild(renderer.domElement);

  const labels = new CSS2DRenderer();
  labels.setSize(window.innerWidth, window.innerHeight);
  Object.assign(labels.domElement.style, { position: "fixed", inset: "0", pointerEvents: "none" });
  container.appendChild(labels.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2("#a9c4dc", 0.0048);

  const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.set(22, 24, 46);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3, 2);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.maxPolarAngle = Math.PI * 0.46;
  controls.minDistance = 4;
  controls.maxDistance = 110;
  controls.autoRotateSpeed = 0.35;

  // ---------------------------------------------------------- sky, sun, light
  const sky = new Sky();
  sky.scale.setScalar(4000);
  Object.assign(sky.material.uniforms.turbidity, { value: 3.2 });
  Object.assign(sky.material.uniforms.rayleigh, { value: 1.9 });
  Object.assign(sky.material.uniforms.mieCoefficient, { value: 0.004 });
  Object.assign(sky.material.uniforms.mieDirectionalG, { value: 0.82 });
  scene.add(sky);

  const envScene = new THREE.Scene();
  const envSky = new Sky();
  envSky.scale.setScalar(100);
  envSky.material.uniforms = sky.material.uniforms; // shared, so the reflection matches the sky
  envScene.add(envSky);
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null;

  const sun = new THREE.DirectionalLight("#fff4e6", 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, { left: -48, right: 48, top: 48, bottom: -48, near: 1, far: 260 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.04;
  sun.shadow.radius = 3;
  scene.add(sun, sun.target);
  const hemi = new THREE.HemisphereLight("#cfe3ff", "#4b5a2e", 0.55);
  scene.add(hemi);

  const sunDir = new THREE.Vector3();
  let lastEnvDir = new THREE.Vector3(0, -1, 0);
  const warm = new THREE.Color("#ffc890"), noon = new THREE.Color("#fff6ea");
  const hazeLow = new THREE.Color("#d6c0a2"), hazeHigh = new THREE.Color("#a6c1da");

  // Muhanga is 2°S; late November the sun's declination is about -20°.
  function setTime(minutes) {
    const hour = 8 + minutes / 60;
    const H = (hour - 12) * 15 * DEG, lat = -2.1 * DEG, dec = -20 * DEG;
    const sinAlt = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(H);
    const alt = Math.asin(sinAlt);
    const az = Math.atan2(Math.sin(H), Math.cos(H) * Math.sin(lat) - Math.tan(dec) * Math.cos(lat)) + Math.PI;
    sunDir.set(Math.cos(alt) * Math.sin(az), Math.sin(alt), -Math.cos(alt) * Math.cos(az)).normalize();
    sky.material.uniforms.sunPosition.value.copy(sunDir);
    sun.position.copy(sunDir).multiplyScalar(120);
    const k = smoothstep(12, 48, alt / DEG);
    sun.color.copy(warm).lerp(noon, k);
    sun.intensity = 2.4 + 1.6 * k;
    hemi.intensity = 0.4 + 0.25 * k;
    scene.fog.color.copy(hazeLow).lerp(hazeHigh, k);
    if (sunDir.angleTo(lastEnvDir) > 1.5 * DEG) {
      if (envRT) envRT.dispose();
      envRT = pmrem.fromScene(envScene, 0, 0.1, 1000);
      scene.environment = envRT.texture;
      lastEnvDir.copy(sunDir);
    }
    mistTime = minutes;
  }

  // ---------------------------------------------------------- post-processing
  const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
  composer.setPixelRatio(Math.min(dpr, 1.5));
  composer.setSize(window.innerWidth, window.innerHeight);
  composer.addPass(new RenderPass(scene, camera));
  const ao = new GTAOPass(scene, camera, window.innerWidth, window.innerHeight);
  ao.updateGtaoMaterial({ radius: 0.9, distanceExponent: 1.5, thickness: 1.5, scale: 1.1, samples: 12 });
  ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 12 });
  ao.blendIntensity = 0.85;
  // AO must only see solid geometry: sprites, light beams, rings and streaks would print dark shapes
  const baseHide = ao.overrideVisibility.bind(ao);
  ao.overrideVisibility = function () {
    baseHide();
    this.scene.traverse((o) => { if (o.isSprite || (o.material && o.material.transparent)) o.visible = false; });
  };
  composer.addPass(ao);
  const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.32, 0.55, 0.92);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
    labels.setSize(window.innerWidth, window.innerHeight);
  });

  // ---------------------------------------------------------- terrain
  let SITES = [];

  function rawHeight(x, z) {
    const n = fbm(x * 0.052, z * 0.052, 5);
    const ridge = 1 - Math.abs(perlin.noise(x * 0.028, 4.4, z * 0.028));
    let h = 3.4 + n * 10.5 + ridge * 2.0;
    h = softMax(h, 0.9, 1.4); // flat valley floors (marshes, rice)
    const d = distToRiver(x, z);
    h = THREE.MathUtils.lerp(Math.min(h, 1.0), h, smoothstep(2.5, 9, d)); // floodplain
    h = THREE.MathUtils.lerp(0.1, h, smoothstep(0.6, 1.9, d)); // channel
    for (const L of LAKES) {
      const dl = Math.hypot(x - L.x, z - L.z) + perlin.noise(x * 0.3, 8.8, z * 0.3) * 1.2;
      h = THREE.MathUtils.lerp(0.05, h, smoothstep(L.r * 0.7, L.r * 1.6, dl));
    }
    return h;
  }
  function heightAt(x, z) {
    let h = rawHeight(x, z);
    for (const s of SITES) {
      const d = Math.hypot(x - s.wx, z - s.wz);
      const w = 1 - smoothstep(1.6, 4.5, d);
      if (w > 0) h = THREE.MathUtils.lerp(h, s.h, w);
    }
    return h;
  }

  const terrainShader = {
    head: /* glsl */ `
      varying vec3 vWPos; varying vec3 vWNorm;
      ${WIND_GLSL}
      float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
      vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(.1031,.1030,.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
      float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.-2.*f);
        return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
      vec3 voronoi(vec2 x){ vec2 n = floor(x), f = fract(x); float md = 8., md2 = 8.; vec2 mg = vec2(0);
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) { vec2 g = vec2(i, j); vec2 r = g + hash22(n + g) - f; float d = dot(r, r);
          if (d < md) { md2 = md; md = d; mg = g; } else if (d < md2) md2 = d; }
        return vec3(hash12(n + mg), sqrt(md), sqrt(md2) - sqrt(md)); }
      vec3 srgb(vec3 c){ return pow(c, vec3(2.2)); }
      float cloudShade(vec3 p){
        vec2 q = (p.xz + uCloudOffset) * .02;
        float n = vnoise(q) * .6 + vnoise(q * 2.1 + 7.) * .3 + vnoise(q * 4.3 + 3.) * .1;
        return 1. - .26 * smoothstep(.55, .8, n);
      }
      vec3 landColor(vec3 p, vec3 nrm){
        float slope = 1.0 - clamp(nrm.y, 0.0, 1.0);
        float h = p.y;
        float band = h * 7.0 + vnoise(p.xz * 1.3) * .7;          // one terrace per band of height
        float bi = floor(band);
        // terraces are strips along the contour, cut into plots where a smooth noise field crosses them
        float n6 = vnoise(p.xz * 1.1 + 3.) * 5. + vnoise(p.xz * 3.1) * .6;
        float seg = floor(n6);
        float flatness = 1. - smoothstep(.05, .18, slope);
        vec3 v = voronoi(p.xz * 3. + vec2(bi * 1.7, seg * 3.1));
        float id = hash12(vec2(bi * 13.1 + seg, flatness > .5 ? v.x * 57. : 0.));
        float segEdge = min(fract(n6), 1. - fract(n6)) / max(fwidth(n6), 1e-4);
        float hedge = 1. - smoothstep(.0, 1.6, segEdge);
        hedge = max(hedge, flatness * (1. - smoothstep(.0, .03, v.z)));
        float riser = smoothstep(.82, 1., abs(fract(band) - .5) * 2.);
        vec3 c = id < .22 ? vec3(.22,.36,.13)   // banana
               : id < .44 ? vec3(.36,.47,.19)   // beans
               : id < .58 ? vec3(.55,.55,.27)   // maize, drying
               : id < .70 ? vec3(.25,.45,.15)   // tea / sweet potato
               : id < .82 ? vec3(.47,.44,.27)   // fallow
               : id < .90 ? vec3(.50,.34,.22)   // tilled red soil
               :            vec3(.30,.40,.16);  // sorghum
        c = mix(c, vec3(.31,.42,.18), .3);                                    // real plots differ subtly
        c *= .9 + .18 * hash12(vec2(id * 91.7, 3.1));
        c *= .95 + .05 * sin(band * 55. + vnoise(p.xz * 9.) * 2.);            // crop rows along the contour
        c = mix(c, c * .8, hedge);                                          // paths and hedgerows between plots
        c = mix(c, c * .62 + vec3(.04,.05,.02), riser * smoothstep(.05, .25, slope)); // grassy terrace risers
        float forest = smoothstep(.36, .52, slope + (vnoise(p.xz * .3) - .5) * .55);
        c = mix(c, vec3(.13,.21,.10) * (.85 + .3 * vnoise(p.xz * 6.)), forest); // woodland on steep ground
        float marsh = 1. - smoothstep(.75, 1.25, h);
        c = mix(c, vec3(.33,.45,.19) * (.9 + .2 * vnoise(p.xz * 3.)), marsh * .9); // valley marsh and rice
        float shore = 1. - smoothstep(${WATER_LEVEL.toFixed(2)}, ${(WATER_LEVEL + 0.18).toFixed(2)}, h);
        c = mix(c, vec3(.42,.34,.22), shore);
        c *= .9 + .2 * vnoise(p.xz * 18.);
        // gusts rolling across the crops, downwind
        float along = dot(p.xz, uWind);
        float wave = sin(along * 2.2 - uTime * (1.2 + 2.5 * uWindStrength) + vnoise(p.xz * .5) * 3.);
        float crops = (1. - forest) * (1. - marsh);
        c *= 1. + wave * .07 * uWindStrength * crops * smoothstep(.2, .8, vnoise(p.xz * .25 - uWind * uTime * .15));
        float far = smoothstep(30., 95., length(p - cameraPosition));
        c = mix(c, vec3(.27,.38,.17), far * .6);                              // plots blur together with distance
        return srgb(c);
      }`,
  };

  function grainTexture() {
    const n = 512, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d"), img = g.createImageData(n, n), r = rng(3);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = 128 + perlin.noise(x * 0.08, y * 0.08, 0.5) * 70 + perlin.noise(x * 0.3, y * 0.3, 2.5) * 30 + (r() - 0.5) * 40;
      const i = (y * n + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(260, 260);
    t.anisotropy = 8;
    return t;
  }

  let terrain;
  function buildTerrain() {
    const seg = 520;
    const geo = new THREE.PlaneGeometry(TERRAIN_SIZE, TERRAIN_SIZE, seg, seg);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setY(i, heightAt(pos.getX(i), pos.getZ(i)));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.96, metalness: 0, envMapIntensity: 0.35,
      bumpMap: grainTexture(), bumpScale: 1.4 });
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vWPos; varying vec3 vWNorm;")
        .replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWNorm = normalize(mat3(modelMatrix) * objectNormal);");
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <common>", "#include <common>\n" + terrainShader.head)
        .replace("#include <color_fragment>", "#include <color_fragment>\ndiffuseColor.rgb = landColor(vWPos, normalize(vWNorm));")
        .replace("#include <lights_fragment_end>", "#include <lights_fragment_end>\nfloat cs = cloudShade(vWPos); reflectedLight.directDiffuse *= cs; reflectedLight.directSpecular *= cs;");
      Object.assign(sh.uniforms, U);
    };
    terrain = new THREE.Mesh(geo, mat);
    terrain.receiveShadow = true;
    terrain.castShadow = true;
    scene.add(terrain);
  }

  // ---------------------------------------------------------- water
  let waterNormal;
  function rippleTexture() {
    const n = 256, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d"), img = g.createImageData(n, n);
    const h = (x, y) => Math.sin(x * 0.19 + Math.sin(y * 0.07) * 2) * 0.5 + Math.sin(y * 0.23 + x * 0.05) * 0.5
      + perlin.noise(x * 0.06, y * 0.06, 1.5) * 1.2;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const dx = h(x + 1, y) - h(x - 1, y), dy = h(x, y + 1) - h(x, y - 1);
      const v = new THREE.Vector3(-dx, -dy, 2.4).normalize();
      const i = (y * n + x) * 4;
      img.data[i] = (v.x * 0.5 + 0.5) * 255; img.data[i + 1] = (v.y * 0.5 + 0.5) * 255; img.data[i + 2] = (v.z * 0.5 + 0.5) * 255; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(60, 60);
    return t;
  }
  function buildWater() {
    waterNormal = rippleTexture();
    const water = new THREE.Mesh(
      new THREE.PlaneGeometry(TERRAIN_SIZE, TERRAIN_SIZE),
      new THREE.MeshPhysicalMaterial({ color: "#3f7690", roughness: 0.14, metalness: 0.0, normalMap: waterNormal,
        normalScale: new THREE.Vector2(0.35, 0.35), envMapIntensity: 1.1, clearcoat: 0.6, clearcoatRoughness: 0.12 }));
    water.rotation.x = -Math.PI / 2;
    water.position.y = WATER_LEVEL;
    water.receiveShadow = true;
    scene.add(water);
  }

  // ---------------------------------------------------------- vegetation
  const tint = (geo, hex) => {
    const c = new THREE.Color(hex), n = geo.attributes.position.count, a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) a.set([c.r, c.g, c.b], i * 3);
    geo.setAttribute("color", new THREE.BufferAttribute(a, 3));
    return geo;
  };
  const lumpy = (geo, amount, seed) => {
    const r = rng(seed), p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const k = 1 + (r() - 0.5) * amount;
      p.setXYZ(i, p.getX(i) * k, p.getY(i) * (1 + (r() - 0.5) * amount * 0.5), p.getZ(i) * k);
    }
    geo.computeVertexNormals();
    return geo;
  };
  const lobe = (r, seed, sx, sy, sz, x, y, z, hex) =>
    tint(lumpy(new THREE.IcosahedronGeometry(r, 2), 0.32, seed).scale(sx, sy, sz).translate(x, y, z), hex).toNonIndexed();
  function eucalyptusGeo() { // tall pale trunk, narrow grey-green crown in two masses
    const trunk = tint(new THREE.CylinderGeometry(0.014, 0.028, 0.8, 6).translate(0, 0.4, 0), "#7a705f");
    return mergeGeometries([trunk.toNonIndexed(),
      lobe(0.17, 3, 0.9, 1.7, 0.9, 0.02, 0.74, 0, "#4f6140"), lobe(0.14, 4, 1, 1.5, 1, -0.05, 0.96, 0.03, "#5a6b48")]);
  }
  function broadleafGeo() { // a rounded crown made of several leaf masses
    const trunk = tint(new THREE.CylinderGeometry(0.025, 0.045, 0.32, 6).translate(0, 0.16, 0), "#5e4a36");
    return mergeGeometries([trunk.toNonIndexed(),
      lobe(0.2, 7, 1.1, 0.8, 1.1, 0, 0.42, 0, "#2f5222"), lobe(0.15, 8, 1, 0.8, 1, 0.13, 0.5, 0.06, "#3a6128"),
      lobe(0.14, 9, 1, 0.8, 1, -0.1, 0.52, -0.08, "#345a24"), lobe(0.12, 10, 1, 0.8, 1, 0.02, 0.6, 0.1, "#3f672b")]);
  }
  function bananaGeo() { // a pseudostem and arching leaves that droop at the tips
    const parts = [tint(new THREE.CylinderGeometry(0.018, 0.028, 0.2, 6).translate(0, 0.1, 0), "#6f7d3c").toNonIndexed()];
    const outline = new THREE.Shape();
    outline.moveTo(0, 0); outline.quadraticCurveTo(0.045, 0.1, 0.012, 0.26); outline.lineTo(-0.012, 0.26); outline.quadraticCurveTo(-0.045, 0.1, 0, 0);
    for (let i = 0; i < 7; i++) {
      const leaf = new THREE.ShapeGeometry(outline, 6).rotateX(-Math.PI / 2);
      const lp = leaf.attributes.position;
      for (let k = 0; k < lp.count; k++) { const z = -lp.getZ(k); lp.setY(k, 0.09 * z - 0.55 * z * z); }
      leaf.rotateX(-0.25 + (i % 3) * 0.12).rotateY((i / 7) * Math.PI * 2 + i * 0.3).translate(0, 0.19, 0);
      leaf.computeVertexNormals();
      parts.push(tint(leaf, i % 2 ? "#7fae44" : "#6a9a38").toNonIndexed());
    }
    return mergeGeometries(parts);
  }
  function scatter(geo, count, seed, accept, scaleRange, material) {
    const mesh = new THREE.InstancedMesh(geo, material, count);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const r = rng(seed), m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
    const col = new THREE.Color();
    let n = 0, tries = 0;
    while (n < count && tries < count * 40) {
      tries++;
      const x = (r() - 0.5) * 150, z = (r() - 0.5) * 150;
      const h = heightAt(x, z);
      if (h < WATER_LEVEL + 0.25 || !accept(x, z, h, r)) continue;
      if (SITES.some((st) => (x - st.wx) ** 2 + (z - st.wz) ** 2 < (st.hub ? 9 : 4))) continue;
      const k = scaleRange[0] + r() * (scaleRange[1] - scaleRange[0]);
      s.set(k, k * (0.85 + r() * 0.3), k);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r() * Math.PI * 2);
      m.compose(p.set(x, h - 0.02, z), q, s);
      mesh.setMatrixAt(n, m);
      col.setHSL(0.24 + (r() - 0.5) * 0.06, 0.25 + r() * 0.25, 0.42 + r() * 0.2);
      mesh.setColorAt(n, col.lerp(new THREE.Color(1, 1, 1), 0.7));
      n++;
    }
    mesh.count = n;
    scene.add(mesh);
    return mesh;
  }
  const slopeAt = (x, z) => {
    const e = 0.4;
    return Math.hypot(heightAt(x + e, z) - heightAt(x - e, z), heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
  };
  const villageness = (x, z) => perlin.noise(x * 0.09, 21.3, z * 0.09);

  function buildVegetation() {
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.82, envMapIntensity: 0.45, side: THREE.DoubleSide });
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, U);
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\n" + WIND_GLSL)
        .replace("#include <begin_vertex>", `#include <begin_vertex>
        #ifdef USE_INSTANCING
          mat3 swM = mat3(instanceMatrix);
          vec3 ip = instanceMatrix[3].xyz;
          float sc2 = dot(swM[0], swM[0]);
          float hgt = max(position.y, 0.0);
          float gust = .55 + .45 * sin(uTime * 1.1 + ip.x * .31 + ip.z * .23) * sin(uTime * .37 + ip.z * .11);
          float flutter = sin(uTime * 5.5 + ip.x * 3.1 + position.y * 9.) * .12;
          float amount = hgt * hgt * uWindStrength * (gust + flutter) * .55;
          transformed += transpose(swM) * vec3(uWind.x, 0., uWind.y) / sc2 * amount;
        #endif`);
    };
    mat.customProgramCacheKey = () => "sway";
    // eucalyptus woodlots on hilltops and steep ground, in patches
    scatter(eucalyptusGeo(), 11000, 11, (x, z, h, r) =>
      perlin.noise(x * 0.12, 5.5, z * 0.12) + slopeAt(x, z) * 0.35 > 0.3 + r() * 0.1, [0.42, 0.8], mat);
    // banana groves around houses
    scatter(bananaGeo(), 9000, 12, (x, z, h, r) => villageness(x, z) > 0.1 + r() * 0.15 && slopeAt(x, z) < 1.2, [0.5, 0.85], mat);
    // scattered broadleaf trees
    scatter(broadleafGeo(), 5000, 13, (x, z, h, r) => r() < 0.5, [0.45, 0.85], mat);
  }

  // ---------------------------------------------------------- villages
  function buildVillages() {
    const walls = new THREE.BoxGeometry(0.15, 0.075, 0.1).translate(0, 0.0375, 0);
    const roofShape = new THREE.Shape([new THREE.Vector2(-0.062, 0), new THREE.Vector2(0.062, 0), new THREE.Vector2(0, 0.045)]);
    const roof = new THREE.ExtrudeGeometry(roofShape, { depth: 0.17, bevelEnabled: false }).translate(0, 0.075, -0.085).rotateY(Math.PI / 2);
    const count = 4200;
    const wallMesh = new THREE.InstancedMesh(walls, new THREE.MeshStandardMaterial({ roughness: 0.92 }), count);
    const roofMesh = new THREE.InstancedMesh(roof, new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0.65, envMapIntensity: 1.2 }), count);
    for (const m of [wallMesh, roofMesh]) { m.castShadow = true; m.receiveShadow = true; }
    const wallCols = ["#c9a27a", "#a65a3a", "#e4d8c4", "#efeee8", "#8b6a4a", "#d8b98f"].map((c) => new THREE.Color(c));
    const roofCols = ["#b9bdc1", "#a9adb2", "#8e5a3c", "#9c3f2a", "#c7cacc", "#6b4a3a"].map((c) => new THREE.Color(c));
    const r = rng(21), m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1), p = new THREE.Vector3();
    let n = 0, tries = 0;
    while (n < count && tries < count * 60) {
      tries++;
      let x, z;
      if (r() < 0.25 && SITES.length) { // towns gather round the health centres
        const st = SITES[Math.floor(r() * SITES.length)];
        const a = r() * Math.PI * 2, d = 2.6 + r() * 4;
        x = st.wx + Math.cos(a) * d; z = st.wz + Math.sin(a) * d;
      } else { x = (r() - 0.5) * 140; z = (r() - 0.5) * 140; }
      const h = heightAt(x, z);
      if (h < 1.3 || slopeAt(x, z) > 1.1 || villageness(x, z) < 0.05 + r() * 0.2) continue;
      if (SITES.some((st) => (x - st.wx) ** 2 + (z - st.wz) ** 2 < (st.hub ? 7 : 4))) continue;
      const k = 0.55 + r() * 0.45;
      s.set(k, k, k);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r() * Math.PI);
      m.compose(p.set(x, h - 0.015, z), q, s);
      wallMesh.setMatrixAt(n, m);
      roofMesh.setMatrixAt(n, m);
      wallMesh.setColorAt(n, wallCols[Math.floor(r() * wallCols.length)]);
      roofMesh.setColorAt(n, roofCols[Math.floor(r() * roofCols.length)]);
      n++;
    }
    wallMesh.count = roofMesh.count = n;
    scene.add(wallMesh, roofMesh);
  }

  // ---------------------------------------------------------- roads
  const roadCurves = [];
  function buildRoads() {
    const mat = new THREE.MeshStandardMaterial({ color: "#86654a", roughness: 1, polygonOffset: true, polygonOffsetFactor: -2 });
    const r = rng(5);
    const hub = SITES.find((s) => s.hub);
    const routes = SITES.filter((s) => !s.hub).map((s) => [hub, s]);
    routes.push([SITES[1], SITES[6]], [SITES[3], SITES[4]], [SITES[4], SITES[5]]);
    for (const [a, b] of routes) {
      const pts = [];
      const dx = b.wx - a.wx, dz = b.wz - a.wz, len = Math.hypot(dx, dz), nx = -dz / len, nz = dx / len;
      for (let i = 0; i <= 6; i++) {
        const t = i / 6, wob = i === 0 || i === 6 ? 0 : (r() - 0.5) * len * 0.22;
        pts.push(new THREE.Vector3(a.wx + dx * t + nx * wob, 0, a.wz + dz * t + nz * wob));
      }
      const curve = new THREE.CatmullRomCurve3(pts);
      const heights = new Float32Array(257);
      for (let i = 0; i <= 256; i++) { const q = curve.getPointAt(i / 256); heights[i] = Math.max(heightAt(q.x, q.z), WATER_LEVEL + 0.02); }
      roadCurves.push({ curve, len, heights });
      const samples = Math.ceil(len / 0.12), pos = [], idx = [];
      for (let i = 0; i <= samples; i++) {
        const p = curve.getPointAt(i / samples), tan = curve.getTangentAt(i / samples);
        const side = new THREE.Vector3(-tan.z, 0, tan.x).normalize().multiplyScalar(0.05);
        for (const sgn of [-1, 1]) {
          const x = p.x + side.x * sgn, z = p.z + side.z * sgn;
          pos.push(x, Math.max(heightAt(x, z), WATER_LEVEL + 0.02) + 0.03, z);
        }
        if (i < samples) { const k = i * 2; idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setIndex(idx);
      g.computeVertexNormals();
      const road = new THREE.Mesh(g, mat);
      road.receiveShadow = true;
      scene.add(road);
    }
  }

  // ---------------------------------------------------------- mist and clouds
  function softTexture(seed, lumps) {
    const n = 256, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d"), r = rng(seed);
    for (let i = 0; i < lumps; i++) {
      const x = n * (0.32 + r() * 0.36), y = n * (0.4 + r() * 0.2), rad = n * (0.08 + r() * 0.1);
      const gr = g.createRadialGradient(x, y, 0, x, y, rad);
      gr.addColorStop(0, "rgba(255,255,255,0.35)");
      gr.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = gr;
      g.fillRect(0, 0, n, n);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }
  const mist = [], clouds = [];
  let mistTime = 0;
  function buildAtmosphere() {
    const r = rng(31);
    const mistTex = softTexture(4, 14);
    let n = 0;
    while (n < 55) {
      const x = (r() - 0.5) * 140, z = (r() - 0.5) * 140, h = heightAt(x, z);
      if (h > 1.6) continue;
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0.6, fog: true }));
      const k = 7 + r() * 10;
      s.scale.set(k, k * 0.32, 1);
      s.position.set(x, h + 0.5 + r() * 0.5, z);
      s.userData.base = 0.22 + r() * 0.2;
      scene.add(s);
      mist.push(s);
      n++;
    }
    for (let c = 0; c < 16; c++) {
      const tex = softTexture(100 + c, 22);
      const cx = (r() - 0.5) * 200, cz = (r() - 0.5) * 200, cy = 30 + r() * 10;
      for (let i = 0; i < 5; i++) {
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0.85, fog: false }));
        const k = 12 + r() * 14;
        s.scale.set(k, k * 0.55, 1);
        s.position.set(cx + (r() - 0.5) * 16, cy + (r() - 0.5) * 3, cz + (r() - 0.5) * 10);
        scene.add(s);
        clouds.push(s);
      }
    }
  }

  // ---------------------------------------------------------- life: smoke, birds, traffic, wind
  const chimneys = [], birds = [], vehicles = [];
  let streaks, streakData = [];
  function buildLife() {
    const r = rng(77);
    const smokeTex = softTexture(9, 10);
    let tries = 0;
    while (chimneys.length < 16 && tries++ < 4000) { // cooking fires in the villages
      const x = (r() - 0.5) * 90, z = (r() - 0.5) * 90, h = heightAt(x, z);
      if (h < 1.4 || villageness(x, z) < 0.18) continue;
      const puffs = [];
      for (let i = 0; i < 7; i++) {
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, color: "#d9d6cf", transparent: true, depthWrite: false, opacity: 0 }));
        sp.userData.age = i / 7;
        scene.add(sp);
        puffs.push(sp);
      }
      chimneys.push({ x, y: h + 0.12, z, puffs, rate: 0.12 + r() * 0.08 });
    }
    // birds circling over the valleys
    const birdMat = new THREE.MeshStandardMaterial({ color: "#1f1f1f", roughness: 1, side: THREE.DoubleSide });
    const wingGeo = new THREE.BufferGeometry();
    wingGeo.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0.05, 0, 0, -0.05, 0.22, 0, -0.02], 3));
    wingGeo.computeVertexNormals();
    for (let f = 0; f < 4; f++) {
      const cx = (r() - 0.5) * 70, cz = (r() - 0.5) * 70;
      for (let i = 0; i < 7; i++) {
        const b = new THREE.Group();
        const left = new THREE.Mesh(wingGeo, birdMat), right = new THREE.Mesh(wingGeo, birdMat);
        right.scale.x = -1;
        b.add(left, right);
        b.userData = { cx, cz, rad: 2.5 + r() * 3, alt: 9 + r() * 4, speed: 0.25 + r() * 0.15, phase: r() * Math.PI * 2, left, right };
        scene.add(b);
        birds.push(b);
      }
    }
    // moto-taxis and pickups on the dirt roads
    const motoGeo = new THREE.BoxGeometry(0.035, 0.05, 0.1).translate(0, 0.045, 0);
    const truckGeo = new THREE.BoxGeometry(0.08, 0.07, 0.17).translate(0, 0.05, 0);
    for (const [geo, count, palette] of [[motoGeo, 34, ["#1d6b3a", "#c1121f", "#f2bb07", "#1c4f8f"]], [truckGeo, 10, ["#f2f2ee", "#b9bdc1", "#2f5d8a"]]]) {
      const mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.2 }), count);
      mesh.castShadow = true;
      for (let i = 0; i < count; i++) {
        mesh.setColorAt(i, new THREE.Color(palette[Math.floor(r() * palette.length)]));
        vehicles.push({ mesh, i, road: roadCurves[Math.floor(r() * roadCurves.length)], t: r(), dir: r() < 0.5 ? 1 : -1, speed: (geo === motoGeo ? 0.9 : 0.6) * (0.7 + r() * 0.6) });
      }
      scene.add(mesh);
    }
    // streaks of moving air, thicker and faster as the wind rises
    const c = document.createElement("canvas"); c.width = 64; c.height = 4;
    const g = c.getContext("2d"), gr = g.createLinearGradient(0, 0, 64, 0);
    gr.addColorStop(0, "rgba(255,255,255,0)"); gr.addColorStop(0.7, "rgba(255,255,255,0.9)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 64, 4);
    const N = 260;
    streaks = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 0.03).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, opacity: 0.0, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: true }), N);
    streaks.frustumCulled = false;
    for (let i = 0; i < N; i++) { const x = (r() - 0.5) * 70, z = (r() - 0.5) * 70; streakData.push({ x, z, y: 4 + r() * 9, g: heightAt(x, z) * 0.35, life: r(), len: 1.2 + r() * 2.2 }); }
    scene.add(streaks);
  }
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
  function updateLife(now, dt) {
    const wx = U.uWind.value.x, wz = U.uWind.value.y, ws = U.uWindStrength.value;
    for (const ch of chimneys) for (const sp of ch.puffs) {
      sp.userData.age += dt * ch.rate;
      if (sp.userData.age > 1) sp.userData.age -= 1;
      const a = sp.userData.age;
      sp.position.set(ch.x + wx * a * ws * 2.6, ch.y + a * 1.6 * (1 - ws * 0.4), ch.z + wz * a * ws * 2.6);
      sp.scale.setScalar(0.15 + a * 0.75);
      sp.material.opacity = 0.42 * Math.sin(Math.PI * Math.min(a * 1.15, 1));
    }
    for (const b of birds) {
      const d = b.userData, t = now / 1000 * d.speed + d.phase;
      d.cx += wx * ws * dt * 0.3; d.cz += wz * ws * dt * 0.3;
      b.position.set(d.cx + Math.cos(t) * d.rad, d.alt + Math.sin(t * 2.3) * 0.3, d.cz + Math.sin(t) * d.rad);
      b.rotation.y = -t;
      const flap = Math.sin(now / 1000 * 9 + d.phase * 5) * 0.55;
      d.left.rotation.z = flap; d.right.rotation.z = -flap;
    }
    for (const v of vehicles) {
      v.t += (v.dir * v.speed * dt) / v.road.len;
      if (v.t > 1 || v.t < 0) { v.dir *= -1; v.t = Math.min(Math.max(v.t, 0), 1); }
      const pt = v.road.curve.getPointAt(v.t), tan = v.road.curve.getTangentAt(v.t);
      const side = 0.035 * v.dir;
      _p.set(pt.x - tan.z * side, 0, pt.z + tan.x * side);
      _p.y = v.road.heights[Math.round(v.t * 256)] + 0.05;
      _q.setFromAxisAngle(UP, Math.atan2(tan.x * v.dir, tan.z * v.dir));
      v.mesh.setMatrixAt(v.i, _m.compose(_p, _q, _s.set(1.4, 1.4, 1.4)));
    }
    for (const [mesh] of [[vehicles[0]?.mesh], [vehicles[vehicles.length - 1]?.mesh]]) if (mesh) mesh.instanceMatrix.needsUpdate = true;
    // air streaks around the camera's focus
    const yaw = Math.atan2(wx, wz) - Math.PI / 2;
    _q.setFromAxisAngle(UP, yaw);
    const cx = controls.target.x, cz = controls.target.z;
    streakData.forEach((sd, i) => {
      sd.life += dt * (0.25 + ws * 0.5);
      sd.x += wx * dt * (3 + ws * 14); sd.z += wz * dt * (3 + ws * 14);
      if (sd.life > 1 || Math.abs(sd.x - cx) > 36 || Math.abs(sd.z - cz) > 36) {
        sd.life = 0; sd.x = cx + (Math.random() - 0.5) * 70; sd.z = cz + (Math.random() - 0.5) * 70; sd.g = heightAt(sd.x, sd.z) * 0.35;
      }
      const fade = Math.sin(Math.PI * sd.life);
      streaks.setMatrixAt(i, _m.compose(_p.set(sd.x, sd.y + sd.g, sd.z), _q, _s.set(sd.len * (0.4 + ws), 1, Math.max(fade, 0.001))));
    });
    streaks.instanceMatrix.needsUpdate = true;
    streaks.material.opacity = 0.1 + Math.min(ws, 1.2) * 0.35;
  }

  // ---------------------------------------------------------- sites
  const siteObjs = [];
  function crossTexture() {
    const n = 256, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d");
    g.fillStyle = "#e9ece6"; g.beginPath(); g.arc(n / 2, n / 2, n * 0.48, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#6c8f3e"; g.beginPath(); g.arc(n / 2, n / 2, n * 0.42, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#d1262c"; g.fillRect(n * 0.42, n * 0.22, n * 0.16, n * 0.56); g.fillRect(n * 0.22, n * 0.42, n * 0.56, n * 0.16);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
  }
  function beamTexture() {
    const c = document.createElement("canvas"); c.width = 4; c.height = 128;
    const g = c.getContext("2d"), gr = g.createLinearGradient(0, 128, 0, 0);
    gr.addColorStop(0, "rgba(255,255,255,0.9)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 4, 128);
    return new THREE.CanvasTexture(c);
  }
  function building(w, h, d, wall, roofCol, metal = true) {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color: wall, roughness: 0.88 }));
    body.position.y = h / 2;
    const shape = new THREE.Shape([new THREE.Vector2(-d / 2 - 0.04, 0), new THREE.Vector2(d / 2 + 0.04, 0), new THREE.Vector2(0, d * 0.32)]);
    const roof = new THREE.Mesh(new THREE.ExtrudeGeometry(shape, { depth: w + 0.08, bevelEnabled: false }),
      new THREE.MeshStandardMaterial({ color: roofCol, roughness: metal ? 0.3 : 0.8, metalness: metal ? 0.7 : 0 }));
    roof.rotation.y = Math.PI / 2;
    roof.position.set(-w / 2 - 0.04, h, 0);
    const trim = new THREE.Mesh(new THREE.BoxGeometry(w + 0.01, h * 0.22, d + 0.01), new THREE.MeshStandardMaterial({ color: "#2f5d8a", roughness: 0.7 }));
    trim.position.y = h * 0.11;
    g.add(body, roof, trim);
    g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    return g;
  }
  function buildSites() {
    const padTex = crossTexture(), beamTex = beamTexture();
    for (const s of SITES) {
      const g = new THREE.Group();
      g.position.set(s.wx, s.h, s.wz);
      const ground = new THREE.Mesh(new THREE.CircleGeometry(s.hub ? 2.2 : 1.5, 48),
        new THREE.MeshStandardMaterial({ color: "#a48a63", roughness: 1, polygonOffset: true, polygonOffsetFactor: -1 }));
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = 0.02;
      ground.receiveShadow = true;
      g.add(ground);
      if (s.hub) {
        const hangar = building(1.9, 0.55, 1.1, "#d6d2c8", "#a9adb2");
        hangar.position.set(-1.2, 0, -0.6);
        const office = building(0.9, 0.4, 0.6, "#efeee8", "#9c3f2a", false);
        office.position.set(0.9, 0, -1.3);
        const rail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 1.6), new THREE.MeshStandardMaterial({ color: "#3b3f44", metalness: 0.6, roughness: 0.4 }));
        rail.position.set(1.1, 0.25, 0.5); rail.rotation.x = -0.25; rail.castShadow = true;
        const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.75, 0.05, 40), new THREE.MeshStandardMaterial({ color: "#3c3f43", roughness: 0.75 }));
        pad.position.set(0, 0.04, 0.6); pad.receiveShadow = true;
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.62, 48), new THREE.MeshBasicMaterial({ color: "#f2bb07" }));
        ring.rotation.x = -Math.PI / 2; ring.position.set(0, 0.075, 0.6);
        g.add(hangar, office, rail, pad, ring, windsock());
      } else {
        const main = building(1.0, 0.36, 0.55, "#dcd5c6", "#a9adb2");
        main.position.set(-0.9, 0, -0.5);
        const wing = building(0.55, 0.32, 0.45, "#dcd5c6", "#a9adb2");
        wing.rotation.y = Math.PI / 2;
        wing.position.set(-1.25, 0, 0.35);
        const zone = new THREE.Mesh(new THREE.CircleGeometry(0.62, 48),
          new THREE.MeshStandardMaterial({ map: padTex, roughness: 0.9, polygonOffset: true, polygonOffsetFactor: -2 }));
        zone.rotation.x = -Math.PI / 2;
        zone.position.set(0.35, 0.04, 0.35);
        zone.receiveShadow = true;
        g.add(main, wing, zone);
      }
      // request beacon: a soft column of light and a ring on the ground
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.3, 9, 16, 1, true),
        new THREE.MeshBasicMaterial({ color: "#ff3b3b", map: beamTex, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false }));
      beam.position.set(0.35, 4.5, 0.35);
      const pulse = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.74, 48),
        new THREE.MeshBasicMaterial({ color: "#ff3b3b", transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
      pulse.rotation.x = -Math.PI / 2;
      pulse.position.set(s.hub ? 0 : 0.35, 0.09, s.hub ? 0.6 : 0.35);
      const charge = pulse.clone();
      charge.material = pulse.material.clone();
      charge.material.color.set("#4fe08b");
      g.add(beam, pulse, charge);
      const label = new CSS2DObject(s.labelEl);
      label.position.set(0, s.hub ? 2.0 : 1.4, 0);
      g.add(label);
      scene.add(g);
      siteObjs.push({ ...s, group: g, beam, pulse, charge, request: null, chargingUntil: 0 });
    }
  }
  let sock;
  function windsock() {
    const g = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.016, 0.7, 6), new THREE.MeshStandardMaterial({ color: "#d9d9d9", metalness: 0.5, roughness: 0.4 }));
    pole.position.y = 0.35;
    const c = document.createElement("canvas"); c.width = 64; c.height = 8;
    const x = c.getContext("2d");
    for (let i = 0; i < 5; i++) { x.fillStyle = i % 2 ? "#ffffff" : "#ff6a13"; x.fillRect(i * 12.8, 0, 12.8, 8); }
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.06, 0.32, 12, 1, true).rotateZ(-Math.PI / 2).translate(0.16, 0, 0),
      new THREE.MeshStandardMaterial({ map: tex, side: THREE.DoubleSide, roughness: 0.8 }));
    tex.rotation = Math.PI / 2;
    sock = new THREE.Group();
    sock.position.y = 0.68;
    sock.add(cone);
    g.add(pole, sock);
    g.position.set(1.9, 0, -0.4);
    g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    return g;
  }

  // ---------------------------------------------------------- drone
  const drone = new THREE.Group();
  const navLights = [];
  let prop;
  (function buildDrone() {
    const white = new THREE.MeshPhysicalMaterial({ color: "#dedfdc", roughness: 0.35, clearcoat: 0.7, clearcoatRoughness: 0.25 });
    const red = new THREE.MeshPhysicalMaterial({ color: "#c1121f", roughness: 0.4, clearcoat: 0.5 });
    const dark = new THREE.MeshStandardMaterial({ color: "#2a2d31", roughness: 0.5, metalness: 0.3 });
    const pod = new THREE.Mesh(new THREE.CapsuleGeometry(0.12, 0.62, 8, 16).rotateX(Math.PI / 2), white);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.03, 0.3), white);
    wing.position.set(0, 0.1, 0.06);
    const stripe = new THREE.Mesh(new THREE.BoxGeometry(2.11, 0.032, 0.07), red);
    stripe.position.set(0, 0.1, -0.07);
    drone.add(pod, wing, stripe);
    for (const sx of [-0.38, 0.38]) { // twin tail booms with an inverted V tail
      const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.0, 8).rotateX(Math.PI / 2), dark);
      boom.position.set(sx, 0.1, -0.38);
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.025, 0.16), white);
      fin.position.set(sx * 0.62, 0.0, -0.86);
      fin.rotation.z = sx > 0 ? -0.6 : 0.6;
      drone.add(boom, fin);
    }
    prop = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.03, 0.04), dark);
    prop.position.z = 0.5;
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.29, 24), new THREE.MeshBasicMaterial({ color: "#222", transparent: true, opacity: 0.12, depthWrite: false }));
    disc.position.z = 0.505;
    const cargo = new THREE.Group();
    for (let i = 0; i < 2; i++) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.1, 0.18), red);
      box.position.set(i ? 0.09 : -0.09, -0.15, 0.05);
      cargo.add(box);
    }
    drone.userData.cargo = cargo;
    drone.add(prop, disc, cargo);
    const light = (color, x, z) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), new THREE.MeshBasicMaterial({ color }));
      m.position.set(x, 0.1, z);
      m.material.color.multiplyScalar(6); // over the bloom threshold
      drone.add(m);
      navLights.push(m);
      return m;
    };
    light("#ff2020", -1.05, 0.06);
    light("#20ff60", 1.05, 0.06);
    drone.userData.strobe = light("#ffffff", 0, -0.92);
    drone.scale.setScalar(1.15);
    drone.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    scene.add(drone);
  })();

  // ---------------------------------------------------------- flight paths
  const HOVER = 2.6;
  function sitePoint(i) {
    const s = SITES[i];
    return s.hub ? new THREE.Vector3(s.wx, s.h + 0.35, s.wz + 0.6) : new THREE.Vector3(s.wx + 0.35, s.h + HOVER, s.wz + 0.35);
  }
  function flightCurve(from, to, fraction = 1) {
    const a = sitePoint(from), b = sitePoint(to);
    let cruise = 0;
    for (let k = 0; k <= 30; k++) { const p = a.clone().lerp(b, k / 30); cruise = Math.max(cruise, heightAt(p.x, p.z)); }
    cruise += 3.4;
    const pts = [a.clone(), new THREE.Vector3(a.x, Math.max(a.y + 1.4, cruise - 0.8), a.z)];
    for (const t of [0.25, 0.5, 0.75]) { const p = a.clone().lerp(b, t); pts.push(new THREE.Vector3(p.x, cruise, p.z)); }
    pts.push(new THREE.Vector3(b.x, Math.max(b.y + 1.4, cruise - 0.8), b.z), b.clone());
    const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
    if (fraction >= 1) return curve;
    const cut = curve.getPointAt(fraction);
    const fall = new THREE.Vector3(cut.x + 1.4, heightAt(cut.x + 1.4, cut.z) + 0.15, cut.z);
    const sub = [];
    for (let k = 0; k <= 8; k++) sub.push(curve.getPointAt((fraction * k) / 8));
    sub.push(new THREE.Vector3((cut.x + fall.x) / 2, cut.y - 0.6, cut.z), fall);
    return new THREE.CatmullRomCurve3(sub, false, "centripetal");
  }

  // ---------------------------------------------------------- effects
  const effects = [];
  function addEffect(obj, dur, update) { scene.add(obj); effects.push({ obj, t0: performance.now(), dur, update }); }
  function dropParcel(i, emergency) {
    const s = SITES[i];
    const g = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.15, 0.2), new THREE.MeshStandardMaterial({ color: "#c1121f", roughness: 0.6 }));
    const chute = new THREE.Mesh(new THREE.SphereGeometry(0.42, 18, 8, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: emergency ? "#e8333a" : "#f2f2ee", side: THREE.DoubleSide, roughness: 0.7 }));
    chute.position.y = 0.75;
    g.add(box, chute);
    g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    const start = sitePoint(i).clone().add(new THREE.Vector3(0, -0.25, 0));
    const end = new THREE.Vector3(s.wx + 0.4, s.h + 0.12, s.wz + 0.4);
    addEffect(g, 1700, (k) => {
      g.position.lerpVectors(start, end, 1 - (1 - k) ** 1.4);
      g.rotation.y = k * 2.5;
      g.rotation.z = Math.sin(k * 12) * 0.08;
      chute.scale.setScalar(k > 0.92 ? Math.max(1 - (k - 0.92) * 12, 0.01) : Math.min(k * 8, 1));
    });
  }
  function smoke(at) {
    for (let i = 0; i < 18; i++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.3, 10, 8),
        new THREE.MeshStandardMaterial({ color: "#4a4a48", transparent: true, opacity: 0.7, depthWrite: false, roughness: 1 }));
      const dir = new THREE.Vector3(Math.random() - 0.5, 1 + Math.random(), Math.random() - 0.5);
      addEffect(m, 3200, (k) => { m.position.copy(at).addScaledVector(dir, k * 3.2); m.scale.setScalar(1 + k * 3.5); m.material.opacity = 0.7 * (1 - k); });
    }
  }
  function loadParcel() {
    const hub = SITES.find((s) => s.hub);
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, 0.16), new THREE.MeshStandardMaterial({ color: "#c1121f" }));
    const a = new THREE.Vector3(hub.wx - 1.2, hub.h + 0.3, hub.wz - 0.6);
    addEffect(box, 700, (k) => box.position.lerpVectors(a, drone.position, k));
  }

  // ---------------------------------------------------------- state from app.js
  let wind = { speed: 0, dir: 0 };
  function setRequest(i, r) {
    const o = siteObjs[i];
    o.request = r;
    if (r) {
      const col = r.emergency ? "#ff2b2b" : "#ffb300";
      o.beam.material.color.set(col);
      o.pulse.material.color.set(col);
    }
  }
  function setCharging(until) { const hub = siteObjs.find((o) => o.hub); if (hub) hub.chargingUntil = until; }
  function setCargo(n) { drone.userData.cargo.children.forEach((b, i) => (b.visible = i < n)); }
  // env wind_dir is the heading the wind blows towards, in env axes (x east, y north)
  function setWind(speedKmh, dirRad) {
    wind = { speed: speedKmh, dir: dirRad };
    U.uWind.value.set(Math.cos(dirRad), -Math.sin(dirRad));
    U.uWindStrength.value = speedKmh / 40;
  }
  // point the drone along its path, crab into the crosswind and bank through turns
  const _fwd = new THREE.Vector3(), _prevFwd = new THREE.Vector3(0, 0, 1);
  let bank = 0;
  function flyDrone(p, ahead, dt) {
    drone.position.copy(p);
    _fwd.subVectors(ahead, p);
    if (_fwd.lengthSq() < 1e-8) return;
    _fwd.normalize();
    // keep the nose within ~25° of level: climb-outs and descents are shallow, never vertical
    let fx = _fwd.x, fz = _fwd.z, fl = Math.hypot(fx, fz);
    if (fl < 0.2) { fx = _prevFwd.x; fz = _prevFwd.z; fl = Math.hypot(fx, fz) || 1; }
    const pitch = THREE.MathUtils.clamp(Math.atan2(_fwd.y, Math.hypot(_fwd.x, _fwd.z)), -0.45, 0.45);
    _fwd.set((fx / fl) * Math.cos(pitch), Math.sin(pitch), (fz / fl) * Math.cos(pitch));
    drone.lookAt(p.x + _fwd.x, p.y + _fwd.y, p.z + _fwd.z);
    const flat = new THREE.Vector2(_fwd.x, _fwd.z).normalize();
    const cross = flat.x * U.uWind.value.y - flat.y * U.uWind.value.x; // crosswind component
    drone.rotateY(-cross * U.uWindStrength.value * 0.35);
    const turn = Math.atan2(_prevFwd.x * _fwd.z - _prevFwd.z * _fwd.x, _prevFwd.x * _fwd.x + _prevFwd.z * _fwd.z);
    bank += (THREE.MathUtils.clamp(-turn / Math.max(dt, 1e-3) * 0.35, -0.6, 0.6) - bank) * Math.min(dt * 4, 1);
    drone.rotateZ(bank + Math.sin(performance.now() / 260) * 0.03 * U.uWindStrength.value);
    _prevFwd.copy(_fwd);
  }
  function placeDrone(i) { drone.position.copy(sitePoint(i)); drone.rotation.set(0, 0, 0); }

  // ---------------------------------------------------------- build + frame
  function build(sites) {
    SITES = sites.map((s) => {
      let sum = 0, n = 0; // the mean height of the ground around the site, so it sits on its hill
      for (let a = 0; a < 12; a++) for (const r of [0, 1.5, 3]) { sum += rawHeight(s.x + Math.cos(a) * r, -s.y + Math.sin(a) * r); n++; }
      return { ...s, wx: s.x, wz: -s.y, h: Math.max(sum / n, WATER_LEVEL + 0.6) };
    });
    buildTerrain();
    buildWater();
    buildRoads();
    buildVillages();
    buildVegetation();
    buildAtmosphere();
    buildLife();
    buildSites();
    placeDrone(0);
    setTime(0);
    ready = true;
  }

  let slowSince = 0, aoOn = true, startedAt = 0;
  if (new URLSearchParams(location.search).get("quality") === "low") { ao.enabled = false; aoOn = false; composer.setPixelRatio(1); }
  let ready = false;
  function update(now, dt, followDrone) {
    if (!ready) { controls.update(); composer.render(); return; }
    U.uTime.value = now / 1000;
    const ws = U.uWindStrength.value;
    U.uCloudOffset.value.x -= U.uWind.value.x * dt * (0.6 + ws * 2.2);
    U.uCloudOffset.value.y -= U.uWind.value.y * dt * (0.6 + ws * 2.2);
    updateLife(now, dt);
    prop.rotation.z += dt * 45;
    drone.userData.strobe.visible = Math.floor(now / 120) % 9 === 0;
    for (const o of siteObjs) {
      const on = !!o.request;
      const speed = o.request?.emergency ? 5.5 : 2.2;
      const wave = 0.5 + 0.5 * Math.sin((now / 1000) * speed);
      o.beam.material.opacity = on ? 0.35 + 0.4 * wave : 0;
      o.beam.visible = o.pulse.visible = on;
      const ph = (now / 1100) % 1;
      o.pulse.material.opacity = on ? 0.9 * (1 - ph) : 0;
      o.pulse.scale.setScalar(on ? 1 + ph * 2 : 1);
      const charging = now < o.chargingUntil;
      o.charge.material.opacity = charging ? 0.5 + 0.4 * wave : 0;
      o.charge.visible = charging;
    }
    // valley mist burns off by mid-morning and drifts with the wind
    const morning = 1 - smoothstep(30, 170, mistTime);
    const wv = (wind.speed / 40) * dt;
    for (const s of mist) {
      s.material.opacity = s.userData.base * morning;
      s.visible = morning > 0.01;
      s.position.x += Math.cos(wind.dir) * wv * 0.4;
      s.position.z -= Math.sin(wind.dir) * wv * 0.4;
    }
    for (const c of clouds) {
      c.position.x += Math.cos(wind.dir) * wv * 2.2;
      c.position.z -= Math.sin(wind.dir) * wv * 2.2;
      if (c.position.x > 110) c.position.x -= 220; if (c.position.x < -110) c.position.x += 220;
      if (c.position.z > 110) c.position.z -= 220; if (c.position.z < -110) c.position.z += 220;
      c.material.color.copy(sun.color).lerp(new THREE.Color(1, 1, 1), 0.6);
    }
    if (sock) { // the windsock turns to the wind and lifts as it strengthens
      sock.rotation.y = wind.dir;
      sock.rotation.z = -(1 - Math.min(wind.speed / 25, 1)) * 1.1 + Math.sin(now / 180) * 0.04;
    }
    if (waterNormal) waterNormal.offset.set(now * 0.000012 * (1 + wind.speed / 20), now * 0.000008);
    for (let i = effects.length - 1; i >= 0; i--) {
      const fx = effects[i], k = Math.min((now - fx.t0) / fx.dur, 1);
      fx.update(k);
      if (k >= 1) { scene.remove(fx.obj); effects.splice(i, 1); }
    }
    if (followDrone) controls.target.lerp(drone.position, 0.08);
    controls.update();
    composer.render();
    labels.render(scene, camera);
    // drop ambient occlusion if the machine cannot hold ~35 fps
    if (!startedAt) startedAt = now;
    if (dt > 0.029) { if (!slowSince) slowSince = now; } else slowSince = 0;
    if (aoOn && slowSince && now - slowSince > 3000 && now - startedAt > 5000) { ao.enabled = false; aoOn = false; }
  }

  return {
    scene, camera, controls, drone, build, update, setTime, setRequest, setCharging, setCargo, setWind,
    placeDrone, flyDrone, sitePoint, flightCurve, dropParcel, smoke, loadParcel, heightAt,
    toggleOrbit: () => (controls.autoRotate = !controls.autoRotate),
  };
}
