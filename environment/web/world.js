// The 3D world at true scale: real satellite imagery of Muhanga draped over
// real elevation (baked into map/ by scripts/bake_map.py), with the drone hub,
// the six health centres, villages and trees built on it (places.js) and a
// 3.3 m fixed-wing delivery drone that flies real flight plans (flight.js).
// From the map view the drone is drawn larger so it stays visible; Follow
// swaps the map for a third-person chase camera with a sky, clouds and haze,
// and cuts to ground cameras for each launch, drop and landing. app.js drives
// it; nothing here knows about the RL messages beyond site positions.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { Sky } from "three/addons/objects/Sky.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { planFlight, sampleFlight, AGL, AIRSPEED, SLOW, TURN_R } from "./flight.js";
import { buildHub, buildClinic, createScatter, loadModels, sway } from "./places.js";

// World units are kilometres: x east, z south (env y north maps to -z).
const EXAG = 1.0;        // true relief: the hills are as steep as they are
const BASE_M = 1400;     // elevation that sits at y = 0
const KIVU_M = 1462;     // Lake Kivu's surface
const DEG = Math.PI / 180;
const metresToY = (m) => ((m - BASE_M) / 1000) * EXAG;

// uniforms shared by every wind-driven shader
const U = {
  uTime: { value: 0 },
  uWind: { value: new THREE.Vector2(1, 0) },   // world-space direction the wind blows towards
  uWindStrength: { value: 0.3 },               // 0 calm .. 1 = 40 km/h
  uCloudOffset: { value: new THREE.Vector2() },
};
const smoothstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
function rng(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
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
  renderer.toneMappingExposure = 0.9;
  container.appendChild(renderer.domElement);

  const labels = new CSS2DRenderer();
  labels.setSize(window.innerWidth, window.innerHeight);
  Object.assign(labels.domElement.style, { position: "fixed", inset: "0", pointerEvents: "none" });
  container.appendChild(labels.domElement);

  const scene = new THREE.Scene();
  const haze = new THREE.Color("#0b1113");
  scene.background = haze.clone();
  scene.fog = new THREE.FogExp2(haze.clone(), 0.0072);

  const camera = new THREE.PerspectiveCamera(34, window.innerWidth / window.innerHeight, 0.2, 1200);
  camera.position.set(7.3, 28.1, 62.4); // HOME below
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(-2, 2, 16);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minPolarAngle = 0.06 * Math.PI;
  controls.maxPolarAngle = 0.42 * Math.PI;  // look down at the land, never at the horizon
  controls.minDistance = 0.4;
  controls.maxDistance = 95;
  controls.autoRotateSpeed = 0.3;

  // ---------------------------------------------------------- sky (seen only from the chase camera)
  // A gradient dome with the sun and a deck of drifting cumulus. Below the
  // horizon it is the colour of the haze, so the far edge of the map melts
  // into it. uCine blends it in: 0 is the dark map look, 1 the chase look.
  const skyU = {
    uSun: { value: new THREE.Vector3(0, 1, 0) },
    uZenith: { value: new THREE.Color("#2f62a0") },
    uHorizon: { value: new THREE.Color("#a9bccb") },
    uDark: { value: new THREE.Color("#0b1113") },
    uCine: { value: 0 },
    uTime: U.uTime,
    uCloudOffset: U.uCloudOffset,
  };
  const skyDome = new THREE.Mesh(new THREE.SphereGeometry(900, 48, 24), new THREE.ShaderMaterial({
    uniforms: skyU, side: THREE.BackSide, depthWrite: false, fog: false,
    vertexShader: /* glsl */ `varying vec3 vDir; void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      varying vec3 vDir;
      uniform vec3 uSun, uZenith, uHorizon, uDark; uniform float uCine, uTime; uniform vec2 uCloudOffset;
      float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
      float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.-2.*f);
        return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
      float fbm(vec2 p){ float s = 0., a = .5; for (int i = 0; i < 6; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(13.7, 7.3); a *= .5; } return s; }
      void main(){
        vec3 d = normalize(vDir);
        float h = d.y;
        float sd = max(dot(d, normalize(uSun)), 0.);
        float up = pow(clamp(h, 0., 1.), .5);
        vec3 sky = mix(uHorizon, uZenith, up);
        sky += vec3(1., .8, .58) * pow(sd, 5.) * .3 * (1. - up * .7);    // warm haze around the sun
        sky += vec3(1., .92, .8) * pow(sd, 90.) * 1.2;                   // glow
        sky += vec3(1., .96, .9) * smoothstep(.99955, .9998, sd) * 40.;  // the disc, over the bloom threshold
        if (h > 0.) {
          vec2 uv = d.xz / (h + .08) * 1.5 + uCloudOffset * .02 + vec2(uTime * .004, 0.);
          float n = fbm(uv);
          float cov = smoothstep(.47, .72, n) * smoothstep(.0, .22, h);
          float dens = smoothstep(.5, .95, n);
          vec3 cloud = mix(vec3(1.08, 1.06, 1.02), mix(uHorizon, vec3(.5, .54, .6), .6), dens * .85); // bright edges, grey cores
          cloud += vec3(1., .86, .62) * pow(sd, 8.) * .7 * (1. - dens);                              // a silver lining near the sun
          sky = mix(sky, cloud, cov * .92);
        }
        sky = mix(uHorizon, sky, smoothstep(-.015, .04, h)); // below the horizon: the haze the land fades into
        gl_FragColor = vec4(mix(uDark, sky, uCine), 1.);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  }));
  skyDome.renderOrder = -1;
  skyDome.frustumCulled = false;
  skyDome.userData.noAO = true;
  scene.add(skyDome);

  // cumulus over the ridges, soft sprites with flat grey bases
  const clouds = new THREE.Group();
  const cloudMat = new THREE.SpriteMaterial({ transparent: true, depthWrite: false, opacity: 0, fog: true });
  (function buildClouds() {
    const n = 256, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d"), r = rng(5);
    for (let i = 0; i < 52; i++) {
      const a = r() * Math.PI * 2, d = Math.sqrt(r()) * 0.3;
      const x = n / 2 + Math.cos(a) * d * n, y = n * 0.56 + Math.sin(a) * d * n * 0.45 - r() * n * 0.1;
      const rad = n * (0.09 + r() * 0.13);
      const gr = g.createRadialGradient(x, y, 0, x, y, rad);
      gr.addColorStop(0, "rgba(255,255,255,0.55)"); gr.addColorStop(0.6, "rgba(255,255,255,0.24)"); gr.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = gr;
      g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
    }
    g.globalCompositeOperation = "destination-out"; // a flat base
    const cut = g.createLinearGradient(0, n * 0.64, 0, n * 0.76);
    cut.addColorStop(0, "rgba(0,0,0,0)"); cut.addColorStop(1, "rgba(0,0,0,1)");
    g.fillStyle = cut; g.fillRect(0, 0, n, n);
    g.globalCompositeOperation = "source-atop";     // sunlit tops, grey undersides
    const sh = g.createLinearGradient(0, n * 0.2, 0, n * 0.75);
    sh.addColorStop(0, "rgba(255,250,240,0)"); sh.addColorStop(1, "rgba(112,122,138,0.9)");
    g.fillStyle = sh; g.fillRect(0, 0, n, n);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    cloudMat.map = tex;
    const rr = rng(31);
    for (let i = 0; i < 110; i++) { // cumulus with bases near 3,900 m, a couple of kilometres over the ridges
      const a = rr() * Math.PI * 2, dist = 3 + Math.sqrt(rr()) * 60;
      const cl = new THREE.Group();
      cl.position.set(Math.cos(a) * dist, 2.5 + rr() * 0.9, Math.sin(a) * dist);
      const size = 0.7 + rr() * 1.7, k = 5 + Math.floor(rr() * 6);
      for (let j = 0; j < k; j++) {
        const sp = new THREE.Sprite(cloudMat);
        sp.position.set((rr() - 0.5) * size * 1.2, (rr() - 0.3) * size * 0.16, (rr() - 0.5) * size * 1.2);
        const sc = size * (0.55 + rr() * 0.55);
        sp.scale.set(sc, sc * 0.6, 1);
        cl.add(sp);
      }
      clouds.add(cl);
    }
    clouds.visible = false;
    scene.add(clouds);
  })();

  // ---------------------------------------------------------- sun and light
  // The sky is never seen; it only lights the scene through the environment map.
  const envScene = new THREE.Scene();
  const envSky = new Sky();
  envSky.scale.setScalar(100);
  Object.assign(envSky.material.uniforms.turbidity, { value: 4 });
  Object.assign(envSky.material.uniforms.rayleigh, { value: 1.6 });
  envScene.add(envSky);
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null;

  const sun = new THREE.DirectionalLight("#fff1df", 2.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, { left: -40, right: 40, top: 40, bottom: -40, near: 1, far: 260 });
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.03;
  sun.shadow.radius = 3;
  scene.add(sun, sun.target);
  const hemi = new THREE.HemisphereLight("#bcd3e6", "#26301f", 0.9);
  scene.add(hemi);

  const sunDir = new THREE.Vector3();
  const lastEnvDir = new THREE.Vector3(0, -1, 0);
  const warm = new THREE.Color("#ffc58c"), noon = new THREE.Color("#fff4e8");
  const hazeLow = new THREE.Color("#17140f"), hazeHigh = new THREE.Color("#0b1113");
  const hazeNow = hazeHigh.clone();
  // the chase camera's sky: warmer and lower in the morning and afternoon
  const horizonLow = new THREE.Color("#b9b4a8"), horizonHigh = new THREE.Color("#9fb5c9");
  const zenithLow = new THREE.Color("#3a6496"), zenithHigh = new THREE.Color("#2259a0");

  // Muhanga is 2°S; late November the sun's declination is about -20°.
  function setTime(minutes) {
    const hour = 8 + minutes / 60;
    const H = (hour - 12) * 15 * DEG, lat = -2.1 * DEG, dec = -20 * DEG;
    const sinAlt = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(H);
    const alt = Math.asin(sinAlt);
    const az = Math.atan2(Math.sin(H), Math.cos(H) * Math.sin(lat) - Math.tan(dec) * Math.cos(lat)) + Math.PI;
    sunDir.set(Math.cos(alt) * Math.sin(az), Math.sin(alt), -Math.cos(alt) * Math.cos(az)).normalize();
    envSky.material.uniforms.sunPosition.value.copy(sunDir);
    sun.position.copy(sunDir).multiplyScalar(120);
    const k = smoothstep(12, 48, alt / DEG);
    sun.color.copy(warm).lerp(noon, k);
    sun.intensity = 1.9 + 1.0 * k;
    hemi.intensity = 0.75 + 0.25 * k;
    hazeNow.copy(hazeLow).lerp(hazeHigh, k);
    skyU.uHorizon.value.copy(horizonLow).lerp(horizonHigh, k);
    skyU.uZenith.value.copy(zenithLow).lerp(zenithHigh, k);
    skyU.uSun.value.copy(sunDir);
    if (sunDir.angleTo(lastEnvDir) > 1.5 * DEG) {
      if (envRT) envRT.dispose();
      envRT = pmrem.fromScene(envScene, 0, 0.1, 1000);
      scene.environment = envRT.texture;
      lastEnvDir.copy(sunDir);
    }
  }

  // ---------------------------------------------------------- post-processing
  const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
  composer.setPixelRatio(Math.min(dpr, 1.5));
  composer.setSize(window.innerWidth, window.innerHeight);
  composer.addPass(new RenderPass(scene, camera));
  const ao = new GTAOPass(scene, camera, window.innerWidth, window.innerHeight);
  ao.updateGtaoMaterial({ radius: 0.6, distanceExponent: 1.5, thickness: 1.2, scale: 1.0, samples: 12 });
  ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 12 });
  ao.blendIntensity = 0.6;
  // AO must only see solid geometry: sprites, lines and glows would print dark shapes
  const baseHide = ao.overrideVisibility.bind(ao);
  ao.overrideVisibility = function () {
    baseHide();
    this.scene.traverse((o) => { if (o.isSprite || o.isLine || o.userData.noAO || (o.material && o.material.transparent)) o.visible = false; });
  };
  composer.addPass(ao);
  const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.6, 0.55, 2.2) // only lights and glows flare;
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  // Photographic finish, in display space: soft focus toward the edges, cool
  // shadows, a heavy vignette and a little grain.
  const grade = new ShaderPass({
    uniforms: { tDiffuse: { value: null }, uRes: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 }, uCine: { value: 0 } },
    vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse; uniform vec2 uRes; uniform float uTime; uniform float uCine; varying vec2 vUv;
      float hash(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
      void main(){
        vec2 q = (vUv - .5) * vec2(uRes.x / uRes.y, 1.);
        float r = length(q);
        float blur = smoothstep(.42, 1.05, r) * 3.5 * (1. - .85 * uCine);
        vec3 c = texture2D(tDiffuse, vUv).rgb;
        if (blur > .05) {
          vec3 acc = c; float w = 1.;
          for (int i = 0; i < 8; i++) {
            float a = float(i) * .7854 + .39;
            acc += texture2D(tDiffuse, vUv + vec2(cos(a), sin(a)) * blur / uRes).rgb;
            acc += texture2D(tDiffuse, vUv + vec2(cos(a), sin(a)) * blur * .5 / uRes).rgb;
            w += 2.;
          }
          c = acc / w;
        }
        float l = dot(c, vec3(.2126, .7152, .0722));
        c = mix(vec3(l), c, .9 + .14 * uCine);                     // slightly muted on the map
        c = mix(c * vec3(.9, 1., 1.03), c, smoothstep(.0, .45, l)); // cool shadows
        c = pow(max(c, 0.), vec3(1.06));
        c = mix(c, c * c * (3. - 2. * c), .35);                    // a gentle S-curve for contrast
        c *= mix(mix(.28, .52, uCine), 1., smoothstep(1.05, .28, r)); // vignette
        c *= mix(mix(.62, .78, uCine), 1., smoothstep(.0, .22, vUv.y)); // darker under the cards
        c += (hash(vUv * uRes + fract(uTime) * 91.) - .5) * .022;  // grain
        gl_FragColor = vec4(c, 1.);
      }`,
  });
  composer.addPass(grade);
  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
    labels.setSize(window.innerWidth, window.innerHeight);
    grade.uniforms.uRes.value.set(window.innerWidth * composer._pixelRatio, window.innerHeight * composer._pixelRatio);
  }
  window.addEventListener("resize", resize);
  resize();

  // ---------------------------------------------------------- map data and the Blender models
  const modelsReady = loadModels();
  let MAP = null;
  const mapReady = (async () => {
    const meta = await (await fetch("map/map.json")).json();
    const loader = new THREE.TextureLoader();
    const layer = async (L) => {
      const [tex, buf] = await Promise.all([
        loader.loadAsync(`map/${L.image}`),
        fetch(`map/${L.height}`).then((r) => r.arrayBuffer()),
      ]);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
      return { ...L, tex, h: new Uint16Array(buf) };
    };
    const [inner, outer] = await Promise.all([layer(meta.layers.inner), layer(meta.layers.outer)]);
    MAP = { meta, inner, outer };
  })();

  // bilinear elevation (metres) from one layer; x east, z south, in km
  function sample(L, x, z) {
    const n = L.height_px, s = (n - 1) / (2 * L.half_km);
    const fx = Math.min(Math.max((x + L.half_km) * s, 0), n - 1.001);
    const fy = Math.min(Math.max((z + L.half_km) * s, 0), n - 1.001);
    const ix = Math.floor(fx), iy = Math.floor(fy), tx = fx - ix, ty = fy - iy, i = iy * n + ix;
    const a = L.h[i], b = L.h[i + 1], c = L.h[i + n], d = L.h[i + n + 1];
    return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty;
  }
  // the ground graded flat under the hub and each centre, blending back into the hills
  function elevation(x, z) {
    let m = rawElevation(x, z);
    for (const p of pads) {
      const d = Math.hypot(x - p.x, z - p.z);
      if (d < p.r1) m += (p.m - m) * (1 - smoothstep(p.r0, p.r1, d));
    }
    return m;
  }
  function rawElevation(x, z) {
    const inH = MAP.inner.half_km;
    const edge = Math.max(Math.abs(x), Math.abs(z));
    if (edge >= inH) return sample(MAP.outer, x, z);
    const w = smoothstep(inH - 3, inH, edge); // the fine map hands over to the coarse one at its rim
    const fine = sample(MAP.inner, x, z);
    return w > 0 ? fine + (sample(MAP.outer, x, z) - fine) * w : fine;
  }
  const heightAt = (x, z) => metresToY(elevation(x, z));

  // ---------------------------------------------------------- terrain
  let SITES = [];
  const overlay = {
    uSitePos: { value: [] },
    uSiteState: { value: [] },   // x: request 0 none, 1 routine, 2 emergency; y: hub; z: charging; w: destination
    uDrone: { value: new THREE.Vector3(0, 0, 0) },
  };
  const terrainHead = (withOverlay) => /* glsl */ `
    varying vec3 vWPos;
    uniform float uTime; uniform vec2 uWind; uniform float uWindStrength; uniform vec2 uCloudOffset;
    float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
    float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.-2.*f);
      return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
    float cloudShade(vec3 p){
      vec2 q = (p.xz + uCloudOffset) * .03;
      float n = vnoise(q) * .6 + vnoise(q * 2.1 + 7.) * .3 + vnoise(q * 4.3 + 3.) * .1;
      return 1. - .38 * smoothstep(.5, .78, n);
    }
    ${withOverlay ? `
    uniform vec2 uSitePos[${SITES.length}]; uniform vec4 uSiteState[${SITES.length}]; uniform vec3 uDrone;
    float ringLine(float r, float R, float px, float w){ return 1. - smoothstep(px * w * .5, px * (w * .5 + 1.), abs(r - R)); }
    float dashes(vec2 d, float n, float spin){ return step(.42, fract(atan(d.y, d.x) / 6.28318 * n + spin)); }
    // map overlays painted onto the ground, so they follow every hill
    vec3 mapOverlay(vec3 p){
      vec3 e = vec3(0.);
      float px = length(fwidth(p.xz));
      for (int i = 0; i < ${SITES.length}; i++) {
        vec2 d = p.xz - uSitePos[i];
        float r = length(d);
        vec4 st = uSiteState[i];
        float R = st.y > .5 ? 1.6 : 1.15;
        e += vec3(.92, .95, .9) * .55 * ringLine(r, R, px, 1.4) * dashes(d, 44., uTime * .01);
        e += vec3(.75, .85, .7) * .035 * (1. - smoothstep(R - px * 2., R, r));
        if (st.x > .5) {
          vec3 col = st.x > 1.5 ? vec3(1., .3, .07) : vec3(1., .6, .14);
          float pulse = .7 + .3 * sin(uTime * (st.x > 1.5 ? 5.5 : 2.2));
          e += col * (exp(-r * r / .06) * 6. + exp(-r * r / 1.1) * .45) * pulse;
          float ph = fract(uTime / 1.7);
          e += col * ringLine(r, R * (.3 + ph * 1.5), px, 2.) * (1. - ph) * 1.6;
        }
        if (st.z > .5) {
          float ph = fract(uTime / 1.1);
          e += vec3(.3, 1., .55) * ringLine(r, R * (.4 + ph), px, 2.) * (1. - ph) * 1.4;
        }
        if (st.w > .5) e += vec3(1., .84, .5) * .9 * ringLine(r, R * 2.1, px, 1.6) * dashes(d, 70., -uTime * .03);
      }
      vec2 dd = p.xz - uDrone.xz;
      float rd = length(dd);
      e += vec3(1.) * (ringLine(rd, .28, px, 1.6) * .9 + exp(-rd * rd / .004) * 1.2);
      return e;
    }` : ""}`;

  function terrainMaterial(tex, withOverlay) {
    const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.93, metalness: 0, envMapIntensity: 0.35 });
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, U, withOverlay ? overlay : {});
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vWPos;")
        .replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;");
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <common>", "#include <common>\n" + terrainHead(withOverlay))
        // the photo already holds its own light: deepen it toward dark greens, and keep
        // bright roofs and quarries well below white
        .replace("#include <map_fragment>", `#include <map_fragment>
          { vec3 c = diffuseColor.rgb; float l = dot(c, vec3(.2126, .7152, .0722));
            c = max(mix(vec3(l), c, 1.3), 0.); c = pow(c, vec3(1.28)) * 1.7; c = min(c / (1. + c * .8) * 1.3, vec3(.6)); diffuseColor.rgb = c; }
          { // up close the 10 m imagery gets fine variation: plots, furrows, bare patches
            float dc = distance(vWPos, cameraPosition);
            float det = vnoise(vWPos.xz * 260.) * .55 + vnoise(vWPos.xz * 61.) * .45;
            diffuseColor.rgb *= mix(1., .86 + .28 * det, smoothstep(2.5, .25, dc)); }`)
        .replace("#include <lights_fragment_end>", "#include <lights_fragment_end>\nfloat cs = cloudShade(vWPos); reflectedLight.directDiffuse *= cs; reflectedLight.directSpecular *= cs; reflectedLight.indirectDiffuse *= mix(1., cs, .5);")
        .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\n" + (withOverlay ? "totalEmissiveRadiance += mapOverlay(vWPos);" : ""));
    };
    mat.customProgramCacheKey = () => (withOverlay ? `terrain-ov-${SITES.length}` : "terrain");
    return mat;
  }

  function terrainMesh(L, segments, sink) {
    const size = 2 * L.half_km;
    const geo = new THREE.PlaneGeometry(size, size, segments, segments);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    const inH = MAP.inner.half_km;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      let y = heightAt(x, z);
      if (sink && Math.abs(x) < inH - 0.01 && Math.abs(z) < inH - 0.01) y -= 0.15; // hidden under the fine map
      pos.setY(i, y);
    }
    geo.computeVertexNormals();
    return geo;
  }

  function buildTerrain() {
    const inner = new THREE.Mesh(terrainMesh(MAP.inner, 640, false), terrainMaterial(MAP.inner.tex, true));
    inner.receiveShadow = true;
    inner.castShadow = true;
    const outer = new THREE.Mesh(terrainMesh(MAP.outer, 400, true), terrainMaterial(MAP.outer.tex, false));
    outer.receiveShadow = true;
    scene.add(inner, outer);
  }

  // ---------------------------------------------------------- Lake Kivu
  let waterNormal;
  function rippleTexture() {
    const n = 256, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d"), img = g.createImageData(n, n);
    const h = (x, y) => Math.sin(x * 0.19 + Math.sin(y * 0.07) * 2) * 0.5 + Math.sin(y * 0.23 + x * 0.05) * 0.5;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const dx = h(x + 1, y) - h(x - 1, y), dy = h(x, y + 1) - h(x, y - 1);
      const v = new THREE.Vector3(-dx, -dy, 2.4).normalize();
      const i = (y * n + x) * 4;
      img.data[i] = (v.x * 0.5 + 0.5) * 255; img.data[i + 1] = (v.y * 0.5 + 0.5) * 255; img.data[i + 2] = (v.z * 0.5 + 0.5) * 255; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(90, 140);
    return t;
  }
  function buildWater() {
    // the lake lies 45-75 km west; this sheet only shows where the ground is below its surface
    waterNormal = rippleTexture();
    const water = new THREE.Mesh(new THREE.PlaneGeometry(72, 106),
      new THREE.MeshPhysicalMaterial({ color: "#0c1a1f", roughness: 0.16, metalness: 0, normalMap: waterNormal,
        normalScale: new THREE.Vector2(0.25, 0.25), envMapIntensity: 0.9, transparent: true, opacity: 0.82 }));
    water.rotation.x = -Math.PI / 2;
    water.position.set(-64, metresToY(KIVU_M + 6), 5);
    scene.add(water);
  }

  // ---------------------------------------------------------- wind
  let streaks;
  const streakData = [];
  function buildWind() {
    const r = rng(77);
    const c = document.createElement("canvas"); c.width = 64; c.height = 4;
    const g = c.getContext("2d"), gr = g.createLinearGradient(0, 0, 64, 0);
    gr.addColorStop(0, "rgba(255,255,255,0)"); gr.addColorStop(0.7, "rgba(255,255,255,0.9)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 64, 4);
    const N = 220;
    streaks = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 0.025).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, opacity: 0.0, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: true }), N);
    streaks.frustumCulled = false;
    for (let i = 0; i < N; i++) streakData.push({ x: (r() - 0.5) * 60, z: (r() - 0.5) * 60, y: 1.5 + r() * 3.5, g: 0, life: r(), len: 1 + r() * 2 });
    scene.add(streaks);
  }
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
  function updateWind(dt) {
    const wx = U.uWind.value.x, wz = U.uWind.value.y, ws = U.uWindStrength.value;
    _q.setFromAxisAngle(UP, Math.atan2(wx, wz) - Math.PI / 2);
    const cx = mode === "map" ? controls.target.x : drone.position.x, cz = mode === "map" ? controls.target.z : drone.position.z;
    streakData.forEach((sd, i) => {
      sd.life += dt * (0.25 + ws * 0.5);
      sd.x += wx * dt * (2.5 + ws * 10); sd.z += wz * dt * (2.5 + ws * 10);
      if (sd.life > 1 || Math.abs(sd.x - cx) > 30 || Math.abs(sd.z - cz) > 30) {
        sd.life = 0; sd.x = cx + (Math.random() - 0.5) * 60; sd.z = cz + (Math.random() - 0.5) * 60; sd.g = 0;
      }
      if (!sd.g) sd.g = heightAt(sd.x, sd.z);
      const fade = Math.sin(Math.PI * sd.life);
      streaks.setMatrixAt(i, _m.compose(_p.set(sd.x, sd.y + sd.g, sd.z), _q, _s.set(sd.len * (0.4 + ws), 1, Math.max(fade, 0.001))));
    });
    streaks.instanceMatrix.needsUpdate = true;
    streaks.material.opacity = (0.06 + Math.min(ws, 1.2) * 0.22) * (1 - cine); // a map-view device only
  }

  // ---------------------------------------------------------- sites
  const siteObjs = [];
  function glowTexture() {
    const n = 128, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d"), gr = g.createRadialGradient(n / 2, n / 2, 0, n / 2, n / 2, n / 2);
    gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.18, "rgba(255,255,255,0.55)");
    gr.addColorStop(0.5, "rgba(255,255,255,0.12)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, n, n);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  function buildSites() {
    const glowTex = glowTexture();
    SITES.forEach((s, i) => {
      overlay.uSitePos.value.push(new THREE.Vector2(s.wx, s.wz));
      overlay.uSiteState.value.push(new THREE.Vector4(0, s.hub ? 1 : 0, 0, 0));
      // the request glow hangs just above the ground, where bloom turns it into a soft flare
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: "#ff6a1a", transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0, fog: false }));
      glow.position.set(s.wx, s.h + 0.35, s.wz);
      glow.scale.setScalar(1.6);
      scene.add(glow);
      const label = new CSS2DObject(s.labelEl);
      label.center.set(0, 0.5);
      label.position.set(s.wx, s.h + 0.05, s.wz);
      scene.add(label);
      siteObjs.push({ ...s, index: i, glow, request: null, chargingUntil: 0 });
    });
  }

  // ---------------------------------------------------------- drone
  // A high-wing fixed-wing delivery drone: teardrop pod, thin tail boom with a
  // swept fin, and a pylon over the wing carrying a nacelle with a tractor and a
  // pusher propeller. Packages drop by parachute from the belly. Built in
  // metres facing +z, then scaled to the map (wingspan ~1.3 km of map).
  const drone = new THREE.Group();
  const navLights = [];
  const props = [];
  // NACA 4-digit section as a closed loop, chord along -z (leading edge at z = 0)
  function airfoil(chord, thick, camber = 0.02, n = 18) {
    const pts = [];
    for (let k = 0; k <= 2 * n; k++) {
      const up = k <= n, t = up ? 1 - k / n : (k - n) / n;
      const x = (1 - Math.cos(t * Math.PI)) / 2;
      const yt = 5 * thick * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
      const yc = camber * (x < 0.4 ? (2 * 0.4 * x - x * x) / 0.16 : (1 - 0.8 + 2 * 0.4 * x - x * x) / 0.36);
      pts.push([-x * chord, (up ? yc + yt : yc - yt) * chord]);
    }
    pts.pop(); // first and last coincide at the trailing edge
    return pts;
  }
  // loft a list of [u, v] loops placed by place(i, u, v) -> Vector3, capped at both ends
  function loft(stations, place) {
    const m = stations[0].length, pos = [], idx = [];
    stations.forEach((loop, i) => loop.forEach(([u, v]) => { const p = place(i, u, v); pos.push(p.x, p.y, p.z); }));
    for (let i = 0; i < stations.length - 1; i++) for (let k = 0; k < m; k++) {
      const a = i * m + k, b = i * m + ((k + 1) % m), c = a + m, d = b + m;
      idx.push(a, c, b, b, c, d);
    }
    for (const [i, flip] of [[0, false], [stations.length - 1, true]]) {
      const base = i * m;
      for (let k = 1; k < m - 1; k++) flip ? idx.push(base, base + k + 1, base + k) : idx.push(base, base + k, base + k + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }
  // a tapered, swept surface from root to tip along `axis` (unit vector), section in the plane
  function surface({ span, root, tip, thick, sweep = 0, axis, normal, origin, steps = 10, camber = 0.015 }) {
    const stations = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps, round = i === steps ? 0.55 : 1; // the tip closes over a softer, shorter section
      const chord = (root + (tip - root) * t) * round;
      stations.push(airfoil(chord, thick, camber).map(([u, v]) => [u - sweep * t - (1 - round) * tip * 0.2, v * (i === steps ? 0.6 : 1)]));
    }
    const ax = axis.clone(), nm = normal.clone();
    return loft(stations, (i, u, v) => origin.clone().addScaledVector(ax, (span * i) / steps).add(new THREE.Vector3(0, 0, u + 0)).addScaledVector(nm, v));
  }
  (function buildDrone() {
    const body = new THREE.MeshPhysicalMaterial({ color: "#d6d9d7", roughness: 0.36, clearcoat: 0.6, clearcoatRoughness: 0.2, envMapIntensity: 0.5 });
    const grey = new THREE.MeshStandardMaterial({ color: "#2a2d30", roughness: 0.45, metalness: 0.4 });
    const red = new THREE.MeshPhysicalMaterial({ color: "#d1261e", roughness: 0.35, clearcoat: 0.6 });
    const glass = new THREE.MeshPhysicalMaterial({ color: "#101418", roughness: 0.06, metalness: 0.1, clearcoat: 1, envMapIntensity: 1.4 });
    const parts = new THREE.Group();

    // pod: blunt round nose, longest radius just ahead of the middle, slender tail
    const L = 1.3, R = 0.19, prof = [];
    for (let i = 0; i <= 40; i++) {
      const t = i / 40;
      prof.push(new THREE.Vector2(Math.max(R * (t ** 1.05 * (1 - t) ** 0.5) / 0.372, 1e-4), -L / 2 + t * L));
    }
    const podGeo = new THREE.LatheGeometry(prof, 40).rotateX(Math.PI / 2).scale(1, 0.92, 1);
    parts.add(new THREE.Mesh(podGeo, body));
    // sensor dome on the nose and the red release under the belly
    const dome = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.6, 1.4), glass);
    dome.position.set(0, 0.13, 0.42);
    parts.add(dome);
    const release = new THREE.Mesh(new THREE.CapsuleGeometry(0.022, 0.12, 4, 10).rotateX(Math.PI / 2), red);
    release.position.set(0, -0.165, 0.12);
    parts.add(release);

    // wing on top of the pod, slight dihedral and taper
    const wingY = 0.17, wingZ = 0.13;
    for (const side of [1, -1]) {
      const dihedral = 3.5 * DEG;
      const w = surface({ span: 1.65, root: 0.3, tip: 0.17, thick: 0.13, sweep: 0.04,
        axis: new THREE.Vector3(side * Math.cos(dihedral), Math.sin(dihedral), 0), normal: new THREE.Vector3(0, 1, 0),
        origin: new THREE.Vector3(0, wingY, wingZ), steps: 12 });
      if (side > 0) w.index.array.reverse(); // the loft faces outward for -x; mirror the winding for +x
      w.computeVertexNormals();
      parts.add(new THREE.Mesh(w, body));
      // red tip so the silhouette reads from above
      const tipMark = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.012, 0.15), red);
      tipMark.position.set(side * 1.58, wingY + Math.sin(dihedral) * 1.58 + 0.012, wingZ - 0.1);
      parts.add(tipMark);
    }

    // tail boom from the top of the pod's tail, a swept fin and a small stabiliser
    const boomLen = 1.15;
    const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.02, boomLen, 10).rotateX(Math.PI / 2), body);
    boom.position.set(0, 0.06, -0.5 - boomLen / 2 + 0.1);
    parts.add(boom);
    const tailZ = -0.5 - boomLen + 0.1;
    const fin = surface({ span: 0.36, root: 0.26, tip: 0.15, thick: 0.1, sweep: 0.12, camber: 0,
      axis: new THREE.Vector3(0, 1, 0), normal: new THREE.Vector3(1, 0, 0), origin: new THREE.Vector3(0, 0.06, tailZ + 0.27), steps: 6 });
    parts.add(new THREE.Mesh(fin, body));
    for (const side of [1, -1]) {
      const st = surface({ span: 0.3, root: 0.17, tip: 0.1, thick: 0.1, sweep: 0.05, camber: 0,
        axis: new THREE.Vector3(side, 0, 0), normal: new THREE.Vector3(0, 1, 0), origin: new THREE.Vector3(0, 0.06, tailZ + 0.18), steps: 5 });
      if (side > 0) st.index.array.reverse();
      st.computeVertexNormals();
      parts.add(new THREE.Mesh(st, body));
    }
    const skid = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.12, 6), grey);
    skid.position.set(0, 0.0, tailZ + 0.04);
    parts.add(skid);

    // pylon over the wing and the twin-propeller nacelle
    const pylon = surface({ span: 0.3, root: 0.2, tip: 0.16, thick: 0.14, sweep: 0.02, camber: 0,
      axis: new THREE.Vector3(0, 1, 0), normal: new THREE.Vector3(1, 0, 0), origin: new THREE.Vector3(0, wingY + 0.03, wingZ + 0.02), steps: 4 });
    parts.add(new THREE.Mesh(pylon, body));
    const nacY = wingY + 0.33, nacZ = wingZ - 0.05;
    const nacelle = new THREE.Mesh(new THREE.CapsuleGeometry(0.042, 0.34, 6, 16).rotateX(Math.PI / 2), body);
    nacelle.position.set(0, nacY, nacZ);
    parts.add(nacelle);
    const bladeShape = new THREE.Shape();
    bladeShape.moveTo(0, -0.018); bladeShape.quadraticCurveTo(0.1, -0.032, 0.27, -0.012);
    bladeShape.lineTo(0.27, 0.008); bladeShape.quadraticCurveTo(0.1, 0.022, 0, 0.018);
    const blade = new THREE.ExtrudeGeometry(bladeShape, { depth: 0.008, bevelEnabled: false }).translate(0, 0, -0.004);
    for (const [z, dir] of [[nacZ + 0.235, 1], [nacZ - 0.235, -1]]) {
      const hub = new THREE.Group();
      hub.position.set(0, nacY, z);
      const spinner = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.06, 14).rotateX(dir * Math.PI / 2), grey);
      spinner.position.z = dir * 0.02;
      hub.add(spinner);
      for (const a of [0, Math.PI]) {
        const b = new THREE.Mesh(blade, grey);
        b.rotation.z = a;
        b.rotation.y = dir * 0.25; // blade pitch
        hub.add(b);
      }
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.28, 32),
        new THREE.MeshBasicMaterial({ color: "#1a1a1a", transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide }));
      hub.add(disc);
      hub.userData.dir = dir;
      props.push(hub);
      parts.add(hub);
    }

    // packages ride in the belly; one red pod shows per package on board
    const cargo = new THREE.Group();
    for (let i = 0; i < 2; i++) {
      const box = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.1, 4, 10).rotateX(Math.PI / 2).scale(1.2, 0.7, 1), red);
      box.position.set(i ? 0.065 : -0.065, -0.15, -0.08);
      cargo.add(box);
    }
    drone.userData.cargo = cargo;
    parts.add(cargo);

    const light = (color, x, y, z) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.03, 10, 8), new THREE.MeshBasicMaterial({ color }));
      m.position.set(x, y, z);
      m.material.color.multiplyScalar(8); // over the bloom threshold
      parts.add(m);
      navLights.push(m);
      return m;
    };
    const tipY = wingY + Math.sin(3.5 * DEG) * 1.65;
    light("#ff2a20", 1.66, tipY, wingZ - 0.05);   // port
    light("#20ff70", -1.66, tipY, wingZ - 0.05);  // starboard
    drone.userData.strobe = light("#ffffff", 0, 0.43, tailZ + 0.12);
    parts.traverse((o) => { if (o.isMesh && !o.material.transparent) { o.castShadow = true; o.receiveShadow = true; } });
    drone.userData.parts = parts; // modelled in metres; update() scales the drone
    drone.add(parts);
    drone.visible = false;
    scene.add(drone);
  })();

  // the drone as built in Blender replaces the one above: same size, same named parts
  function useDroneModel(model) {
    if (!model) return;
    const parts = drone.userData.parts;
    const front = model.get("prop_front"), rear = model.get("prop_rear"), strobe = model.get("strobe");
    const pods = ["cargo_0", "cargo_1"].map((n) => model.get(n)).filter(Boolean);
    parts.clear();
    props.length = 0;
    const cargo = new THREE.Group();
    for (const o of [...model.scene.children]) if (!o.name.startsWith("cargo_")) parts.add(o);
    for (const pod of pods) cargo.add(pod);
    parts.add(cargo);
    drone.userData.cargo = cargo;
    if (front) { front.userData.dir = 1; props.push(front); }
    if (rear) { rear.userData.dir = -1; props.push(rear); }
    if (strobe) drone.userData.strobe = strobe;
  }

  // ---------------------------------------------------------- facilities, villages and flights
  let hub = null;          // the hub's launcher, wire, charging dock and camera spots (places.js)
  const pads = [];         // ground graded flat under the hub and each centre
  let scatter = null;
  let rate = 1;            // playback speed from app.js, for the loiter, handling and effects
  const DRONE_M = 0.001;   // the drone model is built in metres
  const dockQuat = new THREE.Quaternion(), cradleQuat = new THREE.Quaternion();

  // terrain height exactly as the inner mesh draws it (its triangles, not the
  // smooth heightmap), so buildings, parcels and the camera meet the visible ground
  const SEG = 640;
  function groundAt(x, z) {
    const half = MAP.inner.half_km, s = (2 * half) / SEG;
    if (Math.abs(x) >= half - s || Math.abs(z) >= half - s) return heightAt(x, z);
    const fx = (x + half) / s, fz = (z + half) / s;
    const ix = Math.floor(fx), iz = Math.floor(fz), u = fx - ix, v = fz - iz;
    const x0 = ix * s - half, z0 = iz * s - half;
    const ha = heightAt(x0, z0), hb = heightAt(x0, z0 + s), hc = heightAt(x0 + s, z0 + s), hd = heightAt(x0 + s, z0);
    return u + v <= 1 ? ha + (hd - ha) * u + (hb - ha) * v : hc + (hb - hc) * (1 - u) + (hd - hc) * (1 - v);
  }
  // the satellite image's colour at a point (sRGB 0..1), to grow villages and trees that match it
  let imgData = null;
  const IMG_N = 1024;
  function readImagery() {
    const c = document.createElement("canvas");
    c.width = c.height = IMG_N;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(MAP.inner.tex.image, 0, 0, IMG_N, IMG_N);
    imgData = g.getImageData(0, 0, IMG_N, IMG_N).data;
  }
  function colorAt(x, z, out) {
    if (!imgData) { out[0] = out[1] = out[2] = 0.3; return out; }
    const h = MAP.inner.half_km;
    const px = Math.min(Math.max(Math.floor(((x + h) / (2 * h)) * IMG_N), 0), IMG_N - 1);
    const py = Math.min(Math.max(Math.floor(((z + h) / (2 * h)) * IMG_N), 0), IMG_N - 1);
    const i = (py * IMG_N + px) * 4;
    out[0] = imgData[i] / 255; out[1] = imgData[i + 1] / 255; out[2] = imgData[i + 2] / 255;
    return out;
  }
  // the hub's launch and approach side faces the most open, lowest ground around it
  function openHeading(x, z, y) {
    let best = 0, score = Infinity;
    for (let k = 0; k < 24; k++) {
      const a = (k / 24) * Math.PI * 2, dx = Math.sin(a), dz = Math.cos(a);
      let worst = -Infinity;
      for (let d = 0.5; d <= 3; d += 0.25) worst = Math.max(worst, (heightAt(x + dx * d, z + dz * d) - y) / d);
      if (worst < score) { score = worst; best = a; }
    }
    return best;
  }
  function buildPlaces(models) {
    const h0 = SITES[0];
    hub = buildHub({ x: h0.wx, y: h0.h, z: h0.wz, yaw: openHeading(h0.wx, h0.wz, h0.h) }, models.hub);
    scene.add(hub.group);
    const o = new THREE.Object3D();
    o.position.copy(hub.dock); o.lookAt(hub.dock.clone().add(hub.heading)); dockQuat.copy(o.quaternion);
    const up = hub.heading.clone().multiplyScalar(Math.cos(hub.launchPitch)).setY(Math.sin(hub.launchPitch));
    o.position.copy(hub.cradle); o.lookAt(hub.cradle.clone().add(up)); cradleQuat.copy(o.quaternion);
    for (const s of SITES) {
      if (s.hub) continue;
      s.place = buildClinic({ x: s.wx, y: s.h, z: s.wz, yaw: Math.atan2(h0.wx - s.wx, h0.wz - s.wz), seed: s.id }, models.clinic);
      s.place.drop.y = groundAt(s.place.drop.x, s.place.drop.z);
      scene.add(s.place.group);
    }
    readImagery();
    scatter = createScatter({ scene, groundAt, colorAt, half: MAP.inner.half_km,
      keepOut: pads.map((p) => ({ x: p.x, z: p.z, r: p.r0 })), town: { x: h0.wx, z: h0.wz, r: 2.6 }, models });
  }

  // the planned route, dashed like a map overlay; the destination gets a second ring
  const routeMat = new THREE.LineDashedMaterial({ color: "#ffd58a", dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.85, depthWrite: false, fog: false });
  let route = null;
  function showRoute(f, dest = -1) {
    if (route) { scene.remove(route); route.geometry.dispose(); route = null; }
    overlay.uSiteState.value.forEach((v, i) => (v.w = i === dest ? 1 : 0));
    if (!f) return;
    route = new THREE.Line(new THREE.BufferGeometry().setFromPoints(f.route), routeMat);
    route.computeLineDistances();
    scene.add(route);
  }

  // ---------------------------------------------------------- flying a leg
  const _hd = new THREE.Vector3(), _fp = new THREE.Vector3(), _fa = new THREE.Vector3();
  function headingOf() {
    drone.getWorldDirection(_hd);
    _hd.y = 0;
    if (_hd.lengthSq() < 1e-6) _hd.set(0, 0, 1);
    return _hd.normalize().clone();
  }
  // the ground crew carries the drone at about a metre up, e.g. from the dock to the launcher
  function handlingPath(from, to) {
    const lift = hub.padY + 0.0011;
    const pts = [from.clone(), new THREE.Vector3(from.x, lift, from.z), new THREE.Vector3(to.x, lift, to.z), to.clone()];
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
    const len = cum[cum.length - 1];
    return { pts, cum, len, dur: len < 0.0005 ? 0 : THREE.MathUtils.clamp(len * 40000, 900, 2600) }; // ~25 m/s, time-warped
  }
  function carry(path, k, q0, q1) {
    const d = path.len * Math.min(Math.max(k, 0), 1);
    let i = 1;
    while (i < path.cum.length - 1 && path.cum[i] < d) i++;
    const seg = path.cum[i] - path.cum[i - 1];
    drone.position.lerpVectors(path.pts[i - 1], path.pts[i], seg > 0 ? (d - path.cum[i - 1]) / seg : 1);
    drone.quaternion.slerpQuaternions(q0, q1, smoothstep(0, 1, k));
    bank = 0;
  }
  function planLeg({ from, to, drop, crash }) {
    const src = SITES[from], dest = SITES[to];
    const o = { crashAt: crash ?? null };
    const fromHub = src.hub && from !== to;
    if (fromHub) o.launch = { start: hub.cradle, end: hub.railEnd };
    else o.start = { pos: drone.position.clone(), heading: headingOf() };
    if (from === to) o.crashAt = 0;
    else if (dest.hub) Object.assign(o, { kind: "land", target: hub.wire, axis: hub.axis, finalY: hub.wire.y });
    else if (drop) Object.assign(o, { kind: "drop", target: dest.place.drop, finalY: dest.place.drop.y + AGL.drop, exitY: dest.h + AGL.loiter });
    else Object.assign(o, { kind: "visit", target: { x: dest.wx, z: dest.wz }, finalY: dest.h + AGL.loiter });
    const f = planFlight(o, groundAt);
    f.from = from; f.to = to;
    if (fromHub) { f.pre = handlingPath(drone.position, hub.cradle); f.dur += f.pre.dur; }
    return f;
  }
  let following = null, idle = { site: 0 }, handling = null, frameDt = 1 / 60;
  let carriage = null, swing = null, wreck = null;
  // physics that runs in real seconds whatever the drone is doing
  function physics(dtr) {
    if (carriage) { // the carriage hits the end stop, rebounds, then the winch pulls it back down the rail
      carriage.t += dtr;
      const back = smoothstep(2, 7, carriage.t);
      const k = 1 - 0.05 * Math.exp(-5 * carriage.t) * Math.abs(Math.sin(16 * carriage.t)) - back;
      hub.carriage.position.lerpVectors(hub.rail0L, hub.rail1L, Math.max(k, 0));
      if (carriage.t > 7) carriage = null;
    }
    if (swing) { // a damped pendulum on the wire: θ'' = -(g/L) sin θ - c θ'
      for (let n = 0, h = Math.min(dtr, 0.2) / 4; n < 4; n++) { swing.w += (-(9.81 / 1.2) * Math.sin(swing.th) - 1.1 * swing.w) * h; swing.th += swing.w * h; }
      if (Math.abs(swing.th) < 0.002 && Math.abs(swing.w) < 0.01 && (!following || following.kind !== "land")) swing = null;
    }
    if (wreck) { // a dead-stick arrival: it hits, bounces once and comes to rest on a wingtip
      wreck.t += dtr;
      const k = Math.min(wreck.t / 1.1, 1);
      drone.quaternion.slerpQuaternions(wreck.q0, wreck.q1, smoothstep(0, 1, k));
      drone.position.y = wreck.y + 0.0007 * Math.abs(Math.sin(k * Math.PI * 1.5)) * (1 - k);
    }
  }
  function followFlight(f, tMs) {
    if (following !== f) {
      following = f; idle = null; handling = null; f.lastT = -1;
      f.preQ = drone.quaternion.clone();
    }
    let t = tMs;
    if (f.pre && f.pre.dur > 0) {
      if (t < f.pre.dur) { carry(f.pre, t / f.pre.dur, f.preQ, cradleQuat); return { frac: 0, events: [] }; }
      t -= f.pre.dur;
    }
    const st = sampleFlight(f, t, _fp, _fa);
    flyDrone(_fp, _fa, frameDt, st.bank);
    if (f.railLen) {
      if (st.onRail) { hub.carriage.position.lerpVectors(hub.rail0L, hub.rail1L, st.railK); carriage = null; }
      else if (!carriage) carriage = { t: 0 };
    }
    if (f.tCapture != null && t >= f.tCapture && !swing) swing = { th: 0, w: 2.4 }; // the hook takes the wire: the drone pitches forward
    if (swing) drone.rotateX(swing.th);
    for (const e of st.events) onFlightEvent(e, f);
    return st;
  }
  function onFlightEvent(type, f) {
    const now = performance.now();
    if (type === "launch") startShot("launch", f);
    if (type === "climbed" && shot?.kind === "launch") shot.until = now + 700;
    if (type === "approach") startShot(f.kind === "land" ? "land" : "drop", f);
    if (type === "drop") {
      const parcel = dropParcel(drone.position, _fa.clone().sub(_fp).normalize(), !!siteObjs[f.to]?.request?.emergency);
      if (shot?.kind === "drop") { shot.subject = parcel; shot.size = 0.003; }
    }
    if (type === "capture" && shot?.kind === "land") shot.until = now + 2600;
    if (type === "crash") {
      smoke(drone.position.clone());
      const rest = new THREE.Object3D();
      rest.position.copy(drone.position);
      rest.lookAt(drone.position.clone().add(headingOf()));
      rest.rotateZ((Math.random() < 0.5 ? -1 : 1) * 0.45);
      rest.rotateX(0.2);
      wreck = { t: 0, q0: drone.quaternion.clone(), q1: rest.quaternion.clone(), y: groundAt(drone.position.x, drone.position.z) + 0.0004 };
    }
  }
  // between legs: circle a centre at 90 m, or sit in the hub's charging dock
  function idleAt(loc) {
    following = null;
    if (idle && idle.site === loc) return;
    idle = { site: loc };
    if (SITES[loc].hub) handling = { path: handlingPath(drone.position, hub.dock), t: -1400, q0: drone.quaternion.clone() };
    else apHead.copy(headingOf());
  }
  function placeDrone() {
    following = null; handling = null; idle = { site: 0 }; swing = null; wreck = null;
    drone.position.copy(hub.dock);
    drone.quaternion.copy(dockQuat);
    bank = 0;
  }
  const apHead = new THREE.Vector3(0, 0, 1), _lt = new THREE.Vector3(), _lp = new THREE.Vector3(), _la = new THREE.Vector3();
  function autopilot(dt) {
    if (following || !idle) return;
    const s = SITES[idle.site];
    if (s.hub) {
      if (!handling) return;
      handling.t += dt * 1000 * rate;
      if (handling.t < 0) { if (swing) { drone.quaternion.copy(handling.q0); drone.rotateX(swing.th); } return; } // swinging on the wire until the crew reaches it
      swing = null;
      const k = handling.path.dur ? handling.t / handling.path.dur : 1;
      carry(handling.path, k, handling.q0, dockQuat);
      if (k >= 1) handling = null;
      return;
    }
    const R = 0.25, p = drone.position, v = AIRSPEED * SLOW * 0.001 * rate;
    const th = Math.atan2(p.z - s.wz, p.x - s.wx);
    _lt.set(s.wx + Math.cos(th + 0.45) * R - p.x, 0, s.wz + Math.sin(th + 0.45) * R - p.z);
    const cur = Math.atan2(apHead.x, apHead.z), want = Math.atan2(_lt.x, _lt.z);
    const maxRate = (AIRSPEED / 1000 / TURN_R) * SLOW * rate;
    const turn = THREE.MathUtils.clamp(wrapAngle(want - cur), -maxRate * dt, maxRate * dt);
    apHead.set(Math.sin(cur + turn), 0, Math.cos(cur + turn));
    _lp.copy(p).addScaledVector(apHead, v * dt);
    _lp.y += (s.h + AGL.loiter - p.y) * Math.min(dt * 0.5 * rate, 1);
    const omegaReal = turn / Math.max(dt, 1e-3) / (SLOW * rate);
    flyDrone(_lp, _la.copy(_lp).addScaledVector(apHead, 0.02), dt, THREE.MathUtils.clamp(Math.atan((AIRSPEED * omegaReal) / 9.81), -0.6, 0.6));
  }

  // ---------------------------------------------------------- effects (built in metres, simulated in real seconds)
  const effects = [];
  function addEffect(obj, step, minScale = 0.004) {
    obj.scale.setScalar(DRONE_M);
    obj.userData.minScale = minScale;
    scene.add(obj);
    effects.push({ obj, step });
  }
  // the package leaves the belly, falls free for a moment, the paper parachute
  // opens, and it drifts down with the wind at about 5 m/s onto the drop zone
  function dropParcel(at, dir, emergency) {
    const g = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.3, 0.3), new THREE.MeshStandardMaterial({ color: "#c7ae84", roughness: 0.9 }));
    const band = new THREE.Mesh(new THREE.BoxGeometry(0.43, 0.31, 0.06), new THREE.MeshStandardMaterial({ color: emergency ? "#d1261e" : "#2f6fb0", roughness: 0.7 }));
    const chute = new THREE.Mesh(new THREE.SphereGeometry(0.75, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2.4).scale(1, 0.6, 1),
      new THREE.MeshStandardMaterial({ color: emergency ? "#d8332c" : "#e8742e", side: THREE.DoubleSide, roughness: 0.85 })); // bright, to be seen against cloud
    chute.position.y = 2;
    const lp = [];
    for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) lp.push(new THREE.Vector3(sx * 0.18, 0.15, sz * 0.13), new THREE.Vector3(sx * 0.5, 2.08, sz * 0.5));
    const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lp), new THREE.LineBasicMaterial({ color: "#d9d6cf" }));
    const body = new THREE.Group(); // swings under the canopy
    body.add(box, band, lines);
    g.add(body, chute);
    g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    g.position.copy(at);
    g.rotation.y = Math.atan2(dir.x, dir.z);
    chute.scale.setScalar(0.01);
    lines.visible = false;
    // 1.5 kg package, paper canopy of 1 m² with Cd 1.5 at ~1 kg/m³: terminal velocity about 4.4 m/s
    const MASS = 1.5, RHO = 1.0, CDA_BOX = 0.06, CDA_CHUTE = 1.5;
    const vel = dir.clone().multiplyScalar(AIRSPEED), rel = new THREE.Vector3();
    let t = 0, landed = null, tilt = 0, tiltV = 0;
    addEffect(g, (dtr) => {
      if (!landed) {
        for (let n = 0, h = Math.min(dtr, 0.25) / 6; n < 6; n++) {
          t += h;
          const open = smoothstep(0.4, 1.3, t);
          const windX = U.uWind.value.x * (wind.speed / 3.6), windZ = U.uWind.value.y * (wind.speed / 3.6);
          rel.set(vel.x - windX, vel.y, vel.z - windZ);
          const sp = rel.length(), k = (0.5 * RHO * (CDA_BOX + CDA_CHUTE * open) * sp) / MASS;
          vel.x -= rel.x * k * h;
          vel.z -= rel.z * k * h;
          vel.y += (-9.81 - rel.y * k) * h;
          g.position.addScaledVector(vel, h * 0.001);
          // the package hangs below the canopy and swings as the drag pulls it back: a damped pendulum
          const lean = Math.atan2(Math.hypot(rel.x, rel.z), Math.max(-rel.y, 0.5)) * open;
          tiltV += (-(9.81 / 1.9) * (tilt - lean) - 0.9 * tiltV) * h;
          tilt += tiltV * h;
        }
        const op = smoothstep(0.4, 1.3, t);
        chute.scale.set(Math.max(op, 0.01), Math.max(op * (0.8 + 0.2 * Math.sin(t * 9)), 0.01), Math.max(op, 0.01)); // the paper canopy breathes as it fills
        lines.visible = op > 0.2;
        const yaw = Math.atan2(rel.x, rel.z) - g.rotation.y;
        body.rotation.set(-Math.cos(yaw) * tilt * 0.6, 0, Math.sin(yaw) * tilt * 0.6);
        const gy = groundAt(g.position.x, g.position.z) + 0.00015;
        if (g.position.y <= gy) {
          g.position.y = gy;
          const drift = Math.hypot(vel.x, vel.z);
          landed = { t, drift, dirX: drift > 0.01 ? vel.x / drift : 1, dirZ: drift > 0.01 ? vel.z / drift : 0 };
          lines.visible = false;
          if (shot?.kind === "drop") shot.until = performance.now() + 1800;
        }
      } else {
        t += dtr;
        const k = smoothstep(0, 1.4, t - landed.t);
        // the box tips over if it came down drifting; the empty canopy collapses and is dragged downwind
        const tip = landed.drift > 1.2 ? Math.PI / 2 : 0.12;
        body.rotation.set(0, 0, 0);
        body.rotateOnWorldAxis(new THREE.Vector3(landed.dirZ, 0, -landed.dirX), tip * smoothstep(0, 0.5, t - landed.t));
        chute.position.set(Math.sin(-g.rotation.y) * 0, 2 - 1.95 * k, 0);
        chute.position.x = 1.6 * k;
        chute.rotation.z = -1.45 * k;
        if (t - landed.t > 240) return false;
      }
      return true;
    }, 0.004);
    return g;
  }
  function smoke(at) {
    for (let i = 0; i < 26; i++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 10, 8),
        new THREE.MeshStandardMaterial({ color: i < 4 ? "#7a4a2a" : "#3a3a38", transparent: true, opacity: 0.75, depthWrite: false, roughness: 1 }));
      const dir = new THREE.Vector3(Math.random() - 0.5, 1 + Math.random() * 1.5, Math.random() - 0.5);
      const delay = Math.random() * 4;
      let t = -delay;
      addEffect(m, (dtr) => {
        t += dtr;
        const k = Math.max(t, 0) / 14;
        m.visible = t > 0;
        m.position.copy(at).addScaledVector(dir, k * 0.03).add(new THREE.Vector3(U.uWind.value.x, 0, U.uWind.value.y).multiplyScalar(k * 0.02 * (wind.speed / 20)));
        m.scale.setScalar(m.userData.s * (1.5 + k * 9));
        m.material.opacity = 0.75 * (1 - k);
        return k < 1;
      }, 0.003);
    }
  }
  function loadParcel() {
    const box = new THREE.Group();
    box.add(new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.3, 0.3), new THREE.MeshStandardMaterial({ color: "#c7ae84", roughness: 0.9 })));
    const a = hub.supply.clone(), b = drone.position.clone();
    let t = 0;
    addEffect(box, (dtr) => {
      t += dtr / 4; // carried over in about four seconds
      box.position.lerpVectors(a, b, smoothstep(0, 1, t));
      box.position.y += Math.sin(Math.min(t, 1) * Math.PI) * 0.0005;
      return t < 1;
    }, 0.004);
  }

  // ---------------------------------------------------------- state from app.js
  let wind = { speed: 0, dir: 0 };
  function setRequest(i, r) {
    const o = siteObjs[i];
    o.request = r;
    overlay.uSiteState.value[i].x = r ? (r.emergency ? 2 : 1) : 0;
    if (r) o.glow.material.color.set(r.emergency ? "#ff4a14" : "#ffa21f").multiplyScalar(3.5); // over the bloom threshold
  }
  let charging = false;
  function setCharging(on) { charging = on; }
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
  function flyDrone(p, ahead, dt, bankTarget = null) {
    drone.position.copy(p);
    _fwd.subVectors(ahead, p);
    if (_fwd.lengthSq() < 1e-12) return;
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
    const want = bankTarget ?? THREE.MathUtils.clamp(-turn / Math.max(dt, 1e-3) * 0.9, -0.55, 0.55);
    bank += (want - bank) * Math.min(dt * 5, 1); // roll in and out over a fraction of a second
    drone.rotateZ(bank + Math.sin(performance.now() / 260) * 0.03 * U.uWindStrength.value); // and bumps in gusty air
    drone.rotateX(-0.05); // flies a few degrees nose-up: the wing's angle of attack
    _prevFwd.copy(_fwd);
  }

  // ---------------------------------------------------------- camera moves and the destination card
  // framed so every site sits between the left tools and the dock on a 16:10 screen
  const HOME = { target: new THREE.Vector3(-2, 0.4, 16), position: new THREE.Vector3(7.3, 26.5, 62.4) };
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let camMove = null;
  function moveCamera(toTarget, toPosition) {
    camMove = { t0: performance.now(), dur: reduceMotion ? 0 : 1100, fromT: controls.target.clone(), fromP: camera.position.clone(), toT: toTarget, toP: toPosition };
  }
  function viewFrom(target, distance) {
    const dir = camera.position.clone().sub(controls.target).normalize();
    return target.clone().addScaledVector(dir, THREE.MathUtils.clamp(distance, controls.minDistance, controls.maxDistance));
  }
  function focusSite(i) {
    const p = new THREE.Vector3(SITES[i].wx, SITES[i].h, SITES[i].wz);
    if (mode !== "map") { const yaw = chase.head + chase.orbit; return startOut(p, p.clone().add(new THREE.Vector3(-Math.sin(yaw) * 10, 12, -Math.cos(yaw) * 10))); }
    moveCamera(p, viewFrom(p, 16));
  }
  function resetView() { if (mode !== "map") return startOut(HOME.target.clone(), HOME.position.clone()); moveCamera(HOME.target.clone(), HOME.position.clone()); }
  function zoomBy(f) {
    if (mode !== "map") { chase.dist = THREE.MathUtils.clamp(chase.dist * f, 0.006, 0.25); return; }
    moveCamera(controls.target.clone(), viewFrom(controls.target, camera.position.distanceTo(controls.target) * f));
  }

  // ---------------------------------------------------------- chase camera
  // Third person behind the drone. The camera swings round behind it through
  // turns, drifts into a slow orbit while it waits on the ground, pulls back
  // and widens with speed, and can be dragged to look around (it drifts back)
  // or scrolled closer. Turning Follow on swoops down from the map; turning it
  // off climbs back up.
  const MAP_FOV = 34, CHASE_FOV = 52;
  let mode = "map";   // map | in | chase | out
  let cine = 0;       // 0 the map look .. 1 the chase look
  let trans = null;
  // distances in km: the camera rides about 16 m behind the drone
  const chase = { head: 0, orbit: 0, lookYaw: 0, lookPitch: 0, dist: 0.016, speed: 0, last: new THREE.Vector3(),
    pos: new THREE.Vector3(), off: new THREE.Vector3(), lastInput: 0, dragging: false, primed: false };
  const _cPos = new THREE.Vector3(), _cTgt = new THREE.Vector3(), _cQuat = new THREE.Quaternion(), _cMat = new THREE.Matrix4();
  const _cDir = new THREE.Vector3(), _roll = new THREE.Quaternion(), _ctrl = new THREE.Vector3(), _q2 = new THREE.Quaternion(), _Z = new THREE.Vector3(0, 0, 1);
  const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  const ease = (k) => (k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2);
  function bezier(out, a, b, c, t) {
    const u = 1 - t;
    return out.set(u * u * a.x + 2 * u * t * b.x + t * t * c.x, u * u * a.y + 2 * u * t * b.y + t * t * c.y, u * u * a.z + 2 * u * t * b.z + t * t * c.z);
  }
  function setFov(f) { if (Math.abs(camera.fov - f) > 0.01) { camera.fov = f; camera.updateProjectionMatrix(); } }

  function chasePose(dt, now) {
    const moved = (drone.position.distanceTo(chase.last) * 1000) / Math.max(dt, 1e-3); // m/s on screen
    chase.last.copy(drone.position);
    chase.speed += (Math.min(moved, 1500) - chase.speed) * Math.min(dt * 3, 1);
    drone.getWorldDirection(_cDir);
    if (Math.hypot(_cDir.x, _cDir.z) > 0.1) {
      const heading = Math.atan2(_cDir.x, _cDir.z);
      if (!chase.primed) chase.head = heading;
      chase.head += wrapAngle(heading - chase.head) * (1 - Math.exp(-dt * 2.2));
    }
    if (chase.speed < 3) chase.orbit += dt * 0.12;                         // sitting in the dock: a slow orbit
    else chase.orbit = wrapAngle(chase.orbit) * Math.exp(-dt * 1.4);       // flying: back behind it
    if (!chase.dragging && now - chase.lastInput > 1600) {
      chase.lookYaw = wrapAngle(chase.lookYaw) * Math.exp(-dt * 1.2);
      chase.lookPitch *= Math.exp(-dt * 1.2);
    }
    const fast = Math.min(chase.speed / 600, 1);
    const yaw = chase.head + chase.orbit + chase.lookYaw;
    const pitch = THREE.MathUtils.clamp(0.2 + chase.lookPitch, -0.05, 1.2);
    const dist = chase.dist * (1 + fast * 0.5);
    const fx = Math.sin(yaw), fz = Math.cos(yaw), d = drone.position;
    _cPos.set(d.x - fx * Math.cos(pitch) * dist, d.y + Math.sin(pitch) * dist, d.z - fz * Math.cos(pitch) * dist);
    const ws = U.uWindStrength.value, s = now / 1000; // a little buffeting in the wind
    _cPos.x += (Math.sin(s * 1.9) * 0.6 + Math.sin(s * 3.7) * 0.4) * 0.00015 * ws;
    _cPos.y += (Math.sin(s * 2.3 + 1) * 0.6 + Math.sin(s * 4.1) * 0.4) * 0.00012 * ws;
    // smooth the camera's offset from the drone, not its position, so it never falls behind at cruise speed
    _cPos.sub(d);
    if (!chase.primed) { chase.off.copy(_cPos); chase.primed = true; }
    chase.off.lerp(_cPos, 1 - Math.exp(-dt * 6));
    chase.pos.copy(d).add(chase.off);
    // never inside a hill or a roof: clear the ground under the camera and between it and the drone
    let ground = -Infinity;
    for (const t of [0.35, 0.7, 1]) ground = Math.max(ground, groundAt(d.x + (chase.pos.x - d.x) * t, d.z + (chase.pos.z - d.z) * t));
    chase.pos.y = Math.max(chase.pos.y, ground + 0.002);
    _cTgt.set(d.x + fx * dist * 1.5, d.y - dist * 0.04, d.z + fz * dist * 1.5); // look past the drone, toward the horizon
    _cMat.lookAt(chase.pos, _cTgt, UP);
    _cQuat.setFromRotationMatrix(_cMat).multiply(_roll.setFromAxisAngle(_Z, -bank * 0.22));
    return CHASE_FOV + fast * 8;
  }
  // cinematic shots while chasing: beside the launcher for the launch, at the
  // drop zone for the drop (then following the parachute down), beside the
  // recovery masts for the catch. Drag to leave one early.
  let shot = null;
  function startShot(kind, f) {
    if (mode !== "chase" || chase.dragging) return;
    const pos = kind === "launch" ? hub.cams.launch : kind === "land" ? hub.cams.land : SITES[f.to].place?.cams.drop;
    if (!pos) return;
    // the ground the action ends on stays in frame: the drop zone, or the wire and its masts
    const zone = kind === "drop" ? SITES[f.to].place?.drop.clone() : kind === "land" ? hub.wire.clone().setY(hub.padY) : hub.cradle.clone();
    shot = { kind, pos: pos.clone(), zone, look: drone.position.clone(), subject: drone, size: 0.0033, until: Infinity, t0: performance.now(), fov: camera.fov };
    shot.pos.y = Math.max(shot.pos.y, groundAt(shot.pos.x, shot.pos.z) + 0.0015);
  }
  const _sv1 = new THREE.Vector3(), _sv2 = new THREE.Vector3(), _sAim = new THREE.Vector3();
  function runShot(now, dt) {
    const p = shot.subject.position;
    _sAim.copy(p).lerp(shot.zone, 0.35); // aim between the subject and the ground it is heading for
    shot.look.lerp(_sAim, 1 - Math.exp(-dt * 7));
    camera.position.copy(shot.pos);
    camera.up.set(0, 1, 0);
    camera.lookAt(shot.look);
    const d = Math.max(shot.pos.distanceTo(p), 0.001);
    const span = _sv1.copy(p).sub(shot.pos).angleTo(_sv2.copy(shot.zone).sub(shot.pos)) / DEG; // both must fit
    const want = THREE.MathUtils.clamp(Math.max(((2 * Math.atan(shot.size / 2 / d)) / DEG) * 6, span * 1.5), 6, 58);
    shot.fov += (want - shot.fov) * Math.min(dt * 3, 1);
    setFov(shot.fov);
    if (now > shot.until || now - shot.t0 > 25000 || d > 2) shot = null;
  }
  function startIn() {
    mode = "in";
    controls.enabled = false;
    camMove = null;
    Object.assign(chase, { primed: false, orbit: 0, lookYaw: 0, lookPitch: 0, speed: 0 });
    chase.last.copy(drone.position);
    trans = { t0: performance.now(), dur: reduceMotion ? 0 : 3200, fromP: camera.position.clone(), fromQ: camera.quaternion.clone(), fromFov: camera.fov };
  }
  function startOut(toT, toP) {
    shot = null;
    if (mode === "map") return moveCamera(toT, toP);
    mode = "out";
    trans = { t0: performance.now(), dur: reduceMotion ? 0 : 2600, fromP: camera.position.clone(), fromQ: camera.quaternion.clone(), fromFov: camera.fov, toT, toP };
  }
  function setFollow(on) {
    if (on && (mode === "map" || mode === "out")) startIn();
    else if (!on && (mode === "in" || mode === "chase")) {
      const t = drone.position.clone(), yaw = chase.head + chase.orbit;
      t.x = THREE.MathUtils.clamp(t.x, -LIMIT, LIMIT);
      t.z = THREE.MathUtils.clamp(t.z, -LIMIT, LIMIT);
      startOut(t, t.clone().add(new THREE.Vector3(-Math.sin(yaw) * 24, 22, -Math.cos(yaw) * 24)));
    }
  }
  // drag to look around the drone, scroll to move closer or further
  {
    const el = renderer.domElement;
    let lx = 0, ly = 0;
    el.addEventListener("pointerdown", (e) => {
      if (mode !== "chase" && mode !== "in") return;
      shot = null;
      chase.dragging = true; lx = e.clientX; ly = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!chase.dragging) return;
      chase.lookYaw -= (e.clientX - lx) * 0.006;
      chase.lookPitch = THREE.MathUtils.clamp(chase.lookPitch + (e.clientY - ly) * 0.004, -0.3, 0.95);
      lx = e.clientX; ly = e.clientY;
      chase.lastInput = performance.now();
    });
    const end = () => { chase.dragging = false; chase.lastInput = performance.now(); };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("wheel", (e) => {
      if (mode !== "chase") return;
      chase.dist = THREE.MathUtils.clamp(chase.dist * Math.exp(e.deltaY * 0.0012), 0.006, 0.25);
    }, { passive: true });
  }
  let popup = null;
  function attachPopup(el) {
    popup = new CSS2DObject(el);
    popup.center.set(0.5, 1); // the card stands on the site, its knob pointing down at the pin
    popup.visible = false;
    scene.add(popup);
  }
  function showPopup(i) {
    if (!popup) return;
    popup.visible = i != null && i >= 0;
    if (popup.visible) popup.position.set(SITES[i].wx, SITES[i].h + 0.05, SITES[i].wz);
  }

  // ---------------------------------------------------------- build + frame
  let ready = false;
  async function build(sites) {
    await mapReady;
    SITES = sites.map((s) => ({ ...s, wx: s.x, wz: -s.y }));
    // grade the ground flat under the hub (and its approach) and under each compound and drop zone
    for (const s of SITES) pads.push({ x: s.wx, z: s.wz, r0: s.hub ? 0.2 : 0.16, r1: s.hub ? 0.45 : 0.38, m: elevation(s.wx, s.wz) });
    for (const s of SITES) s.h = heightAt(s.wx, s.wz);
    buildTerrain();
    buildWater();
    buildWind();
    buildSites();
    const models = await modelsReady;
    useDroneModel(models.drone);
    buildPlaces(models);
    placeDrone();
    drone.visible = true;
    setTime(0);
    ready = true;
  }

  let slowSince = 0, aoOn = true, startedAt = 0, shadowNear = null;
  if (new URLSearchParams(location.search).get("quality") === "low") { ao.enabled = false; aoOn = false; composer.setPixelRatio(1); resize(); }
  const LIMIT = 40;
  const GREEN = new THREE.Color("#46ff8a"), DIM = new THREE.Color("#1d3326"), WHITE = new THREE.Color(1, 1, 1);
  function aimSock(sock, parentYaw, ws) {
    sock.rotation.y = Math.atan2(-U.uWind.value.y, U.uWind.value.x) - parentYaw;
    sock.rotation.z = -(1 - Math.min(ws * 1.6, 1)) * 1.1; // limp in a calm, straight out in a breeze
  }
  function update(now, dt) {
    frameDt = dt;
    grade.uniforms.uTime.value = now / 1000;
    if (!ready) { controls.update(); composer.render(); return; }
    U.uTime.value = now / 1000;
    const ws = U.uWindStrength.value;
    U.uCloudOffset.value.x -= U.uWind.value.x * dt * (0.01 + ws * 0.03);
    U.uCloudOffset.value.y -= U.uWind.value.y * dt * (0.01 + ws * 0.03);
    updateWind(dt);
    autopilot(dt);
    physics(dt * SLOW * rate);
    sway.uTime.value = now / 1000;
    sway.uWind.value.set(U.uWind.value.x, U.uWind.value.y).multiplyScalar(Math.min(wind.speed / 30, 1.2));
    for (const p of props) p.rotation.z += dt * 55 * p.userData.dir;
    drone.userData.strobe.visible = Math.floor(now / 120) % 9 === 0;
    overlay.uDrone.value.copy(drone.position);
    for (const o of siteObjs) {
      const on = !!o.request;
      const wave = 0.5 + 0.5 * Math.sin((now / 1000) * (o.request?.emergency ? 5.5 : 2.2));
      o.glow.material.opacity = on ? (0.55 + 0.45 * wave) * (1 - cine) : 0;
      o.glow.visible = on && cine < 0.98;
      o.glow.scale.setScalar(1.3 + 0.5 * wave);
      overlay.uSiteState.value[o.index].z = o.hub && charging ? 1 : 0;
    }
    // the charging dock: a green light and strip that breathe while the battery fills
    const pulse = 0.55 + 0.45 * Math.sin(now / 260);
    hub.chargeLight.material.color.copy(charging ? GREEN : DIM).multiplyScalar(charging ? 2.2 + 2.5 * pulse : 1);
    hub.dockStrip.material.color.copy(charging ? GREEN : DIM).multiplyScalar(charging ? 1.6 + 2.4 * pulse : 1);
    aimSock(hub.sock, hub.group.rotation.y, ws);
    for (const s of SITES) if (s.place) aimSock(s.place.sock, s.place.group.rotation.y, ws);
    if (waterNormal) waterNormal.offset.set(now * 0.000012 * (1 + wind.speed / 20), now * 0.000008);
    if (mode === "map") {
      if (camMove) {
        const k = camMove.dur ? Math.min((now - camMove.t0) / camMove.dur, 1) : 1;
        const e = ease(k);
        controls.target.lerpVectors(camMove.fromT, camMove.toT, e);
        camera.position.lerpVectors(camMove.fromP, camMove.toP, e);
        if (k >= 1) camMove = null;
      }
      controls.target.x = THREE.MathUtils.clamp(controls.target.x, -LIMIT, LIMIT);
      controls.target.z = THREE.MathUtils.clamp(controls.target.z, -LIMIT, LIMIT);
      controls.update();
      cine = 0;
    } else if (mode === "out") {
      const k = trans.dur ? Math.min((now - trans.t0) / trans.dur, 1) : 1, e = ease(k);
      _ctrl.set(trans.fromP.x, trans.toP.y * 0.8 + trans.fromP.y * 0.2, trans.fromP.z); // climb first, then glide over
      bezier(camera.position, trans.fromP, _ctrl, trans.toP, e);
      _cMat.lookAt(trans.toP, trans.toT, UP);
      camera.quaternion.slerpQuaternions(trans.fromQ, _q2.setFromRotationMatrix(_cMat), e);
      setFov(trans.fromFov + (MAP_FOV - trans.fromFov) * e);
      cine = 1 - e;
      if (k >= 1) {
        mode = "map";
        controls.target.copy(trans.toT);
        camera.position.copy(trans.toP);
        controls.enabled = true;
        trans = null;
        controls.update();
      }
    } else {
      const fov = chasePose(dt, now);
      if (mode === "in") {
        const k = trans.dur ? Math.min((now - trans.t0) / trans.dur, 1) : 1, e = ease(k);
        _ctrl.set(chase.pos.x, trans.fromP.y * 0.75 + chase.pos.y * 0.25, chase.pos.z); // cross over high, then dive in behind
        bezier(camera.position, trans.fromP, _ctrl, chase.pos, e);
        camera.quaternion.slerpQuaternions(trans.fromQ, _cQuat, ease(Math.min(k * 1.15, 1)));
        setFov(trans.fromFov + (fov - trans.fromFov) * e);
        cine = e;
        if (k >= 1) { mode = "chase"; trans = null; }
      } else if (shot) {
        runShot(now, dt);
        cine = 1;
      } else {
        camera.position.copy(chase.pos);
        camera.quaternion.copy(_cQuat);
        setFov(fov);
        cine = 1;
      }
    }
    // the drone at true size up close; from far away it stays big enough to see, riding above the ground
    const camD = camera.position.distanceTo(drone.position);
    const dsc = Math.max(DRONE_M, camD * 0.0075);
    drone.scale.setScalar(dsc);
    drone.userData.parts.position.y = ((dsc - DRONE_M) * 0.9) / dsc;
    // effects run in real seconds: take-off, drops and landings play at 2.5x real time
    for (let i = effects.length - 1; i >= 0; i--) {
      const fx = effects[i];
      const s = Math.max(DRONE_M, camera.position.distanceTo(fx.obj.position) * fx.obj.userData.minScale);
      fx.obj.userData.s = s;
      fx.obj.scale.setScalar(s);
      if (!fx.step(dt * SLOW * rate)) { scene.remove(fx.obj); effects.splice(i, 1); }
    }
    // near plane and shadows follow the scale of what the camera is looking at
    const camGround = groundAt(camera.position.x, camera.position.z);
    const near = shot ? 0.0012 : mode === "map"
      ? THREE.MathUtils.clamp(Math.min(camera.position.distanceTo(controls.target) * 0.01, (camera.position.y - camGround) * 0.3), 0.002, 0.2)
      : THREE.MathUtils.clamp(camD * 0.25, 0.0012, 0.2);
    if (Math.abs(camera.near - near) > near * 0.05) { camera.near = near; camera.updateProjectionMatrix(); }
    const local = cine > 0.5 || camera.position.y - camGround < 3;
    if (local !== shadowNear) {
      shadowNear = local;
      const e = local ? 3 : 40;
      Object.assign(sun.shadow.camera, { left: -e, right: e, top: e, bottom: -e, near: local ? 0.05 : 1, far: local ? 30 : 260 });
      sun.shadow.camera.updateProjectionMatrix();
      sun.shadow.normalBias = local ? 0.0012 : 0.03;
      sun.shadow.bias = local ? -0.00012 : -0.0003;
    }
    const focus = local ? (mode === "map" ? controls.target : shot ? shot.look : drone.position) : null;
    if (focus) { sun.target.position.copy(focus); sun.position.copy(focus).addScaledVector(sunDir, 12); }
    else { sun.target.position.set(0, 0, 0); sun.position.copy(sunDir).multiplyScalar(120); }
    scatter.update(camera.position, cine > 0.05 || camera.position.y - camGround < 4);
    // the look: dark aerial map at 0, sky, clouds and blue haze at 1
    skyU.uCine.value = cine;
    skyU.uDark.value.copy(hazeNow);
    scene.fog.color.copy(hazeNow).lerp(skyU.uHorizon.value, cine);
    scene.fog.density = 0.0072 + (0.017 - 0.0072) * cine;
    scene.background.copy(scene.fog.color);
    clouds.visible = cine > 0.01;
    cloudMat.opacity = 0.92 * cine;
    cloudMat.color.copy(sun.color).lerp(WHITE, 0.6);
    for (const cl of clouds.children) {
      cl.position.x -= U.uWind.value.x * dt * (0.004 + ws * 0.01);
      cl.position.z -= U.uWind.value.y * dt * (0.004 + ws * 0.01);
      if (Math.hypot(cl.position.x, cl.position.z) > 64) cl.position.multiplyScalar(-0.98);
    }
    renderer.toneMappingExposure = 0.9 + 0.05 * cine;
    grade.uniforms.uCine.value = cine;
    skyDome.position.copy(camera.position);
    composer.render();
    labels.render(scene, camera);
    // drop ambient occlusion if the machine cannot hold ~35 fps
    if (!startedAt) startedAt = now;
    if (dt > 0.029) { if (!slowSince) slowSince = now; } else slowSince = 0;
    if (aoOn && slowSince && now - slowSince > 3000 && now - startedAt > 5000) { ao.enabled = false; aoOn = false; }
  }

  return {
    scene, camera, controls, renderer, composer, drone, build, update, setTime, setRequest, setCharging, setCargo, setWind,
    placeDrone, planLeg, followFlight, idleAt, showRoute, loadParcel, heightAt,
    attachPopup, showPopup, focusSite, resetView, zoomBy, setFollow,
    setRate: (r) => { rate = r; },
    isReady: () => ready,
    isChasing: () => mode !== "map",
    toggleOrbit: () => (controls.autoRotate = !controls.autoRotate),
  };
}
