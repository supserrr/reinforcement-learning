// The 3D world: real satellite imagery of Muhanga draped over real elevation
// (baked into map/ by scripts/bake_map.py), graded dark like an aerial
// photograph at dusk, with map-style markers for the hub and the six health
// centres and a fixed-wing delivery drone. app.js drives it; nothing here
// knows about the RL messages beyond site positions.
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

// World units are kilometres: x east, z south (env y north maps to -z).
const EXAG = 4.5;        // vertical exaggeration, so 1.5 km of relief reads from 50 km away
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
  controls.minDistance = 3.5;
  controls.maxDistance = 95;
  controls.autoRotateSpeed = 0.3;

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
    scene.fog.color.copy(hazeLow).lerp(hazeHigh, k);
    scene.background.copy(scene.fog.color);
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
    this.scene.traverse((o) => { if (o.isSprite || o.isLine || (o.material && o.material.transparent)) o.visible = false; });
  };
  composer.addPass(ao);
  const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.6, 0.55, 2.2) // only lights and glows flare;
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  // Photographic finish, in display space: soft focus toward the edges, cool
  // shadows, a heavy vignette and a little grain.
  const grade = new ShaderPass({
    uniforms: { tDiffuse: { value: null }, uRes: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 } },
    vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse; uniform vec2 uRes; uniform float uTime; varying vec2 vUv;
      float hash(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
      void main(){
        vec2 q = (vUv - .5) * vec2(uRes.x / uRes.y, 1.);
        float r = length(q);
        float blur = smoothstep(.42, 1.05, r) * 3.5;
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
        c = mix(vec3(l), c, .9);                                   // slightly muted
        c = mix(c * vec3(.9, 1., 1.03), c, smoothstep(.0, .45, l)); // cool shadows
        c = pow(max(c, 0.), vec3(1.06));
        c = mix(c, c * c * (3. - 2. * c), .35);                    // a gentle S-curve for contrast
        c *= mix(.28, 1., smoothstep(1.05, .28, r));               // vignette
        c *= mix(.62, 1., smoothstep(.0, .22, vUv.y));             // darker under the cards
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

  // ---------------------------------------------------------- map data
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
  function elevation(x, z) {
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
            c = max(mix(vec3(l), c, 1.3), 0.); c = pow(c, vec3(1.28)) * 1.7; c = min(c / (1. + c * .8) * 1.3, vec3(.6)); diffuseColor.rgb = c; }`)
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
      if (sink && Math.abs(x) < inH - 0.01 && Math.abs(z) < inH - 0.01) y -= 0.6; // hidden under the fine map
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
    const cx = controls.target.x, cz = controls.target.z;
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
    streaks.material.opacity = 0.06 + Math.min(ws, 1.2) * 0.22;
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
    parts.scale.setScalar(0.4);
    drone.add(parts);
    drone.visible = false;
    scene.add(drone);
  })();

  // ---------------------------------------------------------- flight paths
  const HOVER = 1.0, CLEAR = 1.4;
  function sitePoint(i) {
    const s = SITES[i];
    return new THREE.Vector3(s.wx, s.h + (s.hub ? 0.22 : HOVER), s.wz);
  }
  function flightCurve(from, to, fraction = 1) {
    const a = sitePoint(from), b = sitePoint(to);
    let cruise = 0;
    for (let k = 0; k <= 40; k++) { const p = a.clone().lerp(b, k / 40); cruise = Math.max(cruise, heightAt(p.x, p.z)); }
    cruise += CLEAR;
    const pts = [a.clone(), new THREE.Vector3(a.x, Math.max(a.y + 0.7, cruise - 0.4), a.z)];
    for (const t of [0.25, 0.5, 0.75]) { const p = a.clone().lerp(b, t); pts.push(new THREE.Vector3(p.x, cruise, p.z)); }
    pts.push(new THREE.Vector3(b.x, Math.max(b.y + 0.7, cruise - 0.4), b.z), b.clone());
    const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
    if (fraction >= 1) return curve;
    const cut = curve.getPointAt(fraction);
    const fall = new THREE.Vector3(cut.x + 0.8, heightAt(cut.x + 0.8, cut.z) + 0.05, cut.z);
    const sub = [];
    for (let k = 0; k <= 8; k++) sub.push(curve.getPointAt((fraction * k) / 8));
    sub.push(new THREE.Vector3((cut.x + fall.x) / 2, cut.y - 0.4, cut.z), fall);
    return new THREE.CatmullRomCurve3(sub, false, "centripetal");
  }
  // the planned route, dashed like a map overlay; the destination gets a second ring
  const routeMat = new THREE.LineDashedMaterial({ color: "#ffd58a", dashSize: 0.32, gapSize: 0.22, transparent: true, opacity: 0.9, depthWrite: false, fog: false });
  let route = null;
  function showRoute(curve, dest = -1) {
    if (route) { scene.remove(route); route.geometry.dispose(); route = null; }
    overlay.uSiteState.value.forEach((v, i) => (v.w = i === dest ? 1 : 0));
    if (!curve) return;
    route = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getSpacedPoints(200)), routeMat);
    route.computeLineDistances();
    scene.add(route);
  }

  // ---------------------------------------------------------- effects
  const effects = [];
  function addEffect(obj, dur, update) { scene.add(obj); effects.push({ obj, t0: performance.now(), dur, update }); }
  function dropParcel(i, emergency) {
    const s = SITES[i];
    const g = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.06, 0.08), new THREE.MeshStandardMaterial({ color: "#d1261e", roughness: 0.6 }));
    const chute = new THREE.Mesh(new THREE.SphereGeometry(0.17, 18, 8, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: emergency ? "#e8333a" : "#f2f2ee", side: THREE.DoubleSide, roughness: 0.7 }));
    chute.position.y = 0.3;
    g.add(box, chute);
    g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    const start = sitePoint(i).clone().add(new THREE.Vector3(0, -0.1, 0));
    const end = new THREE.Vector3(s.wx + 0.05, s.h + 0.04, s.wz + 0.05);
    addEffect(g, 1700, (k) => {
      g.position.lerpVectors(start, end, 1 - (1 - k) ** 1.4);
      g.rotation.y = k * 2.5;
      g.rotation.z = Math.sin(k * 12) * 0.08;
      chute.scale.setScalar(k > 0.92 ? Math.max(1 - (k - 0.92) * 12, 0.01) : Math.min(k * 8, 1));
    });
  }
  function smoke(at) {
    for (let i = 0; i < 18; i++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8),
        new THREE.MeshStandardMaterial({ color: "#3a3a38", transparent: true, opacity: 0.7, depthWrite: false, roughness: 1 }));
      const dir = new THREE.Vector3(Math.random() - 0.5, 1 + Math.random(), Math.random() - 0.5);
      addEffect(m, 3200, (k) => { m.position.copy(at).addScaledVector(dir, k * 1.6); m.scale.setScalar(1 + k * 3.5); m.material.opacity = 0.7 * (1 - k); });
    }
  }
  function loadParcel() {
    const hub = SITES.find((s) => s.hub);
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.05, 0.07), new THREE.MeshStandardMaterial({ color: "#d1261e" }));
    const a = new THREE.Vector3(hub.wx - 0.5, hub.h + 0.05, hub.wz - 0.3);
    addEffect(box, 700, (k) => box.position.lerpVectors(a, drone.position, k));
  }

  // ---------------------------------------------------------- state from app.js
  let wind = { speed: 0, dir: 0 };
  function setRequest(i, r) {
    const o = siteObjs[i];
    o.request = r;
    overlay.uSiteState.value[i].x = r ? (r.emergency ? 2 : 1) : 0;
    if (r) o.glow.material.color.set(r.emergency ? "#ff4a14" : "#ffa21f").multiplyScalar(3.5); // over the bloom threshold
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

  // ---------------------------------------------------------- camera moves and the destination card
  // framed so every site sits between the left tools and the dock on a 16:10 screen
  const HOME = { target: new THREE.Vector3(-2, 2, 16), position: new THREE.Vector3(7.3, 28.1, 62.4) };
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let camMove = null;
  function moveCamera(toTarget, toPosition) {
    camMove = { t0: performance.now(), dur: reduceMotion ? 0 : 1100, fromT: controls.target.clone(), fromP: camera.position.clone(), toT: toTarget, toP: toPosition };
  }
  function viewFrom(target, distance) {
    const dir = camera.position.clone().sub(controls.target).normalize();
    return target.clone().addScaledVector(dir, THREE.MathUtils.clamp(distance, controls.minDistance, controls.maxDistance));
  }
  function focusSite(i) { const p = new THREE.Vector3(SITES[i].wx, SITES[i].h, SITES[i].wz); moveCamera(p, viewFrom(p, 16)); }
  function resetView() { moveCamera(HOME.target.clone(), HOME.position.clone()); }
  function zoomBy(f) { moveCamera(controls.target.clone(), viewFrom(controls.target, camera.position.distanceTo(controls.target) * f)); }
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
    SITES = sites.map((s) => {
      const wx = s.x, wz = -s.y;
      return { ...s, wx, wz, h: heightAt(wx, wz) };
    });
    buildTerrain();
    buildWater();
    buildWind();
    buildSites();
    placeDrone(0);
    drone.visible = true;
    setTime(0);
    ready = true;
  }

  let slowSince = 0, aoOn = true, startedAt = 0;
  if (new URLSearchParams(location.search).get("quality") === "low") { ao.enabled = false; aoOn = false; composer.setPixelRatio(1); resize(); }
  const LIMIT = 40;
  function update(now, dt, followDrone) {
    grade.uniforms.uTime.value = now / 1000;
    if (!ready) { controls.update(); composer.render(); return; }
    U.uTime.value = now / 1000;
    const ws = U.uWindStrength.value;
    U.uCloudOffset.value.x -= U.uWind.value.x * dt * (0.6 + ws * 2.2);
    U.uCloudOffset.value.y -= U.uWind.value.y * dt * (0.6 + ws * 2.2);
    updateWind(dt);
    for (const p of props) p.rotation.z += dt * 55 * p.userData.dir;
    drone.userData.strobe.visible = Math.floor(now / 120) % 9 === 0;
    overlay.uDrone.value.copy(drone.position);
    for (const o of siteObjs) {
      const on = !!o.request;
      const wave = 0.5 + 0.5 * Math.sin((now / 1000) * (o.request?.emergency ? 5.5 : 2.2));
      o.glow.material.opacity = on ? 0.55 + 0.45 * wave : 0;
      o.glow.visible = on;
      o.glow.scale.setScalar(1.3 + 0.5 * wave);
      overlay.uSiteState.value[o.index].z = now < o.chargingUntil ? 1 : 0;
    }
    if (waterNormal) waterNormal.offset.set(now * 0.000012 * (1 + wind.speed / 20), now * 0.000008);
    for (let i = effects.length - 1; i >= 0; i--) {
      const fx = effects[i], k = Math.min((now - fx.t0) / fx.dur, 1);
      fx.update(k);
      if (k >= 1) { scene.remove(fx.obj); effects.splice(i, 1); }
    }
    if (camMove) {
      const k = camMove.dur ? Math.min((now - camMove.t0) / camMove.dur, 1) : 1;
      const e = k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2;
      controls.target.lerpVectors(camMove.fromT, camMove.toT, e);
      camera.position.lerpVectors(camMove.fromP, camMove.toP, e);
      if (k >= 1) camMove = null;
    }
    if (followDrone) controls.target.lerp(drone.position, 0.08);
    controls.target.x = THREE.MathUtils.clamp(controls.target.x, -LIMIT, LIMIT);
    controls.target.z = THREE.MathUtils.clamp(controls.target.z, -LIMIT, LIMIT);
    controls.update();
    composer.render();
    labels.render(scene, camera);
    // drop ambient occlusion if the machine cannot hold ~35 fps
    if (!startedAt) startedAt = now;
    if (dt > 0.029) { if (!slowSince) slowSince = now; } else slowSince = 0;
    if (aoOn && slowSince && now - slowSince > 3000 && now - startedAt > 5000) { ao.enabled = false; aoOn = false; }
  }

  return {
    scene, camera, controls, renderer, composer, drone, build, update, setTime, setRequest, setCharging, setCargo, setWind,
    placeDrone, flyDrone, sitePoint, flightCurve, showRoute, dropParcel, smoke, loadParcel, heightAt,
    attachPopup, showPopup, focusSite, resetView, zoomBy,
    isReady: () => ready,
    toggleOrbit: () => (controls.autoRotate = !controls.autoRotate),
  };
}
