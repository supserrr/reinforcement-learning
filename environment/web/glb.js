// A small loader for the binary glTF files that scripts/blender/build_assets.py
// writes: meshes with positions, normals and uvs, metal-rough materials with an
// optional colour texture, emissive strength, and the node tree with names.
// That is all those files use, so this stays independent of the three.js
// version the page bundles. Leaf and fence materials ("leaf_*", "alpha_*")
// become alpha-tested and two-sided, the way foliage cards are drawn.
import * as THREE from "three";

const TYPES = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const NORM = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };

export async function loadGLB(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error(`${url} is not a GLB file`);
  let json = null, bin = null;
  for (let off = 12; off < buf.byteLength;) {
    const len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, off + 8, len)));
    else if (type === 0x004e4942) bin = buf.slice(off + 8, off + 8 + len);
    off += 8 + len;
  }

  const view = (i) => { const bv = json.bufferViews[i]; return { data: bin, offset: bv.byteOffset || 0, length: bv.byteLength, stride: bv.byteStride || 0 }; };
  function accessor(i) {
    const a = json.accessors[i], Arr = TYPES[a.componentType], n = SIZES[a.type];
    const bv = view(a.bufferView), elem = n * Arr.BYTES_PER_ELEMENT;
    let out;
    if (!bv.stride || bv.stride === elem) out = new Arr(bv.data, bv.offset + (a.byteOffset || 0), a.count * n);
    else { // interleaved: copy it out
      out = new Arr(a.count * n);
      const src = new DataView(bv.data, bv.offset + (a.byteOffset || 0));
      const get = { 5126: "getFloat32", 5125: "getUint32", 5123: "getUint16", 5121: "getUint8", 5122: "getInt16", 5120: "getInt8" }[a.componentType];
      for (let k = 0; k < a.count; k++) for (let c = 0; c < n; c++) out[k * n + c] = src[get](k * bv.stride + c * Arr.BYTES_PER_ELEMENT, true);
    }
    if (a.normalized && NORM[a.componentType]) out = Float32Array.from(out, (v) => v / NORM[a.componentType]);
    return { array: out, itemSize: n };
  }

  const textures = await Promise.all((json.textures || []).map(async (t) => {
    const img = json.images[t.source];
    const bv = view(img.bufferView);
    const blob = new Blob([new Uint8Array(bv.data, bv.offset, bv.length)], { type: img.mimeType });
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const tex = new THREE.Texture(bitmap);
    tex.flipY = false;
    const s = json.samplers?.[t.sampler] || {};
    const wrap = { 33071: THREE.ClampToEdgeWrapping, 33648: THREE.MirroredRepeatWrapping, 10497: THREE.RepeatWrapping };
    tex.wrapS = wrap[s.wrapS] ?? THREE.RepeatWrapping;
    tex.wrapT = wrap[s.wrapT] ?? THREE.RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
  }));

  const materials = (json.materials || []).map((m) => {
    const pbr = m.pbrMetallicRoughness || {};
    const name = m.name || "";
    const mat = new THREE.MeshStandardMaterial({
      name,
      color: new THREE.Color().fromArray(pbr.baseColorFactor || [1, 1, 1, 1]),
      roughness: pbr.roughnessFactor ?? 1,
      metalness: pbr.metallicFactor ?? 1,
      side: m.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
    });
    if (pbr.baseColorTexture) {
      mat.map = textures[pbr.baseColorTexture.index];
      mat.map.colorSpace = THREE.SRGBColorSpace;
    }
    if (m.emissiveFactor) {
      const k = m.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? 1;
      mat.emissive = new THREE.Color().fromArray(m.emissiveFactor).multiplyScalar(k);
    }
    if (/^(leaf|alpha)_/.test(name)) { mat.alphaTest = 0.5; mat.transparent = false; mat.side = THREE.DoubleSide; }
    else if (m.alphaMode === "BLEND") mat.transparent = true;
    else if (m.alphaMode === "MASK") mat.alphaTest = m.alphaCutoff ?? 0.5;
    return mat;
  });

  const meshes = (json.meshes || []).map((m) => m.primitives.map((p) => {
    const g = new THREE.BufferGeometry();
    for (const [attr, name] of [["POSITION", "position"], ["NORMAL", "normal"], ["TEXCOORD_0", "uv"], ["COLOR_0", "color"]]) {
      if (p.attributes[attr] == null) continue;
      const a = accessor(p.attributes[attr]);
      g.setAttribute(name, new THREE.BufferAttribute(a.array, a.itemSize));
    }
    if (p.indices != null) g.setIndex(new THREE.BufferAttribute(accessor(p.indices).array, 1));
    return { geometry: g, material: materials[p.material] || new THREE.MeshStandardMaterial() };
  }));

  const nodes = json.nodes.map((n) => {
    let o;
    const prims = n.mesh != null ? meshes[n.mesh] : [];
    if (prims.length === 1) o = new THREE.Mesh(prims[0].geometry, prims[0].material);
    else { o = new THREE.Group(); for (const p of prims) o.add(new THREE.Mesh(p.geometry, p.material)); }
    o.name = n.name || "";
    if (n.matrix) new THREE.Matrix4().fromArray(n.matrix).decompose(o.position, o.quaternion, o.scale);
    if (n.translation) o.position.fromArray(n.translation);
    if (n.rotation) o.quaternion.fromArray(n.rotation);
    if (n.scale) o.scale.fromArray(n.scale);
    return o;
  });
  json.nodes.forEach((n, i) => (n.children || []).forEach((c) => nodes[i].add(nodes[c])));
  const scene = new THREE.Group();
  for (const i of json.scenes[json.scene || 0].nodes) scene.add(nodes[i]);
  scene.traverse((o) => { if (o.isMesh) { o.castShadow = !o.material.transparent; o.receiveShadow = true; } });
  return { scene, get: (name) => scene.getObjectByName(name) };
}
