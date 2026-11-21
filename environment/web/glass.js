// Liquid glass: the map behind a card bends at the card's rounded edges, as
// through a thick lens, with a little colour fringing. Each card gets an SVG
// filter whose displacement map is drawn for its size and corner radius.
// Only Chromium can use an SVG filter as a backdrop-filter; other browsers keep
// the frosted glass from style.css.

const SVG_NS = "http://www.w3.org/2000/svg";
const chromium = !!navigator.userAgentData?.brands?.some((b) => /Chromium/.test(b.brand));
let defs = null;

function ensureDefs() {
  if (defs) return defs;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("aria-hidden", "true");
  Object.assign(svg.style, { position: "absolute", width: "0", height: "0", overflow: "hidden" });
  defs = document.createElementNS(SVG_NS, "defs");
  svg.append(defs);
  document.body.append(svg);
  return defs;
}

// R and G hold the x and y offset (128 = none); near the rim they point inward,
// so the backdrop is pulled from inside the card and appears to bend round its edge
function edgeMap(w, h, radius, bezel) {
  const s = 0.5; // half resolution; feImage stretches it back
  const cw = Math.max(2, Math.round(w * s)), ch = Math.max(2, Math.round(h * s));
  const c = document.createElement("canvas");
  c.width = cw; c.height = ch;
  const g = c.getContext("2d"), img = g.createImageData(cw, ch), d = img.data;
  const r = Math.min(radius, w / 2, h / 2);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const px = (x + 0.5) / s - w / 2, py = (y + 0.5) / s - h / 2;
    const qx = Math.abs(px) - (w / 2 - r), qy = Math.abs(py) - (h / 2 - r);
    const ox = Math.max(qx, 0), oy = Math.max(qy, 0);
    const inside = r - Math.hypot(ox, oy) - Math.min(Math.max(qx, qy), 0); // distance in from the edge
    let nx = 0, ny = 0, k = 0;
    if (inside < bezel) {
      if (qx > 0 && qy > 0) { const l = Math.hypot(ox, oy) || 1; nx = (ox / l) * Math.sign(px); ny = (oy / l) * Math.sign(py); }
      else if (qx > qy) nx = Math.sign(px);
      else ny = Math.sign(py);
      const t = 1 - Math.max(inside, 0) / bezel;
      k = t * t * t;
    }
    const i = (y * cw + x) * 4;
    d[i] = 128 - nx * k * 127;
    d[i + 1] = 128 - ny * k * 127;
    d[i + 2] = 128;
    d[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

function filterFor(id, w, h, radius) {
  const f = document.createElementNS(SVG_NS, "filter");
  f.id = id;
  for (const [k, v] of Object.entries({ x: 0, y: 0, width: w, height: h, filterUnits: "userSpaceOnUse",
    primitiveUnits: "userSpaceOnUse", "color-interpolation-filters": "sRGB" })) f.setAttribute(k, v);
  const scale = 46;
  const channel = (name, s, row) => `
    <feDisplacementMap in="SourceGraphic" in2="map" scale="${s}" xChannelSelector="R" yChannelSelector="G" result="d${name}"/>
    <feColorMatrix in="d${name}" type="matrix" values="${row} 0 0 0 1 0" result="${name}"/>`;
  f.innerHTML = `
    <feImage href="${edgeMap(w, h, radius, 22)}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none" result="map"/>
    ${channel("r", scale, "1 0 0 0 0  0 0 0 0 0  0 0 0 0 0")}
    ${channel("g", scale * 0.93, "0 0 0 0 0  0 1 0 0 0  0 0 0 0 0")}
    ${channel("b", scale * 0.86, "0 0 0 0 0  0 0 0 0 0  0 0 1 0 0")}
    <feBlend in="r" in2="g" mode="screen" result="rg"/>
    <feBlend in="rg" in2="b" mode="screen"/>`;
  return f;
}

let n = 0;
export function liquidGlass(el) {
  if (!chromium) return;
  const id = `liquid-glass-${n++}`;
  let last = "";
  const apply = () => {
    const w = Math.round(el.offsetWidth), h = Math.round(el.offsetHeight);
    const radius = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0;
    const key = `${w}x${h}`;
    if (!w || !h || key === last) return;
    last = key;
    ensureDefs().querySelector(`#${id}`)?.remove();
    defs.append(filterFor(id, w, h, radius));
    // re-assigning the property makes the browser pick up the rebuilt filter
    el.style.backdropFilter = el.style.webkitBackdropFilter = "none";
    requestAnimationFrame(() => {
      const v = `url(#${id}) blur(12px) saturate(1.7) brightness(0.8)`;
      el.style.backdropFilter = el.style.webkitBackdropFilter = v;
      el.classList.add("refracts");
    });
  };
  new ResizeObserver(apply).observe(el);
  apply();
}
