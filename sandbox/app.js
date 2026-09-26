import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const $ = id => document.getElementById(id);
const bin = (u, T) => fetch(u).then(r => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); }).then(b => new T(b));
const json = u => fetch(u).then(r => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); });
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const DEFAULT_FOOD = [30, 0], DEFAULT_DANGER = [15, 0];

// ---------------------------------------------------------------- assets
const [meta, verts, faces, walk, npos, ngrp] = await Promise.all([
  json('assets/fly.json'), bin('assets/fly_verts.bin', Float32Array), bin('assets/fly_faces.bin', Uint32Array),
  bin('assets/walk.bin', Float32Array), bin('assets/neurons.bin', Float32Array), bin('assets/neuron_groups.bin', Uint8Array)]);
let val = null, bidx = null, brates = null, K, OD, endoOA = 12;   // brain decision table (precompute_valence.py)
const R = meta.reflex, MOT = meta.motion, G = meta.walk.geoms, NN = ngrp.length;
const groupSize = meta.legend.map(l => l.count);

// ---------------------------------------------------------------- renderer helper
function makeView(el, { bloom, bg, zUp }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  el.prepend(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', e => {     // driver reset -> recover instead of staying black
    e.preventDefault();
    document.body.insertAdjacentHTML('beforeend', '<div style="position:fixed;inset:0;display:grid;place-items:center;background:#05070dcc;color:#fff;font:600 16px Inter;z-index:9">Graphics reset, reloading...</div>');
    setTimeout(() => location.reload(), 1200);
  });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(bg);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.2, 800);   // tight near/far = precise depth, no z-fighting
  if (zUp) camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(256, 256), ...bloom);
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;                  // collapsed panel
    renderer.setSize(w, h, false); composer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el); resize();
  return { renderer, scene, camera, controls, composer };
}

// ================================================================ ARENA
const A = makeView($('arena'), { bloom: [0.55, 0.5, 0.82], bg: 0x070b14, zUp: true });
A.scene.fog = new THREE.FogExp2(0x070b14, 0.0085);
A.camera.position.set(-9, -13, 9);
A.controls.target.set(0, 0, 1);
A.controls.maxPolarAngle = Math.PI * 0.49;

const hemi = new THREE.HemisphereLight(0xa9cfff, 0x20160c, 0.7); hemi.position.set(0, 0, 1); A.scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 2.4);
sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 1, far: 80 });
sun.shadow.bias = -0.0004; sun.shadow.radius = 4;
A.scene.add(sun, sun.target);

// ground: shaded floor + additive odor field / grid overlay
const floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: 0x10141d, roughness: .92 }));
floor.receiveShadow = true; A.scene.add(floor);
const odorMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4,
  uniforms: { uFood: { value: new THREE.Vector2(...DEFAULT_FOOD) }, uDanger: { value: new THREE.Vector2(...DEFAULT_DANGER) },
    uDangerS: { value: 1.6 }, uTime: { value: 0 } },
  vertexShader: `varying vec2 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.); vW = w.xy; gl_Position = projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `varying vec2 vW; uniform vec2 uFood, uDanger; uniform float uDangerS, uTime;
    float level(float I){ return clamp((log(I)/2.302585 + 3.4)/3.4, 0., 1.); }
    float grid(float s){ vec2 g = abs(fract(vW/s - .5) - .5) / fwidth(vW/s); return 1. - min(min(g.x, g.y), 1.); }
    void main(){
      float df = distance(vW, uFood), dd = distance(vW, uDanger);
      float lf = level(1./(df*df + .6)), ld = uDangerS > 0. ? level(uDangerS/(dd*dd + .6)) : 0.;
      float rf = smoothstep(.43, .49, abs(fract(lf*6. - uTime*.25) - .5)) * lf;   // thin bright iso-rings
      float rd = smoothstep(.43, .49, abs(fract(ld*6. - uTime*.35) - .5)) * ld;
      vec3 c = vec3(1., .5, .12) * (pow(lf, 2.4)*.5 + rf*.7) + vec3(.15, .5, 1.) * (pow(ld, 2.4)*.55 + rd*.8);
      c += vec3(.22, .32, .5) * (grid(5.)*.07 + grid(25.)*.12);
      gl_FragColor = vec4(c, 1.);
    }`
});
const odorPlane = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), odorMat);
odorPlane.position.z = 0.02; A.scene.add(odorPlane);

// food: glowing fruit + reach ring
function glowBall(color, r, emissive) {
  return new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), new THREE.MeshPhysicalMaterial({
    color, emissive: color, emissiveIntensity: emissive, roughness: .25, clearcoat: 1 }));
}
const food = new THREE.Group(); A.scene.add(food);
const foodBall = glowBall(0xff8a2a, 0.9, 2.2); foodBall.position.z = 1.2; foodBall.castShadow = true; food.add(foodBall);
const foodLight = new THREE.PointLight(0xff8a2a, 40, 30, 2); foodLight.position.z = 2; food.add(foodLight);
const ring = (r, color) => { const m = new THREE.Mesh(new THREE.RingGeometry(r - 0.06, r, 96),
  new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .55, blending: THREE.AdditiveBlending, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8 }));
  m.position.z = 0.04; return m; };
food.add(ring(2, 0xffb070));
// danger: core + pulsing cloud + drifting particles
const danger = new THREE.Group(); A.scene.add(danger);
const dangerBall = glowBall(0x3ca0ff, 0.75, 2.4); dangerBall.position.z = 1.2; danger.add(dangerBall);
const dangerCloud = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), new THREE.MeshBasicMaterial({
  color: 0x2a7cff, transparent: true, opacity: .09, blending: THREE.AdditiveBlending, depthWrite: false }));
dangerCloud.position.z = 1.2; danger.add(dangerCloud);
const dangerLight = new THREE.PointLight(0x3ca0ff, 30, 25, 2); dangerLight.position.z = 2; danger.add(dangerLight);
const NP = 260, pPos = new Float32Array(NP * 3), pSeed = Float32Array.from({ length: NP * 3 }, Math.random);
const particles = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(pPos, 3)),
  new THREE.PointsMaterial({ color: 0x7cc4ff, size: 0.12, transparent: true, opacity: .8, blending: THREE.AdditiveBlending, depthWrite: false }));
danger.add(particles);
// invisible hit spheres for dragging
const hitMat = new THREE.MeshBasicMaterial({ visible: false });
const foodHit = new THREE.Mesh(new THREE.SphereGeometry(2.6), hitMat); foodHit.position.z = 1.2; food.add(foodHit);
const dangerHit = new THREE.Mesh(new THREE.SphereGeometry(2.6), hitMat); dangerHit.position.z = 1.2; danger.add(dangerHit);

// ---------------------------------------------------------------- realistic props (toggle): apple + rolled newspaper
function makeApple() {
  const g = new THREE.Group();
  const prof = [[0, -1], [.45, -.97], [.9, -.78], [1.18, -.35], [1.24, .1], [1.13, .55], [.82, .88], [.42, .98], [.16, .86], [0, .78]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const geo = new THREE.LatheGeometry(prof, 64);
  const col = [], red = new THREE.Color(0xb3121c), deep = new THREE.Color(0x5e0710), blush = new THREE.Color(0xf0b43a), pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const c = red.clone().lerp(blush, Math.max(0, x * .55 - .1) * (0.6 + .4 * Math.sin(z * 9 + y * 4) ** 2)).lerp(deep, Math.max(0, -y - .5) * .6);
    col.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geo.computeVertexNormals();
  const body = new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: .32, clearcoat: 1, clearcoatRoughness: .15, sheen: .4, sheenColor: 0xffd0a0 }));
  body.castShadow = true; g.add(body);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(.045, .07, .62, 12), new THREE.MeshStandardMaterial({ color: 0x5a3a1c, roughness: .8 }));
  stem.position.set(.05, 1.02, 0); stem.rotation.z = -.25; stem.castShadow = true; g.add(stem);
  const leafShape = new THREE.Shape(); leafShape.moveTo(0, 0); leafShape.quadraticCurveTo(.35, .22, .8, 0); leafShape.quadraticCurveTo(.35, -.22, 0, 0);
  const leaf = new THREE.Mesh(new THREE.ShapeGeometry(leafShape, 16), new THREE.MeshStandardMaterial({ color: 0x3f9b2c, roughness: .55, side: THREE.DoubleSide }));
  leaf.position.set(.1, 1.18, 0); leaf.rotation.set(.5, .3, .35); leaf.castShadow = true; g.add(leaf);
  g.rotation.x = Math.PI / 2;            // lathe axis (y) -> world z-up
  g.scale.setScalar(1.35); g.position.z = 1.35;
  return g;
}
function newspaperTexture() {
  const c = document.createElement('canvas'); c.width = 1024; c.height = 512;
  const x = c.getContext('2d');
  x.fillStyle = '#ece6d6'; x.fillRect(0, 0, 1024, 512);
  x.fillStyle = '#1b1b1b'; x.font = 'bold 64px Georgia, serif'; x.fillText('THE DAILY FLY', 40, 80);
  x.fillRect(40, 96, 944, 4);
  x.font = 'bold 30px Georgia, serif'; x.fillText('Dopamine: the brain chemical behind every snack', 40, 142);
  for (let col = 0; col < 4; col++) for (let row = 0; row < 15; row++) {
    const w = 200 - (row % 5 === 4 ? 70 : Math.random() * 20);
    x.fillStyle = `rgba(30,30,30,${.55 + Math.random() * .3})`; x.fillRect(40 + col * 240, 170 + row * 21, w, 7);
  }
  x.fillStyle = '#9a9488'; x.fillRect(760, 170, 200, 130);
  x.fillStyle = '#5b574f'; x.beginPath(); x.arc(860, 235, 40, 0, 7); x.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.wrapS = THREE.RepeatWrapping; t.anisotropy = 8;
  return t;
}
function makePaper() {
  const pivot = new THREE.Group();             // hinge at the handle end; the far end slams onto the danger spot
  const tex = newspaperTexture(), L = 13;
  const roll = new THREE.Group(); roll.position.x = L / 2; roll.rotation.z = Math.PI / 2; pivot.add(roll);
  [[1.05, 0], [.78, .25], [.52, .5]].forEach(([r, inset], k) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L - inset, 64, 1, true), new THREE.MeshStandardMaterial({
      map: tex, color: k ? 0xd8d1bf : 0xffffff, roughness: .85, side: THREE.DoubleSide }));
    m.castShadow = k === 0; roll.add(m);
  });
  const band = new THREE.Mesh(new THREE.TorusGeometry(1.07, .06, 8, 48), new THREE.MeshStandardMaterial({ color: 0xc0392b, roughness: .5 }));
  band.rotation.x = Math.PI / 2; band.position.y = -L * .22; roll.add(band);
  pivot.position.set(-9, 0, 1.05);
  return pivot;
}
const apple = makeApple(); apple.visible = false; food.add(apple);
const paperAim = new THREE.Group(), paperHinge = makePaper(); paperAim.add(paperHinge); paperAim.visible = false; danger.add(paperAim);
let propsOn = false;
function setProps(on) {
  propsOn = on; apple.visible = paperAim.visible = on;
  if (!on && game.over) clearGame();
  foodBall.visible = dangerBall.visible = dangerCloud.visible = particles.visible = !on;
  $('props-btn').classList.toggle('on', on);
  try { localStorage.setItem('fly-sandbox-props', on ? '1' : '0'); } catch (e) {}
}
// ---------------------------------------------------------------- game mode (Props on): swatted = GAME OVER, apple = STAGE CLEAR
const KILL_R = 2.5;                                  // mm from the danger centre
const game = { over: null, t0: 0, contact: false, shake: 0, info: null };
const splat = (() => {
  const sh = new THREE.Shape();
  for (let i = 0; i <= 28; i++) {
    const a = i / 28 * Math.PI * 2, r = 1.1 + .4 * Math.sin(a * 5) * Math.cos(a * 3) + (i % 4 ? 0 : .45);
    i ? sh.lineTo(Math.cos(a) * r, Math.sin(a) * r) : sh.moveTo(Math.cos(a) * r, Math.sin(a) * r);
  }
  const m = new THREE.Mesh(new THREE.ShapeGeometry(sh), new THREE.MeshBasicMaterial({ color: 0x8f9c32, transparent: true, opacity: .85,
    depthWrite: false, polygonOffset: true, polygonOffsetFactor: -8, polygonOffsetUnits: -16 }));
  m.position.z = .08; m.visible = false; A.scene.add(m); return m;
})();
function endGame(kind, info) {
  if (game.over) return;
  Object.assign(game, { over: kind, t0: clock.elapsedTime, contact: false, info });
  if (kind === 'dead') paperAim.rotation.z = Math.atan2(info.at[1] - world.danger[1], info.at[0] - world.danger[0]);   // aim at the fly
  setTimeout(() => showGameOverlay(kind), kind === 'dead' ? 1100 : 450);
}
function showGameOverlay(kind) {
  if (game.over !== kind) return;
  const i = game.info;
  $('game').className = 'game ' + (kind === 'dead' ? 'lose' : 'win');
  $('game-title').textContent = kind === 'dead' ? 'GAME OVER' : 'STAGE CLEAR!';
  $('game-sub').innerHTML = kind === 'dead' ? `SWATTED AT ${i.t.toFixed(2)} S<br>${i.dist.toFixed(1)} MM FROM THE APPLE` : `APPLE REACHED IN ${i.t.toFixed(2)} S`;
  $('game-again').textContent = mode === 'playback' ? '▶ REPLAY' : '▶ START AGAIN';
  $('game-hint').textContent = mode === 'playback' ? 'replay of a real brain + physics run' : `next try uses seed ${+$('seed').value + 1} · sliders stay as set`;
}
function clearGame() {
  game.over = null; game.contact = false; game.shake = 0; $('game').className = 'game hidden';
  flyRoot.scale.set(1, 1, 1); splat.visible = false; paperAim.rotation.z = 0;
}
function squashFly() {
  const [fx, fy] = game.info.at;
  flyRoot.scale.set(1.35, 1.35, .16);
  if (mode === 'playback') flyRoot.position.set(fx * (1 - 1.35), fy * (1 - 1.35), 0);   // scale about the fly, not the origin
  splat.position.set(fx, fy, .08); splat.rotation.z = Math.random() * 6.28; splat.visible = true;
  game.shake = .7;
}
function animatePaper(t) {
  if (!propsOn) return;
  const raised = -0.95 + 0.05 * Math.sin(t * 2.3);
  let a = raised;
  if (game.over === 'dead') {
    const p = t - game.t0;
    if (p < .22) a = raised * (1 - (p / .22) ** 2);                  // slam
    else if (p < .9) a = 0;                                           // hold on the fly
    else a = raised * Math.min(1, ((p - .9) / .6) ** .7);             // lift: reveal the squashed fly
    if (p >= .22 && !game.contact) { game.contact = true; squashFly(); }
  }
  paperHinge.rotation.y = a;
}

// fly body from the real NeuroMechFly meshes
function flyMaterial(n) {
  const std = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: .55, metalness: .05, ...o });
  if (/Eye/i.test(n)) return new THREE.MeshPhysicalMaterial({ color: 0x9a1014, roughness: .3, clearcoat: 1, sheen: 1, sheenColor: 0xff5040 });
  if (/Wing/i.test(n)) return new THREE.MeshPhysicalMaterial({ color: 0xd8ecff, transparent: true, opacity: .3, roughness: .1,
    iridescence: 1, iridescenceIOR: 1.5, side: THREE.DoubleSide, depthWrite: false });
  if (/Haltere/i.test(n)) return std(0xe0d2a8);
  if (/^A\d/.test(n)) { const k = +n.slice(-1); return std(k % 2 ? 0x2e1d10 : 0xc9933f, { roughness: .45 }); }
  if (/Thorax/.test(n)) return new THREE.MeshPhysicalMaterial({ color: 0xa8773a, roughness: .38, clearcoat: .7 });
  if (/Head/.test(n)) return std(0xb0823f);
  if (/Coxa|Femur|Tibia|Tarsus|Trochanter/.test(n)) return std(0x4a3220, { roughness: .6 });
  if (/Pedicel|Funiculus|Arista/.test(n)) return std(0xd0aa6c);
  return std(0x8f6a45);
}
const flyRoot = new THREE.Group(); A.scene.add(flyRoot);
const flyMeshes = meta.meshes.map(m => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(verts.subarray(m.v[0] * 3, (m.v[0] + m.v[1]) * 3), 3));
  g.setIndex(new THREE.BufferAttribute(faces.subarray(m.f[0] * 3, (m.f[0] + m.f[1]) * 3), 1));
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, flyMaterial(m.name));
  mesh.castShadow = !/Wing/i.test(m.name); mesh.matrixAutoUpdate = false;
  flyRoot.add(mesh); return mesh;
});
function setPoses(arr, frame) {
  const o = frame * G * 12;
  for (let k = 0; k < G; k++) {
    const b = o + k * 12, e = flyMeshes[k].matrix.elements;
    // column-major: world = R v + p, with R row-major in arr
    e[0] = arr[b + 3]; e[4] = arr[b + 4]; e[8] = arr[b + 5]; e[12] = arr[b];
    e[1] = arr[b + 6]; e[5] = arr[b + 7]; e[9] = arr[b + 8]; e[13] = arr[b + 1];
    e[2] = arr[b + 9]; e[6] = arr[b + 10]; e[10] = arr[b + 11]; e[14] = arr[b + 2];
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;
    flyMeshes[k].matrixWorldNeedsUpdate = true;
  }
}

// trail coloured by the brain's decision
const TMAX = 8000, TW = 0.12;                       // path ribbon painted on the floor, coloured by valence
const tPts = new Float32Array(TMAX * 2), rPos = new Float32Array(TMAX * 6), rCol = new Float32Array(TMAX * 6);
const rIdx = new Uint32Array((TMAX - 1) * 6);
for (let i = 0; i < TMAX - 1; i++) rIdx.set([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2], i * 6);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(rPos, 3)); trailGeo.setAttribute('color', new THREE.BufferAttribute(rCol, 3));
trailGeo.setIndex(new THREE.BufferAttribute(rIdx, 1)); trailGeo.setDrawRange(0, 0);
const trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: .92, depthWrite: false, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -12 }));
trail.frustumCulled = false; A.scene.add(trail);
let tN = 0;
const cA = new THREE.Color(0xc6ff3c), cV = new THREE.Color(0xff3cd2), c0 = new THREE.Color(0x6b7890);
function valColor(v) { return c0.clone().lerp(v >= 0 ? cA : cV, Math.min(Math.abs(v) * 1.6, 1)); }
function ribbonVerts(i, nx, ny, c) {
  const x = tPts[i * 2], y = tPts[i * 2 + 1];
  rPos.set([x + nx * TW, y + ny * TW, 0.06, x - nx * TW, y - ny * TW, 0.06], i * 6);
  rCol.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
}
function trailPush(x, y, v) {
  if (tN && Math.hypot(x - tPts[(tN - 1) * 2], y - tPts[(tN - 1) * 2 + 1]) < 0.15) return;
  if (tN >= TMAX) return;
  tPts[tN * 2] = x; tPts[tN * 2 + 1] = y;
  let nx = 0, ny = 1;
  if (tN) { const dx = x - tPts[(tN - 1) * 2], dy = y - tPts[(tN - 1) * 2 + 1], l = Math.hypot(dx, dy); nx = -dy / l; ny = dx / l; }
  const c = valColor(v);
  ribbonVerts(tN, nx, ny, c);
  if (tN === 1) ribbonVerts(0, nx, ny, c);
  tN++;
  trailGeo.setDrawRange(0, Math.max(tN - 1, 0) * 6);
  trailGeo.attributes.position.needsUpdate = trailGeo.attributes.color.needsUpdate = true;
}
function trailClear() { tN = 0; trailGeo.setDrawRange(0, 0); }
// beacon ring under the fly, visible when zoomed out
const beacon = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.6, 64), new THREE.MeshBasicMaterial({
  color: 0xdfffb0, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
  polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -12 }));
beacon.position.z = 0.07; A.scene.add(beacon);

// ================================================================ BRAIN
const B = makeView($('brainview'), { bloom: [0.35, 0.35, 0.55], bg: 0x03050a, zUp: false });
B.controls.autoRotate = true; B.controls.autoRotateSpeed = 0.6;
const center = [0, 1, 2].map(a => { let s = 0; for (let i = a; i < npos.length; i += 3) s += npos[i]; return s / NN; });
const bp = new Float32Array(NN * 3), bCol = new Float32Array(NN * 3), bStyle = new Float32Array(NN * 4), glow = new Float32Array(NN);
const legendColors = meta.legend.map(l => new THREE.Color(l.color));
// per group: [rest brightness, rest size, flash brightness, flash size]. Large, busy groups (KCs fire ~39 Hz
// in this model) flash softly; the small decision groups (dopamine, MBONs, octopamine) flash big.
const FLASH = [[.03, .6, .16, 1], [.3, 1.3, 1.5, 2.2], [.3, 1.3, 1.5, 2.2], [.22, 1.1, .9, 1.7], [.35, 1.4, 1.6, 2.4],
  [.45, 1.8, 2, 3], [.45, 1.8, 2, 3], [.45, 1.8, 2, 3], [.3, 1.4, 1.3, 2.2], [.25, 1.2, 1.1, 1.9], [.45, 1.8, 2, 3]];
// glow[] holds each neuron's recent firing rate (Hz, exponential average); brightness = rate / (rate + 30 Hz)
const ACT_TAU = 0.3, ACT_HALF = 30;
for (let i = 0; i < NN; i++) {
  bp[i * 3] = (npos[i * 3] - center[0]) / 100; bp[i * 3 + 1] = -(npos[i * 3 + 1] - center[1]) / 100; bp[i * 3 + 2] = -(npos[i * 3 + 2] - center[2]) / 100;
  const g = ngrp[i], c = legendColors[g];
  bCol.set([c.r, c.g, c.b], i * 3);
  bStyle.set(FLASH[g], i * 4);
}
const brainGeo = new THREE.BufferGeometry();
brainGeo.setAttribute('position', new THREE.BufferAttribute(bp, 3));
brainGeo.setAttribute('aColor', new THREE.BufferAttribute(bCol, 3));
brainGeo.setAttribute('aStyle', new THREE.BufferAttribute(bStyle, 4));
brainGeo.setAttribute('aGlow', new THREE.BufferAttribute(glow, 1));
// decision focus: approach/avoid MBONs (1) and the dopamine neurons that weaken them (.6) are spotlighted
const bDec = Float32Array.from(ngrp, g => (g === 6 || g === 7) ? 1 : (g === 4 || g === 5) ? .6 : 0);
brainGeo.setAttribute('aDec', new THREE.BufferAttribute(bDec, 1));
const brainPts = new THREE.Points(brainGeo, new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uSize: { value: 20 * Math.min(devicePixelRatio, 2) }, uHalf: { value: ACT_HALF }, uFocus: { value: 1 } },
  vertexShader: `attribute vec3 aColor; attribute vec4 aStyle; attribute float aGlow; attribute float aDec; uniform float uSize, uHalf, uFocus; varying vec3 vC;
    void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv;
      float level = aGlow / (aGlow + uHalf);
      float big = 1. + uFocus * aDec * 1.6, dimRest = mix(1., mix(.2, 1., step(.01, aDec)), uFocus);
      gl_PointSize = mix(aStyle.y, aStyle.w, level) * big * uSize / -mv.z;
      vC = aColor * mix(aStyle.x, aStyle.z, level) * dimRest; }`,
  fragmentShader: `varying vec3 vC; void main(){ vec2 d = gl_PointCoord - .5; float r = dot(d,d)*4.; if (r > 1.) discard;
      float a = 1. - r; gl_FragColor = vec4(vC * a * a, 1.); }`
}));
B.scene.add(brainPts);
B.camera.position.set(0, 1.2, 13.5);

// ---------------------------------------------------------------- decision focus: why did the fly decide?
const DEC = meta.decision, decTypes = [...DEC.approach, ...DEC.avoid];
const typeOf = new Int8Array(NN).fill(-1);
decTypes.forEach((m, k) => { for (const i of DEC.mbon_types[m]) typeOf[i] = k; });
function multipliersFromTraces(tr) {       // same rule as brain.py: 1 / (1 + sum share * DAN rate / D50)
  const load = Object.fromEntries(decTypes.map(m => [m, 0]));
  for (const c of DEC.compartments) if (c.mbon in load) load[c.mbon] += c.frac * (tr[c.dan] || 0) / DEC.D50;
  load.MBON11 += (tr.OA_VPM4 || 0) / DEC.D50;
  return Object.fromEntries(decTypes.map(m => [m, 1 / (1 + load[m])]));
}
const sideInput = (d, types) => types.reduce((a, m) => a + (d.mult?.[m] ?? 1), 0) / types.length;
const focusGroup = new THREE.Group(); B.scene.add(focusGroup);
function centroid(g, sgn) {
  const c = new THREE.Vector3(); let n = 0;
  for (let i = 0; i < NN; i++) if (ngrp[i] === g && Math.sign(bp[i * 3]) === sgn) { c.x += bp[i * 3]; c.y += bp[i * 3 + 1]; c.z += bp[i * 3 + 2]; n++; }
  return c.multiplyScalar(1 / Math.max(n, 1));
}
let yTop = -Infinity; for (let i = 1; i < bp.length; i += 3) yTop = Math.max(yTop, bp[i]);
const orbPos = new THREE.Vector3(0, yTop + 1.3, 0);
const CT = { pam: [centroid(4, -1), centroid(4, 1)], ppl1: [centroid(5, -1), centroid(5, 1)],
  app: [centroid(6, -1), centroid(6, 1)], avo: [centroid(7, -1), centroid(7, 1)] };
function beam(a, b, color, lift) {
  const mid = a.clone().add(b).multiplyScalar(.5); mid.y += lift;
  const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, mid, b), 40, .035, 8, false),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  focusGroup.add(m); return m;
}
const beams = { app: CT.app.map(p => beam(p, orbPos, 0xc6ff3c, .9)), avo: CT.avo.map(p => beam(p, orbPos, 0xff3cd2, .9)),
  pam: CT.pam.map((p, i) => beam(p, CT.avo[i], 0x2ee88a, .5)), ppl1: CT.ppl1.map((p, i) => beam(p, CT.app[i], 0xff4b5c, .5)) };
const orb = new THREE.Mesh(new THREE.SphereGeometry(.2, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
orb.position.copy(orbPos); focusGroup.add(orb);
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.cssText = 'position:absolute;inset:0;pointer-events:none';
$('brainview').appendChild(labelRenderer.domElement);
new ResizeObserver(() => { const w = $('brainview').clientWidth, h = $('brainview').clientHeight; if (w && h) labelRenderer.setSize(w, h); }).observe($('brainview'));
function label(pos, color) {
  const d = document.createElement('div'); d.className = 'blabel'; d.style.color = color;
  const o = new CSS2DObject(d); o.position.copy(pos); focusGroup.add(o); return d;
}
const off = (v, x, y) => v.clone().add(new THREE.Vector3(x, y, 0));
const labels = { app: label(off(CT.app[0], -1.3, .35), '#c6ff3c'), avo: label(off(CT.avo[1], 1.3, .35), '#ff3cd2'),
  pam: label(off(CT.pam[0], -1.3, -.45), '#2ee88a'), ppl1: label(off(CT.ppl1[1], 1.3, -.45), '#ff4b5c'), orb: label(off(orbPos, 0, .55), '#fff') };
let focusOn = true;
function updateFocus(d) {
  focusGroup.visible = focusOn; brainPts.material.uniforms.uFocus.value = focusOn ? 1 : 0;
  $('circuit').querySelector('svg').style.display = focusOn ? 'none' : 'block';   // the why-panel replaces the strip
  if (!focusOn) return;
  if (!d) { for (const k in beams) beams[k].forEach(m => { m.material.opacity = .06; }); for (const k in labels) labels[k].textContent = ''; return; }
  const tot = d.app + d.avo + 1, inA = sideInput(d, DEC.approach), inV = sideInput(d, DEC.avoid);
  beams.app.forEach(m => { m.material.opacity = .06 + .9 * d.app / tot; });
  beams.avo.forEach(m => { m.material.opacity = .06 + .9 * d.avo / tot; });
  beams.ppl1.forEach(m => { m.material.opacity = .05 + .85 * (1 - inA); });
  beams.pam.forEach(m => { m.material.opacity = .05 + .85 * (1 - inV); });
  orb.material.color.copy(valColor(d.v)).multiplyScalar(2.2); orb.scale.setScalar(.7 + 1.3 * Math.abs(d.v));
  labels.app.textContent = `approach ${d.app.toFixed(0)} Hz`; labels.avo.textContent = `avoid ${d.avo.toFixed(0)} Hz`;
  labels.ppl1.textContent = `punish DA: approach input ${(inA * 100).toFixed(0)}%`;
  labels.pam.textContent = `reward DA: avoid input ${(inV * 100).toFixed(0)}%`;
  labels.orb.textContent = `decision ${d.v >= 0 ? '+' : ''}${d.v.toFixed(2)}`;
}

// legend + circuit strip
const KEY = [1, 2, 3, 4, 5, 6, 7, 9, 10];
$('brain-legend').innerHTML = '<div style="color:var(--muted);font-size:11px;margin-bottom:4px">group · mean rate</div>' +
  KEY.map(k => `<div class="item"><span><span class="dot" style="background:${meta.legend[k].color}"></span>${meta.legend[k].name}</span><span id="lg${k}">–</span></div>`).join('');
$('circuit').innerHTML = `<svg viewBox="0 0 640 158" font-family="Inter" font-size="11">
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#8b97ad"/></marker>
  <marker id="tee" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M5,0L5,10" stroke-width="3" stroke="#ddd"/></marker></defs>
  ${box(8, 50, 96, 'food smell', 'ORNs', 'c-orn', '#ff9a3c')}${box(124, 50, 96, 'Kenyon cells', 'mushroom body', 'c-kc', '#9b6bff')}
  ${box(248, 6, 112, 'approach MBONs', 'MBON 11·12·13·14', 'c-app', '#c6ff3c')}${box(248, 94, 112, 'avoid MBONs', 'MBON 01·03·04·26·27', 'c-avo', '#ff3cd2')}
  ${box(404, 6, 104, 'punish dopamine', 'PPL1', 'c-ppl', '#ff4b5c')}${box(404, 94, 104, 'reward dopamine', 'PAM', 'c-pam', '#2ee88a')}
  ${box(532, 50, 100, 'decision', 'valence −1…+1', 'c-val', '#e6ecf7')}
  <path d="M104,79H122" stroke="#8b97ad" marker-end="url(#ar)"/><path d="M220,70L246,42" stroke="#8b97ad" marker-end="url(#ar)"/><path d="M220,88L246,116" stroke="#8b97ad" marker-end="url(#ar)"/>
  <path d="M404,35H364" stroke="#ff4b5c" stroke-width="2" marker-end="url(#tee)"/><path d="M404,123H364" stroke="#2ee88a" stroke-width="2" marker-end="url(#tee)"/>
  <path d="M360,46Q470,66 530,74" stroke="#c6ff3c" fill="none" marker-end="url(#ar)"/><path d="M360,112Q470,96 530,86" stroke="#ff3cd2" fill="none" marker-end="url(#ar)"/>
  <text x="383" y="29" fill="#ff8a95" font-size="9" text-anchor="middle">weakens</text><text x="383" y="117" fill="#6ff0b0" font-size="9" text-anchor="middle">weakens</text>
</svg>`;
$('circuit').insertAdjacentHTML('afterbegin', '<div id="why"></div>');
function box(x, y, w, title, sub, id, color) {
  return `<g><rect x="${x}" y="${y}" width="${w}" height="58" rx="9" fill="rgba(20,28,44,.9)" stroke="${color}" stroke-opacity=".7"/>
  <text x="${x + 9}" y="${y + 15}" fill="${color}" font-weight="600">${title}</text>
  <text x="${x + 9}" y="${y + 28}" fill="#8b97ad" font-size="9">${sub}</text>
  <text x="${x + 9}" y="${y + 49}" fill="#e6ecf7" font-family="JetBrains Mono" font-size="14" id="${id}">–</text></g>`;
}

// ================================================================ state
const ui = {
  reward: $('reward'), punish: $('punish'), oa: $('oa'), danger: $('danger'), heading: $('heading'),
  get knobs() { return { reward: +this.reward.value, punish: +this.punish.value, octopamine: +this.oa.value, danger: +this.danger.value,
    heading: +this.heading.value * Math.PI / 180 }; }
};
const world = { food: [...DEFAULT_FOOD], danger: [...DEFAULT_DANGER] };
let mode = 'live';                 // 'live' | 'playback'
let live = null, playing = false;  // live-sim state
let rec = null, recT = 0, recPlaying = false;

function placeObjects() {
  food.position.set(world.food[0], world.food[1], 0); danger.position.set(world.danger[0], world.danger[1], 0);
  odorMat.uniforms.uFood.value.set(...world.food); odorMat.uniforms.uDanger.value.set(...world.danger);
  const s = mode === 'playback' && rec ? rec.params.danger : ui.knobs.danger;
  odorMat.uniforms.uDangerS.value = s; danger.visible = s > 0;
  dangerCloud.scale.setScalar(1.5 + s * 2.2);
}

// ---------------------------------------------------------------- live model (mirrors run.py)
function frac(arr, v) { v = clamp(v, arr[0], arr[arr.length - 1]); let i = 0; while (i < arr.length - 2 && v > arr[i + 1]) i++; return [i, (v - arr[i]) / (arr[i + 1] - arr[i])]; }
const P = (ri, pi, oi) => val.points[(ri * K.length + pi) * OD.length + oi];
function lookup(reward, punish, odor) {
  const [ri, rt] = frac(K, reward), [pi, pt] = frac(K, punish), [oi, ot] = frac(OD, odor);
  const out = { approach: 0, avoid: 0, pam: 0, ppl1: 0 };
  for (const [a, wa] of [[ri, 1 - rt], [ri + 1, rt]]) for (const [b, wb] of [[pi, 1 - pt], [pi + 1, pt]]) for (const [c, wc] of [[oi, 1 - ot], [oi + 1, ot]]) {
    const p = P(a, b, c), w = wa * wb * wc; if (!w) continue;
    out.approach += w * p.approach; out.avoid += w * p.avoid; out.pam += w * p.pam_mean; out.ppl1 += w * p.ppl1_mean;
  }
  return out;
}
function nearestPoint(reward, punish, odor) {
  const n = (arr, v) => arr.reduce((b, x, i) => Math.abs(x - v) < Math.abs(arr[b] - v) ? i : b, 0);
  return P(n(K, reward), n(K, punish), n(OD, odor));
}
async function loadVal() {
  try {
    const [v, bi, br] = await Promise.all([json('assets/valence.json'), bin('assets/brain_idx.bin', Uint32Array), bin('assets/brain_rates.bin', Float32Array)]);
    for (const p of v.points) {       // per-point group means + dopamine means for the UI
      const tr = Object.entries(p.traces);
      const mean = f => { const x = tr.filter(([k]) => f(k)).map(([, y]) => y); return x.reduce((a, b) => a + b, 0) / Math.max(x.length, 1); };
      p.pam_mean = mean(k => k.startsWith('PAM')); p.ppl1_mean = mean(k => k.startsWith('PPL1'));
      const sum = new Float64Array(meta.legend.length);
      for (let j = p.sparse[0]; j < p.sparse[0] + p.sparse[1]; j++) sum[ngrp[bi[j]]] += br[j];
      p.groupRate = Array.from(sum, (x, g) => x / Math.max(groupSize[g], 1));
    }
    [val, bidx, brates, K, OD] = [v, bi, br, v.knob, v.odor_hz];
    endoOA = P(K.indexOf(0), K.indexOf(0), 0).oa_hz;
    return true;
  } catch { return false; }
}
function setStartReady(ready) {
  $('play').disabled = !ready;
  $('play').textContent = ready ? '▶ Start' : '⏳ Brain table computing…';
  $('play').title = ready ? '' : 'precompute_valence.py is still measuring the real brain at 100 settings; this button enables itself when done';
}
const dangerIdx = [], foodIdx = [];
for (let i = 0; i < NN; i++) { if (ngrp[i] === 2) dangerIdx.push(i); if (ngrp[i] === 1) foodIdx.push(i); }

function antennae(x, y, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return ['L', 'R'].map(k => { const [ax, ay] = meta.antenna[k]; return [x + c * ax - s * ay, y + s * ax + c * ay]; });
}
const intensity = (p, src, peak) => peak / Math.max((p[0] - src[0]) ** 2 + (p[1] - src[1]) ** 2, 0.09);
const contrast = (l, r) => { const m = (l + r) / 2; return m > 0 ? (l - r) / m : 0; };
const ornHz = I => R.ORN_MAX_HZ * I / (I + R.ORN_HALF);

// ponytail: gait wobble as a heading random walk (0.35 rad/sqrt s, hand-set, not fitted); without it a fly aimed
// exactly at a source feels no left/right difference and never turns, unlike the physics body.
const YAW_NOISE = 0.35, SUB = 0.005;
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), a | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = r => Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());
function liveReset() {
  const k = ui.knobs;
  live = { x: 0, y: 0, yaw: k.heading, t: 0, acc: 0, app: 0, avo: 0, v: 0, action: [1, 1], reached: null, orn: 0, dangerHz: 0, pt: null, rand: rng(+$('seed').value * 7919 + 1), sub: 0 };
  playing = false; if (val) $('play').textContent = '▶ Start'; clearGame();
  trailClear(); flyRoot.position.set(0, 0, 0); flyRoot.rotation.z = live.yaw; setPoses(walk, 0);
  hud(); glow.fill(0);
}
function liveDecide() {
  const k = ui.knobs, [aL, aR] = antennae(live.x, live.y, live.yaw);
  const fL = intensity(aL, world.food, 1), fR = intensity(aR, world.food, 1);
  const dL = k.danger > 0 ? intensity(aL, world.danger, k.danger) : 0, dR = k.danger > 0 ? intensity(aR, world.danger, k.danger) : 0;
  live.orn = (ornHz(fL) + ornHz(fR)) / 2; live.dangerHz = (ornHz(dL) + ornHz(dR)) / 2;
  const look = lookup(k.reward, k.punish, live.orn), a = 1 - Math.exp(-0.025 / 0.3);
  live.app += a * (look.approach - live.app); live.avo += a * (look.avoid - live.avo);
  live.pam = look.pam; live.ppl1 = look.ppl1;
  live.v = (live.app - live.avo) / (live.app + live.avo + 1);
  const bias = -R.G_FOOD * live.v * contrast(fL, fR) + R.G_DANGER * k.danger * contrast(dL, dR);
  const turn = Math.tanh(bias * bias) * Math.sign(bias);
  const oaHz = k.octopamine < 0 ? 0 : endoOA + R.KNOB_HZ * k.octopamine;
  const drive = R.DRIVE * (1 + R.AROUSAL * Math.min(oaHz / R.KNOB_HZ, 1));
  live.action = [drive, drive]; live.action[turn > 0 ? 1 : 0] -= Math.abs(turn) * 0.8 * drive;
  live.pt = nearestPoint(k.reward, k.punish, live.orn);
}
function liveStep(dt) {
  live.acc += dt;
  while (live.acc >= SUB && playing) {      // fixed 5 ms steps: same seed -> same path at any frame rate
    live.acc -= SUB;
    if (live.sub++ % 5 === 0) liveDecide();   // brain/reflex decision every 25 ms, as in run.py
    const [l, r] = live.action;
    live.yaw += MOT.yaw_per_diff * (r - l) * SUB + YAW_NOISE * Math.sqrt(SUB) * gauss(live.rand);
    const v = MOT.speed_per_drive * (l + r) / 2;
    live.x += Math.cos(live.yaw) * v * SUB; live.y += Math.sin(live.yaw) * v * SUB; live.t += SUB;
    if (Math.hypot(live.x - world.food[0], live.y - world.food[1]) < 2) {
      live.reached = live.t; playing = false; $('play').textContent = '▶ Start';
      if (propsOn) endGame('won', { t: live.t });
    }
    if (playing && propsOn && ui.knobs.danger > 0 && Math.hypot(live.x - world.danger[0], live.y - world.danger[1]) < KILL_R) {
      live.dead = live.t; playing = false; $('play').textContent = '▶ Start';
      endGame('dead', { t: live.t, at: [live.x, live.y], dist: Math.hypot(live.x - world.food[0], live.y - world.food[1]) });
    }
    if (live.t > 40) { playing = false; $('play').textContent = '▶ Start'; }
  }
  flyRoot.position.set(live.x, live.y, 0); flyRoot.rotation.z = live.yaw;
  setPoses(walk, Math.floor((live.t % 1) / meta.walk.dt) % meta.walk.frames);
  trailPush(live.x, live.y, live.v);
}
function liveBrain(dt) {
  if (!live.pt) return;
  const p = live.pt;
  for (let j = p.sparse[0]; j < p.sparse[0] + p.sparse[1]; j++) if (Math.random() < brates[j] * dt) glow[bidx[j]] += 1 / ACT_TAU;
  const pd = live.dangerHz * dt; if (pd > 0) for (const i of dangerIdx) if (Math.random() < pd) glow[i] += 1 / ACT_TAU;
}

// ---------------------------------------------------------------- HUD
function decisionState() {
  if (mode === 'playback' && rec) {
    const r = recRow(recT);
    const mult = r.mult || (val ? multipliersFromTraces(nearestPoint(rec.params.reward, rec.params.punish, r.orn_hz ?? 40).traces) : null);
    return { v: r.valence, app: r.approach_hz, avo: r.avoid_hz, mult, types: Object.fromEntries(decTypes.map((m, k) => [m, rec.typeRate[k]])) };
  }
  if (!live?.pt) return null;
  return { v: live.v, app: live.app, avo: live.avo, types: live.pt.mbon, mult: multipliersFromTraces(live.pt.traces) };
}
function updateWhy(d) {
  const el = $('why');
  if (!d) { el.innerHTML = '<span style="color:var(--muted)">Press Start, or play a recording, to see why the fly decides.</span>'; return; }
  const [word, col] = d.v > .15 ? ['APPROACH', 'var(--approach)'] : d.v < -.15 ? ['AVOID', 'var(--avoid)'] : ['UNDECIDED', 'var(--muted)'];
  const max = Math.max(1, ...Object.values(d.types)), inA = sideInput(d, DEC.approach), inV = sideInput(d, DEC.avoid);
  const bars = (types, c) => types.map(m => `<div class="tb"><span>${m.slice(4)}</span><i style="width:${(d.types[m] / max * 100).toFixed(0)}%;background:${c}"></i><b>${d.types[m].toFixed(0)}</b></div>`).join('');
  el.innerHTML = `<div class="why-top"><span class="verdict" style="color:${col}">${word}</span>
    <span>approach side <b style="color:var(--approach)">${d.app.toFixed(0)} Hz</b> vs avoid side <b style="color:var(--avoid)">${d.avo.toFixed(0)} Hz</b></span></div>
    <div class="tug"><i style="width:${(100 * d.app / (d.app + d.avo + 1e-9)).toFixed(1)}%"></i></div>
    <div class="why-cols"><div><div class="why-h" style="color:var(--approach)">approach MBONs (Hz)</div>${bars(DEC.approach, '#c6ff3c')}</div>
      <div><div class="why-h" style="color:var(--avoid)">avoid MBONs (Hz)</div>${bars(DEC.avoid, '#ff3cd2')}</div></div>
    <div class="because">Dopamine left the <b style="color:var(--approach)">approach</b> side's Kenyon-cell input at <b>${(inA * 100).toFixed(0)}%</b>
      (mostly punishment DA, PPL1) and the <b style="color:var(--avoid)">avoid</b> side's at <b>${(inV * 100).toFixed(0)}%</b> (mostly reward DA, PAM).</div>`;
}
function hud() {
  const dstate = decisionState(); updateWhy(dstate); updateFocus(dstate);
  const s = mode === 'playback' ? recState() : live;
  if (!s) return;
  const d = Math.hypot(s.x - world.food[0], s.y - world.food[1]);
  $('h-time').textContent = `${s.t.toFixed(2)} s`; $('h-dist').textContent = `${d.toFixed(1)} mm`;
  const st = $('h-status');
  if (game.over === 'dead') { st.textContent = `✗ swatted by the newspaper at ${game.info.t.toFixed(2)} s`; st.className = 'err'; }
  else if (s.reached != null && s.t >= s.reached) { st.textContent = `✓ reached the food at ${s.reached.toFixed(2)} s`; st.className = 'reached'; }
  else { st.className = ''; st.textContent = mode === 'playback' ? (recPlaying ? 'replaying recorded run' : 'paused') : (playing ? 'walking…' : 'Press Start'); }
  $('h-val').style.left = `${50 + clamp(s.v, -1, 1) * 50}%`; $('h-valnum').textContent = (s.v >= 0 ? '+' : '') + s.v.toFixed(2);
  const set = (id, v, u = ' Hz') => { $(id).textContent = v == null || Number.isNaN(v) ? '–' : `${v.toFixed(v >= 100 ? 0 : 1)}${u}`; };
  set('c-orn', s.orn); set('c-kc', s.kc); set('c-app', s.app); set('c-avo', s.avo); set('c-pam', s.pam); set('c-ppl', s.ppl1); set('c-val', s.v, '');
  const gr = s.groupRate || (live?.pt?.groupRate);
  for (const k of KEY) $(`lg${k}`).textContent = gr ? `${gr[k].toFixed(1)} Hz` : '–';
}
function liveHudExtras() { if (live?.pt) { live.kc = live.pt.groupRate[3]; live.groupRate = live.pt.groupRate; } }

// ---------------------------------------------------------------- real simulation + playback
async function runReal() {
  const k = ui.knobs;
  const body = { ...k, seed: +$('seed').value, seconds: +$('seconds').value, food: world.food, danger_pos: world.danger };
  $('run-real').disabled = true; $('prog').style.width = '0%';
  let res;
  try { res = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json()); }
  catch { $('real-status').innerHTML = '<span class="err">Server not reachable. Start it with: .venv/Scripts/python sandbox_server.py</span>'; $('run-real').disabled = false; return; }
  if (!res.id) { $('real-status').innerHTML = `<span class="err">${res.error}</span>`; $('run-real').disabled = false; return; }
  const t0 = performance.now();
  const poll = async () => {
    const s = await fetch(`/api/run/${res.id}`).then(r => r.json()).catch(() => ({ status: 'running', progress: 0 }));
    const el = (performance.now() - t0) / 1000;
    if (s.status === 'running') {
      $('prog').style.width = `${(s.progress * 100).toFixed(1)}%`;
      const eta = s.progress > 0.02 ? ` · ~${Math.ceil(el / s.progress * (1 - s.progress) / 60)} min left` : '';
      $('real-status').textContent = `Simulating ${(s.progress * body.seconds).toFixed(2)} / ${body.seconds} s of fly time${eta}`;
      return setTimeout(poll, 3000);
    }
    $('run-real').disabled = false;
    if (s.status !== 'done') { $('real-status').innerHTML = `<span class="err">Simulation failed.</span>`; console.error(s.error); return; }
    $('prog').style.width = '100%'; $('real-status').textContent = `Done in ${(el / 60).toFixed(1)} min. Loading recording…`;
    await loadRecording(res.id);
  };
  poll();
}
async function loadRecording(id) {
  const [run, poses, sidx, scnt] = await Promise.all([json(`/runs/${id}/run.json`), bin(`/runs/${id}/poses.bin`, Float32Array),
    bin(`/runs/${id}/spikes_idx.bin`, Uint32Array), bin(`/runs/${id}/spikes_cnt.bin`, Uint8Array)]);
  rec = { ...run, poses, sidx, scnt, lastW: -1, groupRate: new Array(meta.legend.length).fill(0), typeRate: new Array(decTypes.length).fill(0) };
  world.food = [...run.params.food]; world.danger = [...run.params.danger_pos];
  const s = run.summary;
  $('real-status').textContent = `Recorded run: ${s.reached ? `reached food at ${s.time_to_food.toFixed(2)} s` : `did not reach food (closest ${s.min_food_dist.toFixed(1)} mm)`}.`;
  enterPlayback();
}
function enterPlayback() {
  mode = 'playback'; playing = false; recT = 0; recPlaying = true; rec.lastW = -1;
  $('badge').className = 'badge-real'; $('badge').textContent = 'REAL SIMULATION · RECORDED';
  $('brain-mode').textContent = 'recorded spikes from this run';
  $('playback').classList.add('show'); $('pb-play').textContent = '⏸ Pause';
  clearGame(); flyRoot.position.set(0, 0, 0); flyRoot.rotation.set(0, 0, 0); trailClear(); glow.fill(0); placeObjects();
}
function exitPlayback() {
  mode = 'live'; recPlaying = false; $('badge').className = 'badge-live'; $('badge').textContent = 'LIVE PREVIEW';
  $('brain-mode').textContent = 'rates from the real brain at these settings'; $('playback').classList.remove('show');
  placeObjects(); liveReset();
}
function recRow(t) { const rows = rec.rows; let i = Math.min(Math.floor(t / 0.025), rows.length - 1); return rows[Math.max(i, 0)]; }
function recState() {
  const r = recRow(recT);
  return { x: r.x, y: r.y, t: recT, v: r.valence, app: r.approach_hz, avo: r.avoid_hz, pam: r.pam_hz, ppl1: r.ppl1_hz, orn: r.orn_hz, kc: r.kc_hz,
    reached: rec.summary.time_to_food, groupRate: rec.groupRate };
}
function recStep(dt) {
  const dur = rec.rows.length * 0.025;
  if (recPlaying) recT = Math.min(recT + dt, dur - 1e-6);
  if (recT >= dur - 1e-6) { recPlaying = false; $('pb-play').textContent = '▶ Play recording'; }
  if (propsOn && rec.params.danger > 0) {        // game rule on top of the recorded run
    const hit = rec.rows.find(r => Math.hypot(r.x - world.danger[0], r.y - world.danger[1]) < KILL_R);
    if (hit && recT >= hit.t - 0.025) {
      recT = hit.t - 0.025; recPlaying = false; $('pb-play').textContent = '▶ Play recording';
      endGame('dead', { t: hit.t, at: [hit.x, hit.y], dist: Math.hypot(hit.x - world.food[0], hit.y - world.food[1]) });
    } else if (!hit && rec.summary.reached && recT >= rec.summary.time_to_food - 0.03) endGame('won', { t: rec.summary.time_to_food });
  }
  const f = Math.min(Math.floor(recT / 0.01), rec.frames - 1);
  setPoses(rec.poses, f);
  const w = Math.min(Math.floor(recT / 0.025), rec.windows.length - 1);
  if (w < rec.lastW) { glow.fill(0); trailClear(); rec.lastW = -1; }
  if (w - rec.lastW > 24) {                  // big jump: path for the skipped part, spikes only for the last 0.6 s
    for (let ww = rec.lastW + 1; ww <= w - 24; ww++) trailPush(rec.rows[ww].x, rec.rows[ww].y, rec.rows[ww].valence);
    glow.fill(0); rec.lastW = w - 24;
  }
  const kw = Math.exp(-0.025 / ACT_TAU);
  for (let ww = rec.lastW + 1; ww <= w; ww++) {
    for (let i = 0; i < NN; i++) glow[i] *= kw;
    const [o, n] = rec.windows[ww], gc = new Float64Array(meta.legend.length);
    const tc = new Float64Array(decTypes.length);
    for (let j = o; j < o + n; j++) {
      const i = rec.sidx[j]; glow[i] += rec.scnt[j] / ACT_TAU; gc[ngrp[i]] += rec.scnt[j]; if (typeOf[i] >= 0) tc[typeOf[i]] += rec.scnt[j];
    }
    decTypes.forEach((m, k) => { rec.typeRate[k] += 0.25 * (tc[k] / DEC.mbon_types[m].length / 0.025 - rec.typeRate[k]); });
    for (let g = 0; g < gc.length; g++) rec.groupRate[g] += 0.25 * (gc[g] / Math.max(groupSize[g], 1) / 0.025 - rec.groupRate[g]);
    const row = rec.rows[ww]; trailPush(row.x, row.y, row.valence);
  }
  rec.lastW = w;
  $('pb-scrub').value = Math.round(recT / dur * 1000);
}

// ---------------------------------------------------------------- dragging food / danger
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
let dragging = null;
function pick(e) {
  const r = A.renderer.domElement.getBoundingClientRect();
  ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(ndc, A.camera);
}
A.renderer.domElement.addEventListener('pointerdown', e => {
  if (e.button !== 0) return;
  pick(e);
  const hit = ray.intersectObjects([foodHit, dangerHit])[0];
  if (!hit) return;
  dragging = hit.object === foodHit ? 'food' : 'danger';
  A.controls.enabled = false; A.renderer.domElement.setPointerCapture(e.pointerId);
  if (mode === 'playback') exitPlayback();
});
A.renderer.domElement.addEventListener('pointermove', e => {
  if (!dragging) return;
  pick(e); const p = new THREE.Vector3();
  if (ray.ray.intersectPlane(groundPlane, p)) { world[dragging] = [clamp(p.x, -60, 60), clamp(p.y, -60, 60)]; placeObjects(); }
});
A.renderer.domElement.addEventListener('pointerup', () => { if (dragging) { dragging = null; A.controls.enabled = true; liveReset(); } });

// ---------------------------------------------------------------- controls
const fmt = v => (v > 0 ? '+' : '') + (+v).toFixed(2);
for (const [id, out, f] of [['reward', 'v-reward', fmt], ['punish', 'v-punish', fmt], ['oa', 'v-oa', fmt], ['danger', 'v-danger', v => (+v).toFixed(2)],
  ['heading', 'v-heading', v => `${v}°`]]) {
  const upd = () => { $(out).textContent = f($(id).value); placeObjects(); if (id === 'heading' && !playing) liveReset(); };
  $(id).addEventListener('input', upd); upd();
}
$('play').onclick = () => {
  if (mode === 'playback') exitPlayback();
  if (!val) return;
  if (live.reached != null || live.dead != null || live.t > 40) liveReset();
  playing = !playing; $('play').textContent = playing ? '⏸ Pause' : '▶ Start';
};
$('reset').onclick = () => { if (mode === 'playback') exitPlayback(); else liveReset(); };
$('reset-pos').onclick = () => { world.food = [...DEFAULT_FOOD]; world.danger = [...DEFAULT_DANGER]; if (mode === 'playback') exitPlayback(); placeObjects(); liveReset(); };
$('rand-heading').onclick = () => { $('heading').value = Math.round((Math.random() * 360 - 180) / 5) * 5; $('heading').dispatchEvent(new Event('input')); };
$('run-real').onclick = runReal;
$('game-again').onclick = () => {
  if (mode === 'playback') { clearGame(); flyRoot.position.set(0, 0, 0); recT = 0; recPlaying = true; $('pb-play').textContent = '⏸ Pause'; return; }
  $('seed').value = +$('seed').value + 1; liveReset(); playing = true; $('play').textContent = '⏸ Pause';
};
$('props-btn').onclick = () => setProps(!propsOn);
$('focus-btn').onclick = () => { focusOn = !focusOn; $('focus-btn').classList.toggle('on', focusOn); hud(); };
$('seed').addEventListener('input', () => { if (mode === 'live' && !playing) liveReset(); });
$('pb-play').onclick = () => { if (game.over) { clearGame(); recT = 0; } if (recT >= rec.rows.length * 0.025 - 0.01) { recT = 0; } recPlaying = !recPlaying; $('pb-play').textContent = recPlaying ? '⏸ Pause' : '▶ Play recording'; };
$('pb-live').onclick = exitPlayback;
$('pb-scrub').addEventListener('input', e => { if (game.over) { clearGame(); flyRoot.position.set(0, 0, 0); } recPlaying = false; $('pb-play').textContent = '▶ Play recording'; recT = +e.target.value / 1000 * rec.rows.length * 0.025; });
let follow = true;
$('cam-follow').onclick = () => { follow = true; $('cam-follow').classList.add('on'); $('cam-over').classList.remove('on'); };
$('cam-over').onclick = () => {
  follow = false; $('cam-over').classList.add('on'); $('cam-follow').classList.remove('on');
  const xs = [0, world.food[0], world.danger[0]], ys = [0, world.food[1], world.danger[1]];
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 14;
  A.controls.target.set(cx, cy, 0); A.camera.position.set(cx - 0.35 * span, cy - 0.95 * span, 0.95 * span);
};

// headless live-preview run, used to check the preview against real runs (see README)
window.sandboxDebug = {
  run({ sliders = {}, food, danger, heading, seed = 0, seconds = 6 } = {}) {
    for (const [k, v] of Object.entries(sliders)) $(k).value = v;
    if (food) world.food = food; if (danger) world.danger = danger;
    placeObjects(); liveReset();
    if (heading != null) live.yaw = heading;
    live.rand = rng(seed * 7919 + 1); playing = true;
    while (playing && live.t < seconds) liveStep(SUB);
    playing = false; $('play').textContent = '▶ Start';
    return { reached: live.reached, x: live.x, y: live.y, v: live.v };
  }
};

// ---------------------------------------------------------------- loop
placeObjects(); liveReset();
try { if (localStorage.getItem('fly-sandbox-props') === '1') setProps(true); } catch (e) {}
if (!await loadVal()) {
  setStartReady(false);
  const poll = setInterval(async () => { if (await loadVal()) { clearInterval(poll); setStartReady(true); } }, 20000);
}
const runParam = new URLSearchParams(location.search).get('run');   // reopen a past real run: ?run=<id>
if (runParam && /^[0-9a-f]{10}$/.test(runParam)) loadRecording(runParam);
const clock = new THREE.Clock(); let hudT = 0;
const flyPos = new THREE.Vector3();
function frame() {
  const rdt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  if (mode === 'live') {
    const dt = rdt * +$('speed').value;
    if (playing) { liveStep(dt); liveBrain(dt); liveHudExtras(); }
    flyPos.set(live.x, live.y, 1.2);
  } else {
    recStep(rdt * +$('pb-speed').value);
    const r = recRow(recT); flyPos.set(r.x, r.y, 1.2);
  }
  // firing-rate average decays in simulated time (frozen while paused)
  const simDt = mode === 'live' && playing ? rdt * +$('speed').value : 0;   // playback decays per recorded window
  if (simDt > 0) { const k = Math.exp(-simDt / ACT_TAU); for (let i = 0; i < NN; i++) glow[i] *= k; }
  brainGeo.attributes.aGlow.needsUpdate = true;
  // ambience
  odorMat.uniforms.uTime.value = t;
  dangerCloud.material.opacity = 0.07 + 0.03 * Math.sin(t * 2.2);
  foodBall.scale.setScalar(1 + 0.04 * Math.sin(t * 3));
  const rad = 1.2 + ui.knobs.danger * 1.6;
  for (let i = 0; i < NP; i++) {
    const a = pSeed[i * 3] * 6.283 + t * (0.2 + pSeed[i * 3 + 1] * 0.4), rr = rad * Math.sqrt(pSeed[i * 3 + 2]);
    pPos[i * 3] = Math.cos(a) * rr; pPos[i * 3 + 1] = Math.sin(a) * rr; pPos[i * 3 + 2] = 0.3 + ((pSeed[i * 3 + 1] * 4 + t * 0.35) % 3);
  }
  particles.geometry.attributes.position.needsUpdate = true;
  // cameras + light follow
  if (follow) {
    const d = flyPos.clone().sub(A.controls.target).multiplyScalar(0.08);
    A.controls.target.add(d); A.camera.position.add(d);
  }
  sun.position.set(flyPos.x + 12, flyPos.y - 9, 30); sun.target.position.copy(flyPos);
  animatePaper(t);
  beacon.position.x = flyPos.x; beacon.position.y = flyPos.y;
  beacon.material.opacity = clamp((A.camera.position.distanceTo(flyPos) - 14) / 30, 0, 0.75) * (0.75 + 0.25 * Math.sin(t * 4));
  A.controls.update(); B.controls.update();
  const shakeOff = new THREE.Vector3();
  if (game.shake > .01) { shakeOff.set(Math.random() - .5, Math.random() - .5, Math.random() - .5).multiplyScalar(game.shake); game.shake *= Math.exp(-rdt / .12); }
  A.camera.position.add(shakeOff);
  A.composer.render();
  if ($('brainview').clientWidth > 0) { B.composer.render(); labelRenderer.render(B.scene, B.camera); }   // skip hidden brain
  A.camera.position.sub(shakeOff);
  if ((hudT += rdt) > 0.08) { hudT = 0; hud(); }
  requestAnimationFrame(frame);
}
frame();
