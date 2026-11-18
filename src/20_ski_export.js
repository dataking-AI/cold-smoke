// ---------------------------------------------------------------------------
// Motion capture export for the simulated skier. The exported keypoints are
// the game's procedural skeleton, while the physics fields are copied from
// the latest fixed physics substep.
const SkiExport = (() => {
  const KEYPOINT_NAMES = [
    'head', 'neck', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'hip_centre', 'l_hip', 'r_hip', 'l_knee',
    'r_knee', 'l_ankle', 'r_ankle',
  ];
  const JOINT_KEYS = [
    'head', 'neck', 'shL', 'shR', 'elbowL', 'elbowR', 'wristL', 'wristR',
    'pelvis', 'hipL', 'hipR', 'kneeL', 'kneeR', 'ankleL', 'ankleR',
  ];
  const clone3 = v => v && [Number(v[0]) || 0, Number(v[1]) || 0, Number(v[2]) || 0];
  const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len3 = a => Math.hypot(a[0], a[1], a[2]);
  const norm3 = a => { const l = len3(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const round = (v, digits = 6) => Number(Number(v || 0).toFixed(digits));
  const round3 = v => v ? [round(v[0]), round(v[1]), round(v[2])] : [0, 0, 0];

  let R = null;

  function worldPoint(j) {
    const W = typeof Skier !== 'undefined' && Skier.world;
    if (!W || !W.root || !j || typeof Skier.toWorld !== 'function') return null;
    const q = Skier.toWorld(j), o = typeof Render !== 'undefined' && Render.origin ? Render.origin : [0, 0, 0];
    return [q[0] + o[0], q[1] + o[1], q[2] + o[2]];
  }

  function initialiseAxes(hip) {
    const root = Game.root, S = Game.s;
    // H5's FLU convention uses gravity-up for U. The terrain normal is
    // exported separately in ground.normal and must not replace world up.
    let up = [0, 1, 0];
    let forward = clone3(root.fwd) || clone3(S.t) || [0, 0, -1];
    forward = sub3(forward, [up[0] * dot3(forward, up), up[1] * dot3(forward, up), up[2] * dot3(forward, up)]);
    forward = norm3(forward);
    let left = norm3(cross3(up, forward));
    // Re-orthogonalise up so the run-local FLU basis is right handed.
    up = norm3(cross3(forward, left));
    R.origin = hip.slice(); R.forward = forward; R.left = left; R.up = up;
  }

  function runFLU(p) {
    const d = sub3(p, R.origin);
    return [dot3(d, R.forward), dot3(d, R.left), dot3(d, R.up)];
  }

  function runFLUVector(v) {
    return [dot3(v, R.forward), dot3(v, R.left), dot3(v, R.up)];
  }

  function gravityBody(points) {
    const z0 = sub3(points[1], points[8]);
    if (len3(z0) < 1e-6) return null;
    let z = norm3(z0);
    let x = sub3(points[10], points[9]);
    x = sub3(x, [z[0] * dot3(x, z), z[1] * dot3(x, z), z[2] * dot3(x, z)]);
    if (len3(x) < 1e-6) return null;
    x = norm3(x);
    const y0 = cross3(z, x);
    if (len3(y0) < 1e-6) return null;
    const y = norm3(y0), g = [0, 0, -1];
    return [dot3(g, x), dot3(g, y), dot3(g, z)];
  }

  function physicsSnapshot(S, dt) {
    return {
      step_dt: dt,
      step: (R.physics_step || 0) + 1,
      p: clone3(S.p), v: clone3(S.v), n: clone3(S.n), t: clone3(S.t), r: clone3(S.r),
      h: Number(S.h) || 0, e: Number(S.e) || 0, speed: Number(S.speed) || 0,
      contact: !!S.contact, air: !!S.air, load: Number(S.load) || 0,
      lean: Number(S.lean) || 0, beta: Number(S.beta) || 0,
      powder: Number(S.powder) || 0,
    };
  }

  function makeSample(dt) {
    const S = Game.s, W = typeof Skier !== 'undefined' && Skier.world;
    if (!W || !W.J || !W.root) return null;
    const points = JOINT_KEYS.map(k => worldPoint(W.J[k]));
    if (points.some(p => !p)) return null;
    const hips = [(points[9][0] + points[10][0]) * 0.5, (points[9][1] + points[10][1]) * 0.5, (points[9][2] + points[10][2]) * 0.5];
    points[8] = hips;
    if (!R.origin) initialiseAxes(hips);
    const pointsFlu = points.map(runFLU), keypointsFlu = {};
    for (let i = 0; i < KEYPOINT_NAMES.length; i++) {
      keypointsFlu[KEYPOINT_NAMES[i]] = round3(pointsFlu[i]);
    }
    const gravity = gravityBody(pointsFlu);
    if (!gravity) return null;
    const P = R.physics || physicsSnapshot(S, 0);
    const segment = typeof Course !== 'undefined' && Course.segAt ? Course.segAt(-S.p[2]) : null;
    const velocity = clone3(P.v), groundPoint = [P.p[0], P.h, P.p[2]], groundFlu = runFLU(groundPoint);
    return {
      t: round(Number(Game.runT) || 0, 5),
      dt: round(dt, 5),
      physics_dt: round(P.step_dt),
      physics_step: P.step,
      human_pos: round3(runFLU(hips)),
      hip_centre_vel: round3(runFLUVector(velocity)),
      speed: round(P.speed),
      // Ground below the human's current x/y. It remains valid during flight.
      ground_z: round(groundFlu[2]),
      ground: {
        position: round3(groundFlu), normal: round3(runFLUVector(P.n)), tangent: round3(runFLUVector(P.t)), right: round3(runFLUVector(P.r)),
        contact: P.contact, penetration: round(P.e), segment: segment ? segment.type : '', powder: round(P.powder),
      },
      gravity_body: round3(gravity),
      load: round(P.load), lean: round(P.lean), beta: round(P.beta),
      air: P.air, keypoints_world: keypointsFlu,
    };
  }

  function start(options = {}) {
    const hz = Number(options.sampleHz == null ? 30 : options.sampleHz);
    R = {
      active: true, hz: hz > 0 ? hz : 0, acc: 0, physics_step: 0,
      physics: null, origin: null, forward: null, left: null, up: null, samples: [],
    };
    return status();
  }

  function physicsStep(S, dt) {
    if (!R || !R.active) return;
    R.physics_step++;
    R.physics = physicsSnapshot(S, dt);
    R.physics.step = R.physics_step;
  }

  function frame(dt) {
    if (!R || !R.active || !Game || Game.mode !== 'run') return;
    R.acc += Math.max(0, Number(dt) || 0);
    const period = R.hz > 0 ? 1 / R.hz : 0;
    const first = !R.samples.length;
    if (!first && period > 0 && R.acc < period) return;
    if (period > 0) R.acc = first ? 0 : R.acc % period;
    const sample = makeSample(dt);
    if (sample) R.samples.push(sample);
  }

  function data() {
    if (!R) return null;
    return {
      format: 'cold-smoke-ski-v1',
      coordinate_system: 'FLU', position_unit: 'm',
      axes: {
        keypoints_world: 'run-local FLU metres: x forward, y left, z up',
        run_origin: R.origin ? round3(R.origin) : null,
        run_forward: R.forward ? round3(R.forward) : null,
        run_left: R.left ? round3(R.left) : null,
        run_up: R.up ? round3(R.up) : null,
      },
      keypoint_names: KEYPOINT_NAMES.slice(), measured_hz: R.hz || null, samples: R.samples.slice(),
    };
  }

  function status() { return { active: !!(R && R.active), samples: R ? R.samples.length : 0, physics_steps: R ? R.physics_step : 0, sample_hz: R ? R.hz : 0 }; }
  function stop() { if (R) R.active = false; return data(); }
  function clear() { R = null; }

  function download(name = 'cold-smoke-ski.json') {
    const payload = data();
    if (!payload || !payload.samples.length) throw new Error('SkiExport has no samples');
    if (typeof Blob === 'undefined' || typeof document === 'undefined') throw new Error('SkiExport.download requires a browser');
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    return { name, samples: payload.samples.length };
  }

  return { start, stop, clear, frame, physicsStep, data, status, download, get active() { return !!(R && R.active); } };
})();
