const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/20_ski_export.js'), 'utf8');

function record(heading) {
  const joints = Object.fromEntries([
    'head', 'neck', 'shL', 'shR', 'elbowL', 'elbowR', 'wristL', 'wristR',
    'pelvis', 'hipL', 'hipR', 'kneeL', 'kneeR', 'ankleL', 'ankleR',
  ].map(name => [name, [0, 1, 0]]));
  joints.neck = [0, 1.5, 0];
  joints.hipL = [-.15, 1, 0];
  joints.hipR = [.15, 1, 0];
  const state = {
    p: [0, 0, 0], v: [2, -.5, -5], n: [.6, .8, 0],
    t: [0, -.3, -1], r: [1, 0, 0], h: 0, contact: true,
  };
  const context = vm.createContext({
    Game: { root: { fwd: heading }, s: state, mode: 'run', runT: 10 },
    Skier: { world: { root: {}, J: joints }, toWorld: joint => joint },
  });
  vm.runInContext(source + '\nglobalThis.exporter = SkiExport;', context);
  const exporter = context.exporter;
  exporter.start({ sampleHz: 0 });
  exporter.physicsStep(state, .01);
  exporter.frame(.04);
  // Move the complete skeleton downhill, with a lateral component and descent.
  for (const joint of new Set(Object.values(joints))) {
    joint[0] += 2; joint[1] -= .5; joint[2] -= 5;
  }
  state.p = [2, -.5, -5]; state.h = -.5;
  context.Game.runT = 10.07;
  exporter.physicsStep(state, .01);
  exporter.frame(.07);
  return JSON.parse(JSON.stringify(exporter.stop()));
}

test('downhill export is independent of the initial skier heading', () => {
  const downhill = record([0, 0, -1]);
  for (const heading of [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 1, 0]]) {
    assert.deepEqual(record(heading), downhill);
  }
  assert.equal(downhill.axes.reference, 'course_fall_line');
  assert.deepEqual(downhill.axes.run_forward, [0, 0, -1]);
  assert.deepEqual(downhill.axes.run_left, [-1, 0, 0]);
  assert.deepEqual(downhill.axes.run_up, [0, 1, 0]);
  assert.deepEqual(downhill.samples[1].human_pos, [5, -2, -.5]);
  assert.deepEqual(downhill.samples[1].hip_centre_vel, [5, -2, -.5]);
  assert.deepEqual(downhill.samples[1].ground.position, [5, -2, -1.5]);
  assert.deepEqual(downhill.samples[1].ground.normal, [0, -.6, .8]);
  assert.equal(downhill.samples[1].ground_z, -1.5);
  assert.equal(downhill.samples[1].t, 10.07);
  assert.equal(downhill.samples[1].keypoints_world.hip_centre[0], 5);
});
