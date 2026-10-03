# Keepout evaluation

## The headline

| | |
|---|---|
| Frames with somebody in the danger zone, seen | **819 / 819**, 95% Wilson CI [0.9955, 1.0] |
| Time to alert, 4 labelled entries | **0 ms** median and maximum |
| Escalation level correct | **4 / 4** |
| Incidents raised while the view was unusable | **0** across 228 blind frames |
| Kinds of blindness detected | **4 / 4** |
| Guard removal detected | **1.5 s** |
| Person down detected | **8.5 s** |
| Spurious incidents, empty room | **0** in 0.0111 camera-hours |
| Throughput | 26.3 ms/frame on 22 threads, 440 ms/frame on App Runner |

Measured 16 September 2026, OpenCV 5.0.0, `opencv-python-headless==5.0.0.93`,
YOLOX-tiny via `cv2.dnn`. Every figure is produced by `keepout evaluate`, which
writes `eval/results/report.json` plus one JSON file per clip; nothing here was
typed in by hand.

```bash
keepout build-samples          # stage the clips, about two minutes
keepout evaluate               # score them, about four minutes
```

**What this document covers.** Seven clips, 2,440 frames, 3 minutes 46 seconds of
video, six of them carrying frame-level ground truth. Section 8 adds four
unlabelled real construction clips, which can count false alarms but cannot
measure misses.

**No deployed-system outcome trial exists for this class of product.** Not for
Keepout, and not for any camera-based workplace safety system we could source.
Asked whether this prevents injuries in a real factory, the honest answer is that
nobody has published that evidence for any product in the category. The nearest
precedent is a NIOSH pilot that fitted warning lights to three forklifts in one
warehouse for four months and asked nine employees whether they felt safer
(Bobick et al., *Professional Safety* 2020, PMCID PMC11119981). All nine said yes.
That is perceived benefit from nine people. What follows measures something
smaller and checkable: whether the software does what it says on footage where
the answer is known.

---

## 1. The evaluation set

| Clip | Length | What is in it | Labels |
|---|---|---|---|
| `cell-alert` | 24 s | One person walks into the zone of a running conveyor, stands in it, walks out through the frame edge | exact |
| `cell-stopped` | 20 s | The identical approach with the belt stopped | exact |
| `cell-guard` | 24 s | The fixed guard is removed at 9.0 s while the belt runs | exact |
| `cell-down` | 30 s | One person collapses inside the zone at 9.0 s and does not move again | exact |
| `cell-unusable` | 32 s | The camera is nudged, then the lens is covered, then the lights go out, then the feed freezes | exact |
| `cell-quiet` | 40 s | A running machine, stopping and starting, with nobody in the room | exact |
| `courtyard` | 40 s | Unmodified pedestrian footage | none |

### 1.1 The six `cell-*` clips are composites

The **people are real**: person crops matted out of `samples/data/vtest.avi`, the
pedestrian clip in the OpenCV repository (Apache-2.0), by differencing each
detection box against the per-pixel median of the clip. Real human shapes, real
clothing, real motion blur, real compression artefacts.

The **machine cell is rendered** by `keepout.synth`: a concrete floor with slab
joints, hazard stripes, a conveyor whose belt cleats scroll when it is running, a
mesh guard panel that can be removed at a scripted frame, and per-frame sensor
noise. Because it is rendered, the truth is known exactly.

What a composite cannot test: factory lighting, flicker and hard shadow; steam,
coolant mist, dust, and a lens that fouls over weeks; high-visibility clothing,
which is what COCO's person class saw least of; real machine geometry, which
occludes far more than a rendered conveyor; and more than two people in frame.

`courtyard` is the counterweight: 40 seconds of unmodified footage, real camera,
real crowd, heavy mutual occlusion. It carries no frame labels, so it cannot score
detection rate, but it exposes a false-positive mode the staged clips never reach
(section 5.4).

---

## 2. Detection rate and time to alert

Of the frames the labels say somebody's feet were inside the danger zone, on how
many did Keepout report an occupant?

| Clip | Frames with somebody in the zone | Frames we saw them | Rate |
|---|---|---|---|
| `cell-alert` | 200 | 200 | 1.000 |
| `cell-stopped` | 159 | 159 | 1.000 |
| `cell-guard` | 59 | 59 | 1.000 |
| `cell-down` | 323 | 323 | 1.000 |
| `cell-unusable` | 78 | 78 | 1.000 |
| **Total** | **819** | **819** | **1.000** |

819 for 819 is a 95% Wilson interval of **[0.9955, 1.0]**. Read that as: on 819
labelled frames of composited footage at this scale and contrast it did not miss,
and the experiment is not large or varied enough to measure a miss rate below
about half a per cent. Frames are only scored when the view check passed; on
`cell-unusable` that excludes 228 of 384 frames, which is the point of that clip.

**Time to alert**, from the labelled frame where the feet crossed the boundary to
the timestamp on the incident:

| Clip | Labelled entry | Raised | Latency | Level asked for | Level given |
|---|---|---|---|---|---|
| `cell-alert` | 4.25 s | 4.25 s | **0 ms** | alert | alert |
| `cell-stopped` | 3.92 s | 3.92 s | **0 ms** | note | note |
| `cell-guard` | 2.33 s | 2.33 s | **0 ms** | alert | alert |
| `cell-down` | 2.17 s | 2.17 s | **0 ms** | alert | alert |

Zero is not a rounding of something small. There is no smoothing window on entry:
the frame that first satisfies the zone test is the frame that opens the incident,
so the only quantisation is the frame interval, 83 ms at 12 fps. It does **not**
include the latency of getting a frame off a real camera, which on RTSP is another
100 to 400 ms and is not ours to measure here.

A fifth labelled entry exists on `cell-unusable`, at a moment when the camera had
been nudged and the view check had already stopped the pipeline. It is excluded
here and reported in section 4.

---

## 3. False alerts in an empty room

`cell-quiet` is 40 seconds of running machinery in an empty room, starting and
stopping twice. Every incident it produces would be false. It produced **0**, over
0.0111 camera-hours.

0 events in 0.0111 hours implies an upper bound around 270 per camera-hour at 95%
confidence, so this rules out a grossly broken system and nothing more. A rate
worth quoting needs weeks of footage from a real installation. The measurement
with teeth is in section 5.4.

---

## 4. The honesty rails

`cell-unusable` scripts four ways of being blind, one after another.

| Problem | Scripted frames | Frames flagged | Recall |
|---|---|---|---|
| Camera moved | 72 | 72 | 1.00 |
| Lens blocked | 60 | 108 | 1.00 |
| Too dark | 48 | 48 | 1.00 |
| Frozen feed | 60 | 48 | 0.80 |

**Lens blocked over-fires by design.** 108 flagged against 60 scripted, because the
48 frames scripted as "too dark" also have almost no edge content. Both labels are
true of those frames and both lead to the same action: stop answering.

**Frozen feed has a start-up cost.** The test needs 12 consecutive frames below
threshold, so the first second of a freeze is missed by construction. That is the
price of not calling a quiet room a dead stream.

**The number that matters most here is zero.** Incidents raised while the view was
unusable: **0**, across 228 blind frames. That is the failure this product exists
to avoid — not a missed alert, but a confident silence that looks like safety.

### 4.1 The frozen-feed test keys on the mean, not the maximum

Re-encoding a genuinely frozen feed to H.264 and decoding it back produces
differences of up to **7 grey levels** frame to frame, purely from the codec, and
every real camera delivers compressed video. Measured at 320 px wide, mean
absolute difference per frame:

| Condition | mean | p95 | max |
|---|---|---|---|
| Frozen feed, after an H.264 round trip | 0.005 | 0.029 | 7 |
| Live, machine stopped, empty room | 0.075 | 0.409 | 7 |
| Live, very dark | 0.108 | 0.136 | 7 |
| Live, machine running | 2.639 | 2.921 | 86 |
| Live, pedestrian footage | 1.556 | 2.551 | 250 |

The mean separates frozen from live-but-static by about fifteen times where the
maximum does not separate them at all, so the test is a sustained low mean. A
scene genuinely motionless in front of a very low-noise sensor is still not
distinguishable from a frozen feed by pixels alone; a real deployment answers
that with the stream's own timestamps, which a video file does not carry.

---

## 5. Prone people, and the rule that replaced posture

### 5.1 YOLOX-tiny stops seeing people who are lying down

Body angle swept by rotating a matted person sprite and compositing it back onto
its own background, four subjects per angle:

| Body angle | 0° | 10° | 20° | 30° | 40° | **50°** | 60° | 70° | 80° | 90° |
|---|---|---|---|---|---|---|---|---|---|---|
| Detected | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | **0/4** | 0/4 | 1/4 | 2/4 | 1/4 |
| Mean score | 0.907 | 0.906 | 0.899 | 0.887 | 0.682 | — | — | 0.301 | 0.392 | 0.312 |

Detection collapses between 40 and 50 degrees. Sixteen trials per side is a small
experiment and the intervals are wide, but the effect is not subtle: confident
detections at 0.89 become no detection at all. It is a property of the training
data — COCO has very few people lying on the ground, so a detector trained on it
has learned that people are vertical.

**The consequence is severe and easy to miss.** Posture-based person-down detection
measures the aspect ratio of a detection box. No box, no aspect ratio, and the
highest escalation level silently never fires for the case it was built for.

### 5.2 Background contrast changes the answer

The sweep above ran against the courtyard's own pavement, close in tone to the
subjects. On the rendered cell floor, which has far higher contrast against a
dark-clothed person, the same 80-degree collapse **is** detected, at score 0.64 with
a width-to-height ratio of 2.39, for the whole 20 seconds after the fall. That is
why `cell-down` raises its critical through the posture path.

So the honest statement is not "the detector cannot see prone people". It is:
**whether a prone person is detected depends on how well they contrast with the
floor they are lying on, and an installation cannot promise that.**

### 5.3 The vanish rule

A confirmed track that was standing well inside the danger zone, and then stops
being detected without crossing the boundary, is escalated on its own as
`person_unaccounted`, at the same latching CRITICAL level. It is the inference a
human watching the monitor would make: they were there, now there is nobody, and
they cannot have left without walking out past a line we were watching.

Both paths are live and they cover each other. On `cell-down` the collapse at
9.00 s raised at 17.50 s — **8.5 s**, through the posture path.

8.5 seconds is slow on purpose: the posture rule requires the box to be non-upright
**and** essentially motionless for a sustained window, because a worker crouching to
clear a jam produces a wide box too. `tests/test_posture.py` holds a crouching
worker shuffling in place for ten seconds and asserts nothing is raised. The
alternative is an alarm every time somebody kneels, and that alarm gets switched
off within a week.

### 5.4 The vanish rule's false-positive rate on real footage

Nobody collapses in 40 seconds of a pedestrian walkway, so every
`person_unaccounted` incident `courtyard` produces is false.

| Version of the rule | False criticals in 40 s | Per camera-hour |
|---|---|---|
| Position and stillness only | 3 | ~270 |
| Plus a 3 s minimum track lifetime and an overlap test | 1 | ~90 |
| Plus overlap judged at the moment the track was lost, and re-detection recognised | **0** | **0** |

The three conditions, each with a test: the track must have existed for at least
3 seconds, so a detector blip is not a missing person; no other live track may have
been overlapping it at the moment it disappeared, because two people crossing is a
crowd and not an emergency; and a track born after that moment, starting about
where the lost one was, is the same person with a new id.

Zero in forty seconds is one clip, not a rate, and it is not evidence that the rule
suits a walkway. The rule belongs on a zone that is normally empty, entered by one
or two people at a time. On the five staged cell clips, where that holds, it
produced **0** false criticals.

---

## 6. The guard check

| Clip | Guard removed at | Raised at | Latency |
|---|---|---|---|
| `cell-guard` | 9.00 s | 10.50 s | **1.5 s** |

1.5 seconds is the deliberate 1.5 second state-persistence window plus one frame. A
guard does not come off and go back on in half a second, and requiring the change
to hold removes the flicker a single bad frame would cause.

**A lighting change is not a removed guard.** A test dims the scene 38% with the
guard still bolted in place. Raw Canny edge correlation collapses from 1.00 to 0.19
under that dimming, because the gradients fall below the fixed Canny thresholds:
edges are only lighting-invariant if the contrast survives the thresholding. The
guard check runs CLAHE local contrast equalisation before Canny, and the test
passes.

**A known false positive.** On `cell-down`, with the zones left at their defaults
rather than the clip's own, the collapsed body lies across about 34% of the guard
region — just under the 35% occlusion threshold that would return UNKNOWN. The
guard check sees a changed region and raises a guard incident alongside the correct
critical. The failure is conservative: the critical still fires, and the operator
gets one extra line in a list they are already reading because somebody is on the
floor. Both mitigations — a tighter guard region, or a higher occlusion threshold —
are configuration rather than code.

---

## 7. Throughput

Same code, same clips, three machines. This is the measurement that decides whether
the hosted demo can be live or has to be pre-computed.

| Where | Cores | YOLOX-tiny inference | Whole pipeline |
|---|---|---|---|
| Workstation, 22 threads | 22 | ~14 ms | **26.3 ms/frame** |
| Same box, container limited to 2 CPUs | 2 | 76 ms | ~90 ms/frame |
| **AWS App Runner, 4 vCPU** | 4 | **386 ms** | **440 ms/frame** |

Per-stage cost on the workstation, over all 2,440 frames:

| Stage | ms/frame | Share |
|---|---|---|
| Person detection (YOLOX-tiny in `cv2.dnn`) | 17.74 | 67% |
| View check (ECC, Canny, Laplacian, freeze) | 6.80 | 26% |
| Machine motion energy | 0.78 | 3% |
| Guard check | 0.71 | 3% |
| Posture | 0.15 | <1% |
| Escalation | 0.05 | <1% |
| Tracking (Hungarian assignment) | 0.05 | <1% |

**The tracker we had to write costs nothing.** OpenCV 5 removed the trackers from the
main wheel, so detections are associated with our own Hungarian assignment. 0.05 ms
a frame, 0.2% of the budget, and exact rather than greedy.

**App Runner's vCPUs are not desktop cores.** The same ONNX model in the same
`cv2.dnn` call is 14 ms on a 22-thread workstation and 386 ms on App Runner's 4
vCPU: a factor of 27, or about 5 per core. Sizing a CPU inference workload from a
laptop benchmark will be wrong by most of an order of magnitude. That is why every
bundled clip is analysed at **image build time** and served from disk, and why "run
this clip live" re-runs the same code on the instance at 440 ms a frame for anyone
who wants to watch it happen.

Real-time factor on the hosted instance is about 0.19x at 12 fps, so one App Runner
service cannot keep up with one live camera. A real deployment runs the vision at
the edge or on a GPU and uses a service like this for review.

---

## 8. Real fixed-camera footage

Four openly licensed clips from Wikimedia Commons: a concrete pump crew in Malta
(Frank Vincentz, CC BY-SA 3.0), a prefabricated house going up under a crane filmed
as a time-lapse (H. Raab, CC BY-SA 3.0), and two Amazon loading-dock clips
(domdomegg, CC BY 4.0), each listed with its licence in
`eval/real_footage/README.md`. Nobody falls in any of them, so **every critical these
runs raise is false**. Outputs are in `eval/real_footage/results.json`; the evidence
frames themselves stay out of the repository because they show real workers. People
in zone were counted by eye, so treat those counts as approximate.

| Run | Frames usable | Alerts | People in zone | Alerts per person | False criticals |
|---|---|---|---|---|---|
| Courtyard, walkway zone, machine declared running | 400/400 | 17 | ~15 | 1.1 | **0** |
| Malta shot 4, placeholder zones | 755/755 | 0 (20 notes) | — | — | 1 |
| Malta shot 4, config A (crane outriggers) | 755/755 | 0 | 0 | — | **0** |
| Malta shot 4, config B (pump crew) | 755/755 | **16** | ~9 (8 to 10) | 1.8 | **0** |
| House shot 1, time-lapse, defaults | 1,173/1,173 | 15 (40 notes) | ~7 | 2.1 | 5 |
| House shot 1, time-lapse, crane config | 1,173/1,173 | 13 (40 notes) | ~7 | 1.9 | 5 |
| House, full original file | **1,764**/2,689 | 28 | — | — | 13 |

The live service gives a different count on the time-lapse, because it caps an
upload at 900 analysed frames and so reads every second frame of this 1,173-frame
clip. With the crane zones drawn in the upload page, the live run at `ce056a7`
raised 57 incidents, 11 of them criticals held for a human, where the full-rate
local run gave 5 (`eval/real_footage/live-job-7a8a1f4bcff9.json`: 587 frames
analysed, whole clip covered).

**The reference frame is checked before it is trusted.** The house file opens with a
fade from black, and a black reference refuses every subsequent frame as
`camera_moved`. A reference is now only taken from a frame that passes the
frame-level checks, stays provisional until five frames agree with it, and is
replaced — with a `reference_replaced` event in the record — if 25 consecutive
frames disagree with it and agree with each other. A confirmed reference is never
replaced automatically, because a camera knocked out of position and then held
still looks exactly like that run of frames, and must keep refusing until a human
re-draws the zone. A test holds it there for 90 frames.

**The remaining Malta critical** (placeholder zones) is a `person_unaccounted` on a
detection beside the pump truck's cab: a static object 20 px inside a zone that
does not match the scene, steady for over 3 s, then gone. We did not add a "must
have moved" condition to the vanish rule. A worker who stands still in the zone and
then collapses is exactly the case that rule exists for, and one clip with zones
that do not fit the scene is not enough to trade that away.

**The five criticals on the time-lapse stay.** One second of video is a minute of
real time there, so people jump between frames and small figures drop in and out of
detection. That footage is outside what the product is for, and bending the rule to
fit it would weaken it on real-rate video.

### 8.1 A cut to a different camera is partly missed

The full house file cuts to a second, closer camera position at 64.2 s. Of the
1,085 frames after the cut, 920 are refused as `camera_moved`, which is right; the
other 165 are judged usable, which is wrong, because on an unrelated view ECC
sometimes converges to a 2 to 4 px shift and reports agreement. The ECC correlation
coefficient separates the two cases on this file — the first view scores at least
0.79 even across an hour of construction change, frames after the cut at most 0.65 —
but the synthetic `cell-guard` clip scores 0.54 to 0.59 on its own unchanged view,
so a single threshold would break it. The fix is a threshold relative to the
correlation measured while the reference was being confirmed: measured, not built.

### 8.2 How small a person can be

On the steady Amazon clip, YOLOX-tiny tracked nobody: the workers are about 5% of
frame height. `eval/real_footage/person_size.py` shrinks real Malta and courtyard
frames onto a fixed 960x540 canvas and counts how many of the people found at full
size are still found:

| Person height, share of frame | One pass (416 px input) | Full frame + 2x2 tiles |
|---|---|---|
| 4-6% | 14% | 58% |
| 6-8% | 36% | 77% |
| 8-10% | 60% | **95%** |
| 10-12% | 72% | 96% |
| 12-15% | 83% | 96% |
| **15-20%** | **96%** | 100% |
| 20-30% | 98% | 100% |

One pass is reliable from about 15% of frame height, tiling from about 8%, at five
times the cost: 21.7 ms against 108.8 ms median per frame for detection alone. The
cause is the input size — the ONNX export has a fixed 416 px input, so two-scale
inference means tiles. `tiled_detection` is an option in the UI and the API, and
without it a tiled probe every five seconds counts the people the single pass is
missing and warns when they exceed a fifth of the people seen. On Amazon clip 2 the
single pass tracked nobody and said so; with tiling it tracked 15, at 133 ms a frame
against 39. On the Malta clip the probe found 19 people the single pass missed, out
of 71 seen, mostly far background and the crane cab.

### 8.3 The live service's frame budget

The service analyses at most 900 frames per upload, because App Runner takes about
440 ms a frame. Unless the caller sets a stride, the stride rises until the whole
clip fits, and the record says so: "every 2nd frame is analysed (12.5 per second) to
cover all 46.9 s". A caller who turns that off gets a warning naming the seconds
analysed and the seconds that were not.

---

## 9. Reproducing this

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit -e products/keepout

keepout build-samples          # downloads vtest.avi, mattes sprites, renders clips
keepout evaluate               # writes products/keepout/eval/results/
python -m pytest products/keepout/tests   # 143 tests
```

`eval/results/report.json` carries every figure above plus the per-frame detail. The
sample clips and their label files are committed, so the evaluation reproduces
without re-rendering, and re-rendering is deterministic because every generator is
seeded.

---

## Limitations

1. **No outcome trial**, here or anywhere in this product class.
2. **No false-alert rate.** 0.0111 camera-hours of empty-room footage bounds nothing
   useful.
3. **The labelled set is composited.** Real people, rendered machine cell. The real
   construction clips cover high-visibility clothing and crews of eight to twelve,
   but carry no labels, so they count false alarms and cannot measure misses.
4. **Behaviour on real industrial machinery is untested**, and its geometry occludes
   far more than a rendered conveyor.
5. **Prone-person detection depends on floor contrast** and cannot be promised. The
   vanish rule covers it, at the false-positive cost in section 5.4, and
   `person_down` needs the person to have been seen upright first — so somebody
   already on the floor when the camera starts is only caught if they vanish.
6. **Small people are missed** below about 15% of frame height on one detector pass.
7. **A hard cut to a similar-looking camera can be partly missed** (section 8.1).
8. **Machine-running inference may not transfer.** Motion energy is calibrated per
   installation by `suggest_thresholds`, which reports when a running clip and a
   stopped clip are not separable at all from a given angle. On some angles they
   will not be, and then the answer is to wire in the machine's own run signal
   rather than infer it from pixels.
9. **Keepout does not check PPE, deliberately.** OSHA's 2024 PPE rulemaking says the
   dominant real-world failure is PPE that is present and ill-fitting, which looks
   correct on camera and does not protect. Presence detection is not effectiveness
   detection and must not be sold as it.
