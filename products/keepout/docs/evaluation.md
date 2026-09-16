# Evaluation

Every number here is produced by `keepout evaluate`, which writes
`eval/results/report.json` plus one JSON file per clip. Nothing in this document
was typed in by hand; if a figure looks wrong, rerun the command and the file
will disagree with me rather than with itself.

```bash
keepout build-samples          # stage the clips, about two minutes
keepout evaluate               # score them, about four minutes
```

Measured 16 September 2026, OpenCV 5.0.0, `opencv-python-headless==5.0.0.93`,
YOLOX-tiny via `cv2.dnn`.

---

## 0. The thing to read first

**No deployed-system outcome trial exists for this class of product.** Not for
Keepout, and not for any camera-based workplace safety system we could find. If a
judge asks whether this has been shown to prevent injuries in a real factory, the
honest answer is that nobody has published that evidence, for any product in this
category. What follows measures whether the software does what it says on footage
where we know the answer. That is a different and much smaller claim.

The closest thing to a precedent in the literature is a NIOSH pilot that
retrofitted warning lights to three forklifts in one warehouse for four months
and asked nine employees whether they felt safer (Bobick et al., *Professional
Safety* 2020, PMCID PMC11119981). All nine said yes. That is perceived benefit
from nine people, not measured incident reduction. It is precedent, not proof.

---

## 1. The evaluation set

Seven clips, 2,440 frames, 3 minutes 46 seconds of video. Six carry frame-level
ground truth generated at the same moment the pixels were, so the labels cannot
drift from the footage.

| Clip | Length | What is in it | Labels |
|---|---|---|---|
| `cell-alert` | 24 s | One person walks into the zone of a running conveyor, stands in it, walks out through the frame edge | exact |
| `cell-stopped` | 20 s | The identical approach with the belt stopped | exact |
| `cell-guard` | 24 s | The fixed guard is removed at 9.0 s while the belt runs | exact |
| `cell-down` | 30 s | One person collapses inside the zone at 9.0 s and does not move again | exact |
| `cell-unusable` | 32 s | The camera is nudged, then the lens is covered, then the lights go out, then the feed freezes | exact |
| `cell-quiet` | 40 s | A running machine, stopping and starting, with nobody in the room | exact |
| `courtyard` | 40 s | Unmodified real pedestrian footage | none |

### 1.1 Where the pixels come from, and what that costs us

The six `cell-*` clips are **composites**, and this document says so everywhere
rather than in a footnote.

* The **people are real**. Person crops are matted out of `samples/data/vtest.avi`,
  the pedestrian clip that ships in the OpenCV repository under Apache-2.0, by
  differencing each detection box against the per-pixel median of the clip. Real
  human shapes, real clothing, real motion blur, real compression artefacts. The
  detector has to earn every detection.
* The **machine cell is rendered** by `keepout.synth`: a concrete floor with slab
  joints, hazard stripes, a conveyor whose belt cleats genuinely scroll when it is
  running, a mesh guard panel that can be removed at a scripted frame, and
  per-frame sensor noise. Because it is rendered we know the truth exactly.

What a composite cannot tell you, and what would have to be measured on a real
installation before anybody deployed this:

- real factory lighting, including mixed sodium and LED, flicker, and hard shadow;
- steam, coolant mist, airborne dust, and a lens that fouls slowly over weeks;
- high-visibility clothing, which is exactly the colour and texture COCO's person
  class saw least of;
- real machine geometry, which occludes people far more than a rendered conveyor;
- more than two people in frame at once, in a space designed for work.

`courtyard` is the counterweight: 40 seconds of entirely unmodified real footage,
real camera, real crowd, heavy mutual occlusion. It carries no frame labels, so it
cannot score detection rate, but it can and does expose a false-positive mode that
the staged clips never reach (§5).

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

819 for 819 is a 95% Wilson interval of **[0.9955, 1.0]**. The honest reading of
that is not "this never misses". It is "on 819 labelled frames of composited
footage at this scale and contrast, it did not miss, and the experiment is not
large or varied enough to measure a miss rate below about half a percent."

Frames are only scored when the view check passed. On `cell-unusable` that
excludes 228 of 384 frames, which is the point of that clip.

**Time to alert**, from the labelled frame where the feet crossed the boundary to
the timestamp on the incident:

| Clip | Labelled entry | Raised | Latency | Level asked for | Level given |
|---|---|---|---|---|---|
| `cell-alert` | 4.25 s | 4.25 s | **0 ms** | alert | alert |
| `cell-stopped` | 3.92 s | 3.92 s | **0 ms** | note | note |
| `cell-guard` | 2.33 s | 2.33 s | **0 ms** | alert | alert |
| `cell-down` | 2.17 s | 2.17 s | **0 ms** | alert | alert |

Four entries, four alerted on the same frame the labels say the boundary was
crossed, all four at the right level. Median 0 ms, maximum 0 ms, all four inside
the one-second requirement.

Zero is not a rounding of something small. The pipeline has no smoothing window on
entry: the frame that first satisfies the zone test is the frame that opens the
incident, so the only quantisation is the frame interval, 83 ms at 12 fps. What
this measurement does **not** include is the latency of getting the frame off a
real camera and into the process, which on an RTSP stream is typically another
100 to 400 ms and is not ours to measure here.

A fifth labelled entry exists on `cell-unusable`, at a moment when the camera had
been nudged and the view check had already stopped the pipeline. It is excluded
from the latency figures and reported separately in §4, because scoring it would
be scoring the wrong thing.

---

## 3. False alerts

`cell-quiet` is 40 seconds of running machinery in an empty room, starting and
stopping twice. Every incident it produces is false.

| Measure | Value |
|---|---|
| Camera-hours | 0.0111 |
| Spurious incidents | **0** |
| False alerts per camera-hour | **0.0** |

Zero over one hundredth of a camera-hour is a weak result and should be read as
such: it rules out a grossly broken system, and it rules out nothing else. A
useful false-alert rate needs weeks of real footage from a real installation, and
we do not have it. The upper bound implied by 0 events in 0.0111 hours is roughly
270 per camera-hour at 95% confidence, which is to say the experiment is too
short to bound anything.

The measurement that does have teeth is in §5.

---

## 4. The honesty rails

`cell-unusable` scripts four ways of being blind, one after another.

| Problem | Scripted frames | Frames we flagged | Recall |
|---|---|---|---|
| Camera moved | 72 | 72 | 1.00 |
| Lens blocked | 60 | 108 | 1.00 |
| Too dark | 48 | 48 | 1.00 |
| Frozen feed | 60 | 48 | 0.80 |

All four detected, none missed. Two things to read carefully:

**Lens blocked over-fires, on purpose.** 108 flagged against 60 scripted, because
the 48 frames scripted as "too dark" also have almost no edge content and are
flagged as blocked as well. Both labels are true of those frames, and both lead to
the same action, which is to stop answering.

**Frozen feed has a start-up cost.** The test needs 12 consecutive frames below
threshold before it will call a feed frozen, so the first second of a freeze is
missed by construction. That is the price of not calling a quiet room a dead
stream.

**The number that matters most in this section is zero.** Incidents raised while
the view was unusable: **0**. Across 228 blind frames the pipeline never once
produced a zone incident against a polygon it could no longer trust. That is the
failure this product exists to avoid: not a missed alert, but a confident silence
that looks like safety.

### 4.1 The frozen-feed test had to be rebuilt after measuring it

The first implementation keyed on the **maximum** absolute difference between
consecutive frames, reasoning that a real sensor always has noise and a duplicated
frame has none. It never fired. Re-encoding a genuinely frozen feed to H.264 and
decoding it back produces differences of up to **7 grey levels** frame to frame,
purely from the codec, and every real camera delivers compressed video.

Measured at 320 px wide, mean absolute difference per frame:

| Condition | mean | p95 | max |
|---|---|---|---|
| Frozen feed, after an H.264 round trip | 0.005 | 0.029 | 7 |
| Live, machine stopped, empty room | 0.075 | 0.409 | 7 |
| Live, very dark | 0.108 | 0.136 | 7 |
| Live, machine running | 2.639 | 2.921 | 86 |
| Live, real pedestrian footage | 1.556 | 2.551 | 250 |

The **mean** separates frozen from live-but-static by about fifteen times where
the maximum does not separate them at all. The test is now a sustained low mean.

The limit that remains: a scene that is genuinely motionless in front of a very
low-noise sensor is not distinguishable from a frozen feed by pixels alone. In a
real deployment that is answered by the stream's own timestamps or sequence
numbers, which a video file does not carry.

---

## 5. The failure case that shaped the design

### 5.1 YOLOX-tiny stops seeing people who are lying down

A person on the floor is rotated roughly 90 degrees from upright. We swept body
angle by rotating a matted person sprite and compositing it back onto its own
background, four subjects per angle:

| Body angle | Detected | Rate | Mean score |
|---|---|---|---|
| 0° | 4 / 4 | 1.00 | 0.907 |
| 10° | 4 / 4 | 1.00 | 0.906 |
| 20° | 4 / 4 | 1.00 | 0.899 |
| 30° | 4 / 4 | 1.00 | 0.887 |
| 40° | 4 / 4 | 1.00 | 0.682 |
| **50°** | **0 / 4** | **0.00** | — |
| 60° | 0 / 4 | 0.00 | — |
| 70° | 1 / 4 | 0.25 | 0.301 |
| 80° | 2 / 4 | 0.50 | 0.392 |
| 90° | 1 / 4 | 0.25 | 0.312 |

Detection collapses between 40 and 50 degrees. Sixteen trials per side of that
boundary is a small experiment and the confidence intervals are wide, but the
effect is not subtle: confident detections at 0.89 become no detection at all.

This is a property of the training data, not a bug. COCO has very few people lying
on the ground, so a detector trained on it has learned that people are vertical.

**The consequence is severe and it is easy to miss.** Posture-based person-down
detection measures the aspect ratio of a detection box. If there is no box, there
is no aspect ratio, and the highest escalation level in the product silently never
fires for exactly the case it was built for.

### 5.2 Background contrast changes the answer

The sweep above was run against the courtyard's own pavement, which is close in
tone to the subjects. On the rendered cell floor, which has far higher contrast
against a dark-clothed person, the same 80-degree collapse **is** detected, at
score 0.64 with a width-to-height ratio of 2.39, for the whole 20 seconds after
the fall. That is why `cell-down` raises its critical through the posture path.

So the honest statement is not "the detector cannot see prone people". It is:
**whether a prone person is detected depends on how well they contrast with the
floor they are lying on, and that is not something an installation can promise.**

### 5.3 What we did about it

We stopped relying on seeing them.

A confirmed track that was standing well inside the danger zone, and then stops
being detected without crossing the boundary, is escalated on its own as
`person_unaccounted`, at the same latching CRITICAL level. The reasoning is the
one a human watching the monitor would use: they were there, now there is nobody,
and they cannot have left without walking out past a line we were watching.

Both paths are live, and they cover each other:

| Clip | Collapse at | Raised at | Latency | Raised by |
|---|---|---|---|---|
| `cell-down` | 9.00 s | 17.50 s | **8.5 s** | `person_down` (posture) |

8.5 seconds is slow, and it is slow on purpose. The posture rule requires the box
to be non-upright **and** essentially motionless for a sustained window before it
will fire, because a worker crouching to clear a jam produces a wide box too.
`tests/test_posture.py` holds a crouching worker who shuffles in place for ten
seconds and asserts that nothing is raised. The alternative to an 8.5 second delay
is an alarm every time somebody kneels down, and that alarm gets switched off
within a week.

### 5.4 The vanish rule's own false-positive rate, measured on real footage

This is where `courtyard` earns its place. Nobody collapses in 40 seconds of a
pedestrian walkway, so every `person_unaccounted` incident it produces is false.

The first version of the rule produced **three** false criticals in those 40
seconds, a rate of about 270 per camera-hour, which is unusable. The cause was
visible in the evidence frames: people passing behind each other and behind street
furniture, and short-lived detector blips that appeared and vanished.

Two additional conditions fixed most of it, and both are now tested:

1. the track must have existed for at least 3 seconds, so a detector blip is not
   a missing person;
2. no other live track may have been overlapping it when it disappeared, because
   two people crossing is a crowd, not an emergency.

| Version | False criticals in 40 s | Per camera-hour |
|---|---|---|
| Before | 3 | ~270 |
| After | **1** | **~90** |

One in forty seconds of a busy walkway is still far too high for that setting, and
this document is not going to pretend otherwise. What it is honest to say is that
the setting is wrong for the rule. A machine danger zone is a place that is
normally empty, entered by one or two people at a time, with no crowd to occlude
anybody. On the five staged cell clips, where that is the case, the rule produced
**0** false criticals. The courtyard number is the one to quote when asking
whether this belongs on a camera pointed at a busy thoroughfare. It does not.

---

## 6. The guard check

| Clip | Guard removed at | Raised at | Latency | Result |
|---|---|---|---|---|
| `cell-guard` | 9.00 s | 10.50 s | **1.5 s** | correct |

1.5 seconds is the deliberate 1.5 second state-persistence window plus one frame.
A guard does not come off and go back on in half a second, and requiring the
change to hold removes the flicker that a single bad frame would otherwise cause.

**A lighting change is not a removed guard, and getting that right needed a fix.**
A test dims the scene 38% with the guard still bolted in place. The first
implementation called it missing: raw Canny edge correlation collapsed from 1.00 to
0.19, because the gradients fell below the fixed Canny thresholds. Edges are only
lighting-invariant if the contrast survives the thresholding. The guard check now
runs CLAHE local contrast equalisation before Canny, and the same test passes.

**A known false positive.** On `cell-down`, when the danger and guard zones are
left at their defaults rather than the clip's own, the collapsed body lies partly
across the guard region and covers about 34% of it, just under the 35% occlusion
threshold that would return UNKNOWN. The guard check then sees a changed region
and raises a guard incident alongside the correct critical. The failure is
conservative, the critical still fires, and the operator gets one extra line in a
list they are already reading because somebody is on the floor. The mitigations
are to draw the guard region tighter than the floor a person can fall on, or to
raise the occlusion threshold, and both are configuration rather than code.

---

## 7. Throughput

Same code, same clips, three machines. This is the measurement that decides
whether the hosted demo can be live or has to be pre-computed.

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

Two things worth saying plainly.

**The tracker we had to write costs nothing.** OpenCV 5 removed the trackers from
the main wheel, so detections are associated with our own Hungarian assignment.
It is 0.05 ms per frame, 0.2% of the budget, and it is exact rather than greedy.
The cost of writing it was an afternoon; the cost of running it is nothing.

**App Runner's vCPUs are not desktop cores.** The same ONNX model in the same
`cv2.dnn` call is 14 ms on a 22-thread workstation and 386 ms on App Runner's 4
vCPU, a factor of 27. Per core it is still about a factor of 5. Anyone sizing a
CPU inference workload from a laptop benchmark will be wrong by most of an order
of magnitude. This is why every bundled clip is analysed at **image build time**
and served from disk: a judge sees a real alert in under two seconds, and the
"run this clip live" button re-runs the same code on the instance for anyone who
wants to watch it happen at 440 ms a frame.

Real-time factor on the hosted instance is about 0.19x at 12 fps, which means one
App Runner service cannot keep up with one live camera. A real deployment would
run the vision on the edge or on a machine with a GPU, and use this service for
review. That is stated in `docs/report.md` under limitations, not hidden here.

---

## 8. What this evaluation does not establish

1. That Keepout prevents injuries. No outcome trial exists, here or anywhere in
   this product class.
2. A false-alert rate. 0.0111 camera-hours of empty-room footage bounds nothing.
3. Behaviour with more than two people in frame, at real factory scale.
4. Behaviour on real industrial machinery, whose geometry occludes far more than a
   rendered conveyor.
5. Behaviour in high-visibility clothing, which is under-represented in COCO.
6. Whether the machine-running inference transfers. Motion energy is calibrated
   per installation by `suggest_thresholds`, which reports when a running clip and
   a stopped clip are not separable at all from a given camera angle. On some
   angles they will not be, and then the honest answer is to wire in the machine's
   own run signal rather than infer it from pixels.
7. Anything about PPE. Keepout does not check it, deliberately: OSHA's own 2024
   rulemaking says the dominant real-world failure is PPE that is present and
   ill-fitting, which looks correct on camera and does not protect. Presence
   detection is not effectiveness detection and must not be sold as it.

---

## 9. Reproducing this

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit -e products/keepout

keepout build-samples          # downloads vtest.avi, mattes sprites, renders clips
keepout evaluate               # writes products/keepout/eval/results/
python -m pytest products/keepout/tests   # 121 tests
```

`eval/results/report.json` carries every figure above plus the per-frame detail.
The sample clips and their label files are committed, so the evaluation is
reproducible without re-rendering, and re-rendering is deterministic because every
generator is seeded.
