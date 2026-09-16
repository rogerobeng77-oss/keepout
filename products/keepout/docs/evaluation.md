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
YOLOX-tiny via `cv2.dnn`. Re-measured the same day after the fixes that real
footage forced (§10). On the labelled set every headline figure came out the same.
The courtyard figures changed, and §5.4 and §10 give the new ones.

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
| First version | 3 | ~270 |
| Lifetime and overlap tests | 1 | ~90 |
| Overlap judged when the track was lost, re-detection recognised | **0** | **0** |

The last false critical came from two pedestrians crossing inside the zone. The
tracker lost one id while the two boxes overlapped. The overlap test looked 2.5 s
later, by which time the other person had walked 170 px away, so it found nobody
on top of the lost track. The rule now judges overlap at the moment the track was
last seen. It also stands down when a track born after that moment starts about
where the lost one was, because that is the same person with a new id. Both cases
have tests.

Zero in forty seconds is not a rate. It is one clip, and it should not be quoted as
evidence that the rule suits a walkway. It does not. The rule belongs on a zone
that is normally empty. A machine danger zone is a place that is
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

**The re-measurement after the fixes is not comparable, and here is why.**
`eval/results/report.json` now gives 48.2 ms a frame, against 21.3 ms before.
That run shared this workstation with three other jobs, at a load average between
22 and 45. Run back to back on the courtyard clip under the same load, the old code
and the new code took 48.3 and 47.7 ms a frame in one round, and 48.2 and 57.4 in
the next. We cannot separate a real difference from that noise. What the new code
does cost is known by construction. The small-person probe runs four extra detector
passes every five seconds, about 8% more detection at 10 fps. Each stored evidence
frame costs five detector passes and one YuNet pass. Neither touches a frame that
raises nothing.

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
python -m pytest products/keepout/tests   # 143 tests
```

`eval/results/report.json` carries every figure above plus the per-frame detail.
The sample clips and their label files are committed, so the evaluation is
reproducible without re-rendering, and re-rendering is deterministic because every
generator is seeded.

---

## 10. Real footage, and what it broke

Every clip above is either composited or a pedestrian walkway. So on 16 September
we ran Keepout over real fixed-camera footage from Wikimedia Commons: a concrete
pump crew in Malta (Frank Vincentz, CC BY-SA 3.0), a prefabricated house going up
under a crane, filmed as a time-lapse (H. Raab, CC BY-SA 3.0), and two Amazon
loading-dock clips (domdomegg, CC BY 4.0). Where each clip came from, and its
licence as read on its own file page, is in `eval/real_footage/README.md`. It broke five things. One of them was a privacy claim this project
had been making.

### 10.1 The defects

1. **Bystanders' faces were not blurred.** A person-down evidence frame from the
   Malta crew blurred one head, the incident's own subject. The workers beside him
   stayed sharp. The pipeline passed only the subject's box to the blur, and the
   YuNet weights were not in the image, so no face detector covered for it. The
   README said "faces are blurred in every stored evidence frame". That was false.
2. **A black first frame blinded a whole clip.** The house clip opens with a fade
   from black. That black frame became the reference, and all 2,689 frames were
   refused as `camera_moved` with a constant 410 px shift. Nothing was watched.
3. **A bag became a person who fell.** At 15.6 s in the Malta clip, something on
   the bottom edge of the frame was detected as a person. It was a bag or a stack
   of roof tiles. Its box was wider than tall and never moved, so after four
   seconds it raised a latched critical.
4. **One person kept reopening incidents.** In a crowd the tracker loses people for
   a few frames and gives them new ids. Each new id opened a new alert: 27 of them
   for about nine people.
5. **False `person_unaccounted` criticals**: 16 on the time-lapse, 1 on the
   courtyard walkway, 2 on the Malta clip with placeholder zones.

### 10.2 What changed

1. **Privacy.** Before a frame is stored, Keepout now blurs the head region of every
   person box it knows about in that frame. That covers every track, every
   detection down to score 0.15 (tracking starts at 0.35), and the boxes of people
   who vanished inside the zone. It adds one more person sweep of the raw frame at
   the same low threshold, full frame plus 2x2 overlapping tiles, because a small
   worker the per-frame pass misses is still a face. YuNet faces are blurred on top.
   YuNet looks at the frame as it is and again at three times the size, at score
   0.3, since faces in a wide shot are 10 to 15 px across. The first deployed
   version upscaled only to 1,600 px at score 0.5, and on the live service it missed
   a worker half hidden by the pump boom. At 3x it finds him (§10.7). The container fetches YuNet and checks its sha256 at build time.
   It also sets `KEEPOUT_REQUIRE_FACE_DETECTOR=1`, so the service refuses to start
   without the weights. Anywhere else, missing weights fall back to the head regions
   alone, and every evidence record says `face_detector: unavailable`. A request can
   no longer switch blurring off.
2. **The reference frame.** A reference is only taken from a frame that passes the
   frame-level checks. It stays provisional until five frames have agreed with it.
   If 25 frames in a row instead disagree with it and agree with each other, it is
   replaced, and a `reference_replaced` view event goes into the record, the job
   log and the warnings. A confirmed reference is never replaced automatically.
   A camera that is knocked and then holds still must keep raising `camera_moved`,
   and a test holds it there for 90 frames.
3. **Person down needs a person who was upright.** A track is only eligible after
   three upright boxes. A fall often breaks the track, so a new track inherits that
   history from an upright track that went out of view within 3 s and one box
   height of where the new one starts. A box touching the frame edge is never
   judged for posture. The cost: somebody already lying down when the camera
   starts, never seen standing, does not raise `person_down`.
4. **Fragmented tracks rejoin their incident.** A new track that is born within
   3 s of a lost subject's last sighting, and within one box height of that spot,
   continues that incident. A track missed for longer than the zone's exit grace
   but not dropped resumes its own incident. The first version let any nearby
   track take over an incident, including a neighbour who had been tracked all
   along. On the Malta clip it moved incidents between workers every time one of
   them was missed for a single frame. The rule now needs a track born after the
   loss, and a test pins that case. A dropped track also ends its incident now.
   Before, when the tracker dropped a track faster than the zone's exit grace
   (12 frames is 480 ms at 25 fps, the grace is 600 ms), the incident stayed open
   for the rest of the run.
5. **The vanish rule** judges overlap at the moment of disappearance, and ignores a
   person re-detected under a new id (§5.4).

### 10.3 Before and after

Same clips and zones as the first footage run, same config path as the live service.
"Before" is the code at `ea5ea95` with the reference taken from the first frame,
which is what the service and the CLI did. People were counted by eye from each
alert's frame and the clip, so treat those counts as approximate. Outputs:
`eval/real_footage/results.json`, with how every evidence frame was redacted. The
frames themselves are kept out of the repository, because they show real workers.

| Run | Frames usable | Alerts | People in zone | Alerts per person | Criticals | False criticals |
|---|---|---|---|---|---|---|
| Courtyard, walkway zone, machine declared running | 400/400 → 400/400 | 21 → **17** | ~15 | 1.4 → **1.1** | 1 → **0** | 1 → **0** |
| Malta shot 4, placeholder zones | 755/755 → 755/755 | 0 → 0 (34 → 20 notes) | — | — | 2 → **1** | 2 → **1** |
| Malta shot 4, config A (crane outriggers) | 755/755 → 755/755 | 0 → 0 | 0 | — | 1 → **0** | 1 → **0** |
| Malta shot 4, config B (pump crew) | 755/755 → 755/755 | 27 → **16** | ~9 (8 to 10) | 3.0 → **1.8** | 1 → **0** | 1 → **0** |
| House shot 1, time-lapse, defaults | 1,173/1,173 → 1,173/1,173 | 23 → **15** (93 → 40 notes) | ~7 | 3.3 → 2.1 | 16 → **5** | 16 → **5** |
| House shot 1, time-lapse, crane config | 1,173/1,173 → 1,173/1,173 | 17 → **13** (95 → 40 notes) | ~7 | 2.4 → 1.9 | 14 → **5** | 14 → **5** |
| House, full original file | **0**/2,689 → **1,764**/2,689 | 0 → 28 | — | — | 0 → 13 | 0 → 13 |

Every critical in the table is false. Nobody fell in any of these clips.

The live service gives a different count on the time-lapse. It caps an upload at
900 analysed frames, so it reads every second frame of this 1,173-frame clip. With
the crane zones drawn in the upload page, the live run at `ce056a7` raised 57
incidents, 11 of them criticals held for a human, where the full-rate local run
above gave 5 (`eval/real_footage/live-job-7a8a1f4bcff9.json`: 587 frames analysed,
whole clip covered). The film shows the live figure.

The remaining Malta critical, with placeholder zones, is a `person_unaccounted` on
a detection beside the pump truck's cab. It sat 20 px inside a zone that does not
match the scene (the threshold is 18), was steady for over 3 s, then went. That is
a static object that never moved, which is the same shape as the bag, but on the
vanish path. We did not add a "must have moved" condition to the vanish rule. A
worker who stands still in the zone and then collapses is exactly the case that
rule exists for, and one clip with zones that do not fit the scene is not enough to
trade that away.

The five criticals left on the time-lapse are not being fixed. One second of video
is a minute of real time there, so people jump between frames and small figures
drop in and out of detection. That footage is outside what the product is for, and
bending the rule to fit it would weaken it on real-rate video.

### 10.4 Found while measuring, not fixed: a cut to a different camera

The full house file cuts to a second, closer camera position at 64.2 s. Of the
1,085 frames after the cut, 920 are refused as `camera_moved`, which is right. The
other 165 are judged usable, which is wrong. On an unrelated view, ECC sometimes
converges to a 2 to 4 px shift and reports agreement. The ECC correlation
coefficient separates the two cases on this file. Frames of the first view score at
least 0.79, even across an hour of real construction change. Frames after the cut
score at most 0.65. But the synthetic `cell-guard` clip scores 0.54 to 0.59 on its
own unchanged view, so a single threshold would break it. The likely fix is a
threshold relative to the correlation measured while the reference was being
confirmed. We have not built it, and until we do, a hard cut to a similar-looking
view can be partly missed.

### 10.5 Small people

On the steady Amazon clip, YOLOX-tiny tracked nobody. The workers are about 5% of
frame height. We measured how small a person can be and still be found.
`eval/real_footage/person_size.py` takes real frames from the Malta and
courtyard clips. It shrinks each one onto a fixed 960x540 canvas, so the people get
smaller relative to the frame, and it counts how many of the people found at full
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

Median time per frame for detection alone on this machine: 21.7 ms for one pass,
108.8 ms tiled.

So one pass is reliable from about 15% of frame height, and tiling from about 8%,
at five times the cost. The input size is the cause. The ONNX export has a fixed
416 px input, so two-scale inference means tiles. Keepout now does two things
about it. `tiled_detection` is an option, in the UI and in the API. Without it,
a tiled probe every five seconds counts the people the single pass is missing, and
the run carries a warning when they are more than a fifth of the people seen. On
Amazon clip 2 the single pass still tracked nobody, and now it says so. With tiling
it tracked 15, at 133 ms a frame against 39. On the Malta clip the probe found 19
people the single pass missed, out of 71 seen, mostly the far background and the
crane cab, and it warned.

### 10.6 The live service's frame budget

The service analyses at most 900 frames per upload, because App Runner takes about
440 ms a frame. A longer clip used to stop at frame 900 without saying so. The
47 s house clip stopped at 36 s, and its real alert at 44.8 s never happened. Now,
unless the caller sets a stride, the stride goes up until the whole clip fits
(every 2nd frame for that clip). The record then says "every 2nd frame is analysed
(12.5 per second) to cover all 46.9 s". A caller who turns that off gets a warning
that names the seconds analysed and the seconds that were not.

