# Keepout — technical report

A fixed camera over machinery, watched by something that never blinks.
Live: <https://mqrmfp6mbi.eu-west-2.awsapprunner.com>

---

## 0. Summary

Keepout watches one fixed camera over one machine. An operator draws a danger zone
once against a reference frame. A person entering that zone while the machine is
running raises an alert on the same frame, with the picture attached. A person down
and motionless raises a level that stays up until a human acknowledges it. Before
any of that, four checks decide whether the view can be trusted at all, and when it
cannot, Keepout names the problem and measures nothing.

| | |
|---|---|
| Frames with somebody in the zone, seen | **819 / 819**, 95% CI [0.9955, 1.0] |
| Time to alert | **0 ms** median and maximum, 4 of 4 labelled entries |
| Escalation level correct | **4 / 4** |
| Incidents raised while blind | **0** across 228 frames |
| View problems detected | **4 / 4** kinds |
| Guard removal detected | **1.5 s** |
| Person down detected | **8.5 s** |
| Throughput | 26 ms/frame on 22 threads, 440 ms/frame on App Runner |

Method, per-clip detail and reproduction: [evaluation.md](evaluation.md). The
labelled set is composited — real people matted out of OpenCV's `vtest.avi`, placed
into a rendered machine cell — and four unlabelled real construction clips from
Wikimedia Commons check it against footage nobody staged.

---

## 1. The problem

On 9 September 2026 the UK Health and Safety Executive published this, about a
worker at Factory Services UK Limited in Knowsley whose arm had been pulled into a
moving conveyor:

> "There was no guard in place and no emergency stop button in the area. Working
> alone at the time, there was nobody nearby to see or hear what had happened. In
> an effort to raise the alarm, he repeatedly waved at a CCTV camera in the hope
> that someone monitoring the system would spot him and come to his aid, nobody
> did."

Source: HSE press release, 9 September 2026.
<https://press.hse.gov.uk/2026/09/09/manufacturer-fined-after-worker-suffers-life-changing-injuries-in-conveyor-incident/>

Read that as an engineering statement. The camera was installed, pointed at the
right place, and recording. A man with one working arm waved at it deliberately and
repeatedly, and the system did nothing, because the only thing that could turn
those pixels into a response was a person looking at a monitor, and no person was.
Every component of that system worked except the one that was never automated.

### 1.1 Scale

OSHA's Commonly Used Statistics page: "There were 5,283 fatal work injuries in 2023
(3.5 fatalities per 100,000 full-time equivalent workers)."
<https://www.osha.gov/data/commonstats>

**Machine Guarding, 29 CFR 1910.212, is on OSHA's FY2024 top ten most-cited
standards list.** That list is the regulator's own account of what inspectors find
wrong most often, and it maps almost one-to-one onto things a fixed camera can see:
missing fall protection, unguarded machinery, powered industrial trucks near
people.

### 1.2 Periodic inspection is a control that works on inspection days

From OSHA's release on Orchids Builders LLC, 23 July 2026:

> "OSHA cited the employer for two willful and four repeat violations and proposed
> $349,754 in penalties. Orchids Builders LLC has been inspected seven times since
> 2023 and all the cases included fall protection violations."

Seven inspections, every one finding the same class of violation. Enforcement is
recent and heavy on both sides of the Atlantic — in the UK, £650,000 plus £40,000
costs for a fatal trench collapse on 9 September 2026, £400,000 for a forklift
crushing on 13 August — and none of it put a pair of eyes on the machine between
visits. Figures are as published, pounds where the HSE stated pounds and dollars
where OSHA stated dollars, with no conversion applied.

### 1.3 The case that argues against us

The same 9 September HSE release on the trench collapse records that the contracts
manager had **already seen another worker in the same unsupported excavation
earlier that day and did nothing**. He was sentenced to 10 months' imprisonment,
suspended for 18 months.

A camera would have detected exactly what a human supervisor had already detected
and ignored. The gap in that case was enforcement, not perception, and Keepout
would have added nothing. That bounds the claim. Keepout closes the gap where
**nobody was looking**, which is the Knowsley conveyor case: a man alone, in front
of a camera with no viewer. It does nothing about the gap where somebody looked and
chose not to act.

---

## 2. Users

**The product is for the person who is alone** — the worker at Knowsley, with nobody
nearby to see or hear. Not a safety manager reviewing a dashboard, and not a
compliance auditor.

Three people touch it. **The technician or line supervisor** who installs it draws
the danger zone once against a reference frame, confirms the guard is in place so it
can be learned, and calibrates the machine-running threshold from one running clip
and one stopped clip — every step is a polygon on a still image. **Whoever is in the
control room**, or carrying the phone, sees a level, a reason in a sentence, and the
frame that proves it, and presses acknowledge on the one state that will not clear
itself. **The safety manager**, afterwards, reads a log where every entry has a
timestamp, a duration, a machine state and an image, rather than a count.

What none of them ever sees is a name (section 6).

---

## 3. Architecture

Diagrams in [architecture.md](architecture.md). In prose: frames in, incidents with
evidence out, and a hard stop in front of everything else that refuses to answer
from a view it cannot trust.

```
frame
  -> view check .......... camera moved? lens blocked? too dark? feed frozen?
     |                     if any: report it, raise nothing, stop here
  -> person detection .... YOLOX-tiny, ONNX, cv2.dnn, COCO person class only
  -> association ......... Hungarian assignment + constant-velocity prediction
  -> machine state ....... motion energy in the machine polygon, people excluded
  -> guard state ......... CLAHE edge + appearance correlation vs a reference
  -> posture ............. box aspect ratio + stillness over a window
  -> zone occupancy ...... signed distance to the feet, with hysteresis
  -> escalate ............ note / guard / alert / critical, evidence written
```

The pipeline performs no IO. It takes frames and returns state; the service layer
decides where evidence bytes go. That is what makes the synthetic tests possible:
they build sequences whose answer is known by construction and assert on the
returned state, with no files and no model.

---

## 4. The OpenCV 5 implementation

`opencv-python-headless==5.0.0.93`, pinned, asserted at import and again in the
Docker build. OpenCV 4.14.0 was released *after* 5.0.0, so an unpinned install
resolves to 4.x and the entry silently fails the competition's one hard
requirement.

### 4.1 Person detection: cv2.dnn, ONNX only

OpenCV 5 removed `readNetFromCaffe` and `readNetFromDarknet` entirely. Everything
must be ONNX. That rules out YOLOv4's public-domain Darknet weights, and the
AGPL-3.0 licence on the Ultralytics family rules those out for a hosted demo,
because AGPL section 13 extends copyleft to network use.

**YOLOX-tiny** (Megvii, Apache-2.0), official ONNX export, at 416x416:

```python
net = cv2.dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_AUTO)
```

The decode is written out rather than taken from a library, because YOLOX is
anchor-free and the details are where the bugs live: centres are grid-relative and
scale by the stride, sizes are exponentiated, and the letterbox pads bottom-right
only. Getting the padding convention wrong puts every box slightly in the wrong
place, which for this product means slightly the wrong side of a zone boundary.
Only the COCO `person` class is kept.

### 4.2 Tracking: OpenCV 5 removed the trackers, so we wrote one

`cv2.TrackerCSRT`, `cv2.TrackerKCF` and the whole `cv2.legacy` namespace are gone
from the main wheel in 5.0. `opencv-contrib-python` has them but conflicts with the
main wheel and doubles the image, so detections are associated directly.

The method is detection-to-track assignment by minimum total cost, solved exactly
with the **Hungarian algorithm** (Jonker-Volgenant shortest-augmenting-path form, in
`track.py`), with a constant-velocity predictor so a person missed for a frame or
two keeps their identity. Cost blends IoU distance with centre distance: IoU alone
loses a fast walker whose boxes stop overlapping, centre distance alone swaps two
people who pass each other.

Exact rather than greedy, for a reason specific to this product: two workers near a
machine, one steps into the zone, and a greedy pass can hand the entering box to the
wrong track — an incident, with an evidence frame, attached to a person who never
entered. `tests/test_track.py` checks the solver against brute-force enumeration on
random matrices. It costs **0.05 ms per frame**, 0.2% of the budget.

### 4.3 Zone geometry

`cv2.pointPolygonTest` with `measureDist=True` gives signed distance in pixels,
positive inside, which is what the hysteresis thresholds are expressed in.

The contact point is the **bottom-centre of the box**, not the centroid. A floor
polygon drawn on a floor is a statement about where feet may go; a person standing
beside a press with a centroid outside the zone and a hand inside it is a different
question, covered by an optional box-overlap mode.

Hysteresis is not optional. Entry and exit use different thresholds and exit also
requires a grace period, because a person whose feet oscillate two pixels across a
boundary would otherwise generate forty incidents in four seconds;
`tests/test_zones.py` runs that oscillation for sixty frames and asserts one entry.

Zone proposal uses `cv2.accumulate` over frame differences, `cv2.threshold`,
`cv2.morphologyEx`, `cv2.connectedComponentsWithStats` for the largest moving
region, `cv2.dilate` for a standoff margin, and `cv2.convexHull` plus
`cv2.approxPolyDP` for the polygon. It refuses, with a reason, when the scene is
still, when the region is too small, or when most of the frame is moving.

### 4.4 Machine running, from motion energy

Mean absolute frame difference inside the machine polygon, on a blurred, downscaled
grey image so compression noise does not read as motion.

**Person exclusion.** Detected person boxes are zeroed before measuring. Without it,
a worker walking past a stopped machine makes it read as running, precisely when
somebody is next to it. `tests/test_machine.py` asserts both that exclusion
suppresses that and that the scene really does trigger without exclusion, so the
test cannot pass vacuously.

**Hysteresis plus dwell.** Machinery has dead spots — an indexing table is still
between indexes — so the state only changes after the signal has held for
`min_state_ms`.

The absolute value is meaningless across cameras, so `suggest_thresholds` fits them
from a labelled running clip and a labelled stopped clip, **and reports when the two
are not separable at all** from that angle. When they overlap, the answer is to wire
in the machine's own run signal rather than infer it.

### 4.5 Guard checking

Two independent comparisons against a reference captured when an operator confirmed
the guard was in place, combined by taking the **weaker** of the two, so a guard is
called present only when both agree. Asymmetric on purpose: a false "guard missing"
costs somebody walking over to look; a false "guard present" costs an arm.

- **Structure**: normalised correlation of Canny edge maps, after
  `cv2.createCLAHE` local contrast equalisation. Without CLAHE, dimming a scene 38%
  with the guard still bolted in place collapses edge correlation from 1.00 to 0.19
  and the checker calls a present guard missing — edges are only lighting-invariant
  if the contrast survives the thresholding.
- **Appearance**: zero-mean normalised cross-correlation of the grey patch, the
  quantity `cv2.matchTemplate` computes for `TM_CCOEFF_NORMED`, done over an
  arbitrary mask so occluded pixels are excluded rather than averaged in.

Occlusion returns UNKNOWN and holds the previous state, because calling "guard
removed" every time somebody walks in front of it would destroy trust in a week.

### 4.6 Person down, and the measurement that changed the design

The highest level uses two signals a detector already gives for free: the **aspect
ratio** of the box, and **stillness**, measured as peak centre displacement over a
window in units of the box diagonal so it is scale-invariant. Both must agree for a
sustained period.

Then we measured whether the detector can see a person on the floor at all.
Rotating a matted person sprite through body angle:

| Body angle | 0 | 20 | 40 | **50** | 70 | 90 |
|---|---|---|---|---|---|---|
| Detection rate | 1.00 | 1.00 | 1.00 | **0.00** | 0.25 | 0.25 |
| Mean score | 0.91 | 0.90 | 0.68 | — | 0.30 | 0.31 |

Detection collapses between 40 and 50 degrees, and a person lying on the floor is
near 90. **No box means no aspect ratio, and the product's highest escalation level
silently never fires for exactly the case it was built for.** This is a property of
COCO, which contains very few people lying down, not a bug — and not absolute: on a
floor with higher contrast against the subject the same 80-degree collapse *is*
detected, at score 0.64, which is why `cell-down` raises through the posture path.
An installation cannot promise contrast.

So Keepout stopped relying on seeing them. A confirmed track that was standing well
inside the zone and then stops being detected, without crossing the boundary, is
escalated on its own as `person_unaccounted`, at the same latching CRITICAL level.
It is the inference a human watching the monitor would make: they were there, now
there is nobody, and they cannot have left without walking out past a line we were
watching. The rule's own false-positive rate is in evaluation.md section 5.4; it
belongs on a zone that is normally empty, not on a camera pointed at a crowd.

**Upright first.** A fall needs somebody who was standing, so a track becomes
eligible for "down" only after three upright boxes, and a new track inherits that
history from an upright track that left view within 3 s and one box height. A box
touching the frame edge is never judged for posture, because a person cut off by the
edge of the picture is a wide box too — and so is a bag at the bottom of a real
construction frame. The price: somebody already lying down when the camera starts,
and never seen standing, does not raise `person_down`.

### 4.7 The honesty rails

Four checks before anything else, in `viewcheck.py`:

- **Camera moved**: `cv2.findTransformECC` with a Euclidean model against the
  reference the zone was drawn on, reporting the actual translation and rotation.
  Non-convergence is itself evidence and falls back to `cv2.phaseCorrelate`. We
  deliberately do **not** warp the zone into the new view: fitting a homography to a
  partly-changed scene is confident nonsense. A human re-draws it.
- **Lens blocked**: Canny edge fraction plus Laplacian standard deviation.
- **Too dark**: median luminance plus collapsed dynamic range, so night footage with
  working lights passes and a black frame does not.
- **Frozen feed**: a sustained low *mean* absolute difference, not a low maximum. An
  H.264 round trip of a genuinely frozen feed produces differences of up to **7 grey
  levels**, so a maximum-difference test never fires. The mean separates frozen from
  live-but-static by about fifteen times.

Across 228 blind frames in the degraded clip, **zero** zone incidents were raised.

**The reference itself is checked.** A reference is only taken from a frame that
passes the frame-level checks, and stays provisional until five frames agree with
it. If instead 25 consecutive frames disagree with it and agree with each other, the
reference was the odd one out: it is replaced, with a `reference_replaced` event in
the record. A confirmed reference is never replaced automatically, because a camera
knocked out of position that then holds still looks exactly like that run of frames
and must keep refusing until a human re-draws the zone. On a Commons time-lapse that
opens with a fade from black, this is the difference between 0 and 1,764 usable
frames out of 2,689. The gap it does not close: after a hard cut to a
similar-looking second camera, ECC sometimes converges on a small false shift, and
165 of the 1,085 frames after the cut were judged usable (evaluation.md 8.1).

### 4.8 Other OpenCV 5 specifics

- `cv2.FontFace("sans")` for frame annotation, new in 5, rendering UTF-8 through a
  TrueType engine.
- `VideoCapture.get()` returns **-1** for unsupported properties in 5 where 4.x
  returned 0. Every property read in `visioncore.imageio` treats anything below zero
  as missing.
- `cv2.FaceDetectorYN` is present in the main wheel and is used **only to destroy
  information** (section 6). `cv2.FaceRecognizerSF`, which would turn a face into an
  embedding, is never loaded.

---

## 5. AWS deployment

Image built for `linux/amd64` (App Runner has no arm64 runtime), pushed to ECR,
served by App Runner at 4 vCPU / 8 GB with managed TLS and a stable URL. Evidence
frames mirror to S3 under an instance role scoped to `s3:PutObject` on one prefix;
logs are one JSON object per line to CloudWatch. The service is in **eu-west-2**
because the account is limited to two App Runner services per region and both
us-east-1 slots hold unrelated live services; ECR was duplicated into eu-west-2
because App Runner cannot pull across regions.

**Every bundled clip is analysed at image build time.** `keepout demo --bake` runs
the real pipeline over all seven clips during `docker build` and writes the results
and evidence frames into the image, so a cold container shows a real alert in under
two seconds. Not a fixture: the same code runs, and "run this clip live" re-executes
it on the instance. The reason it is necessary is a measurement — the same ONNX
model in the same `cv2.dnn` call runs at **14 ms/frame on a 22-thread workstation
and 386 ms/frame on App Runner's 4 vCPU**, a factor of 27, or about 5 per core.

---

## 6. Privacy, by construction

**Keepout does not know who anybody is.** No face recognition anywhere:
`cv2.FaceRecognizerSF` is never loaded and no embedding is ever computed, so adding
recognition would need a new dependency. Track numbers are per-run integers that
reset when the process restarts and are never joined to a roster, a badge or an
image of a face.

Before an evidence frame is written, the head region of **every person detected in
that frame** is blurred: every track, every detection down to score 0.15 (tracking
starts at 0.35), the boxes of people who vanished inside the zone, and a tiled
low-threshold sweep of the raw frame, because a small worker the per-frame pass
misses is still a face. Every face `cv2.FaceDetectorYN` (YuNet, MIT) finds is
blurred as well, at native size and again at three times the size, since faces in a
wide shot are 10 to 15 px across. The detector is used only to decide what to
destroy. The container fetches and checks YuNet at build time and refuses to start
without it; elsewhere the head regions are still blurred and the record says
`face_detector: unavailable`. Blurring is not a request parameter — a client that
asks for it off is ignored. The limit, stated rather than implied: **a person no
detector finds is not blurred.**

Blur rather than a black box, because a redacted frame still has to let a supervisor
see what the person was doing, which is the whole point of evidence.

A camera watching workers is surveillance unless the design makes it structurally
incapable of being one, and that argument has to be checkable in the source. The
test that holds it honest puts several people in one frame — tracked, untracked,
below the tracking threshold, and found only by the sweep — and checks that every
head region lost its detail.

---

## 7. Evaluation

Full method, per-clip detail and reproduction in [evaluation.md](evaluation.md);
the headline is in section 0. 2,440 frames across seven clips, six carrying
frame-level ground truth generated at the same moment the pixels were.

Time to alert of 0 ms is not a rounding: there is no smoothing window on entry, so
the frame that first satisfies the zone test is the frame that opens the incident,
and the only quantisation is the frame interval. It excludes the latency of getting
a frame off a real camera, which on RTSP is another 100 to 400 ms.

**The measurement that argues against us.** On 40 seconds of unmodified, crowded
pedestrian footage the person-unaccounted rule produces **0** false criticals. Zero
on one clip is not a rate, and the rule still does not belong on a camera pointed at
a crowd.

**Real footage.** On four fixed-camera construction clips from Wikimedia Commons:
the Malta pump crew raises 16 alerts for about nine people in the zone, 1.8 per
person, with no false criticals; the house time-lapse with the crane zone raises 13
alerts and 5 false criticals, which we have not bent the rule to remove, because one
second of that video is a minute of real time; the handheld Amazon dock clip is
refused almost entirely as `camera_moved`. Nobody fell in any of them, so every
critical they raise is false. One YOLOX-tiny pass finds 96% of people at 15-20% of
frame height and 36% at 6-8%; full frame plus 2x2 tiles reaches 95% down to 8-10%,
at five times the cost.

---

## 8. Limitations

The full list, with the numbers behind each, is at the end of
[evaluation.md](evaluation.md). The ones that bound what this entry claims:

1. **No outcome trial exists**, for Keepout or for any camera-based workplace safety
   product in any domain we could source. The nearest precedent, a NIOSH pilot that
   fitted warning lights to three forklifts and asked nine employees whether they
   felt safer, is perceived benefit from nine people.
2. **Detection and enforcement are different problems** (section 1.3).
3. **One service cannot keep up with one live camera.** 440 ms/frame on App Runner is
   about 0.19x real time at 12 fps; a real deployment runs the vision at the edge or
   on a GPU and uses a service like this for review.
4. **The labelled set is composited**, and the real construction footage has no
   labels — so it counts false alarms and cannot measure misses. Neither tests
   factory lighting, steam, coolant, dust or real machine occlusion, and 0.0111
   camera-hours of empty-room footage bounds no false-alert rate.
5. **Prone-person detection depends on floor contrast** and cannot be promised. The
   vanish rule covers it, at the false-positive cost in section 7; person-down needs
   the person to have been seen upright first, and takes 8.5 deliberate seconds.
6. **Small people are missed** below about 15% of frame height on one detector pass,
   a hard cut to a similar-looking camera can be partly missed, and some camera
   angles cannot separate a running machine from a stopped one at all —
   `suggest_thresholds` says so rather than emitting a coin flip.
7. **Keepout does not check PPE, deliberately.** OSHA's 2024 PPE rulemaking says the
   dominant real-world failure is PPE that is present and ill-fitting, which looks
   correct on camera and does not protect. A camera flags absence, not
   effectiveness.

---

## 9. Responsible use

**Keepout counts events and measures geometry.** It does not identify, rank
individuals, measure productivity, or retain anything that could be joined to a
person. That is enforced by what the code does not contain, not by a policy
document.

**It must not be used for discipline.** An incident log that becomes a disciplinary
record stops being a safety tool within a week, because the people it is meant to
protect start working around it, and a system that is worked around is worse than no
system: it creates the appearance of cover.

**Alerting is not guarding.** A detection is a request for a human to look, not a
substitute for a fixed guard, an interlock or an emergency stop. The Knowsley case
turned on the absence of a guard *and* the absence of an emergency stop: a camera
would have summoned help faster, and the guard would have prevented the injury.
Keepout is the last line, and the last line is the worst place to put your only
control.

**Say when it cannot see.** The four honesty rails exist because a safety system that
goes quiet when it goes blind is more dangerous than no system at all. That is why
the refusal state is a designed screen rather than a degraded one, and why
"incidents raised while blind" is a first-class number in the evaluation. Workers
should be told the camera is there, what it computes, what is stored and for how
long, and that no identity is derived.
