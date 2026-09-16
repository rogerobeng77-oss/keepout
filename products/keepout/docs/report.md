# Keepout — technical report

OpenCV AI Competition 2026. Live endpoint: <https://mqrmfp6mbi.eu-west-2.awsapprunner.com>

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

Read that again as an engineering statement. The camera was installed. It was
pointed at the right place. The footage existed and was being recorded. A man with
one working arm waved at it deliberately, repeatedly, and the system did nothing,
because the only thing that could turn those pixels into a response was a person
looking at a monitor, and no person was.

Every component of that system worked except the one that was never automated.

### 1.1 Scale

OSHA's own Commonly Used Statistics page:

> "There were 5,283 fatal work injuries in 2023 (3.5 fatalities per 100,000
> full-time equivalent workers)."

Source: <https://www.osha.gov/data/commonstats>

**Machine Guarding, 29 CFR 1910.212, is on OSHA's FY2024 top ten most-cited
standards list.** That list is the regulator's own account of what inspectors find
wrong most often, and it maps almost one-to-one onto things a fixed camera can
see: missing fall protection, unguarded machinery, powered industrial trucks near
people.

### 1.2 Periodic inspection is not working, said by the regulator

From OSHA's release on Orchids Builders LLC, 23 July 2026:

> "OSHA cited the employer for two willful and four repeat violations and proposed
> $349,754 in penalties. Orchids Builders LLC has been inspected seven times since
> 2023 and all the cases included fall protection violations."

Seven inspections. Every one of them found the same class of violation. Inspection
is a control that works on the days an inspector is present.

The enforcement is real and recent on both sides of the Atlantic:

| Employer | Hazard | Penalty | Date |
|---|---|---|---|
| T Vaughan Limited | Trench collapse, fatal | £650,000 + £40,000 costs | 9 Sep 2026 |
| Appledorn Developments Limited | same incident | £500,000 | 9 Sep 2026 |
| Clearaway Recycling Limited | Forklift, worker crushed | £400,000 + £10,259 costs | 13 Aug 2026 |
| Smith & Nephew Medical Ltd | Forklift load, uncontrolled | £230,000 + £111,000 costs | 14 Sep 2026 |
| Orchids Builders LLC | Fall protection, repeat | $349,754 proposed | 23 Jul 2026 |

Figures are as published: pounds where the HSE stated pounds, dollars where OSHA
stated dollars. No conversion has been applied.

HSE on the waste and recycling sector, in the Clearaway release: fatal incidents
there are "ten times more likely than the all-industry average", and "the waste
and recycling industry saw six workers killed during 2025/26."

### 1.3 The case that argues against us, stated up front

The same 9 September 2026 HSE release on the trench collapse records that the
contracts manager, Anthony O'Connor, had **already seen another worker in the same
unsupported excavation earlier that day and did nothing**. He was sentenced to 10
months' imprisonment, suspended for 18 months.

A camera would have detected exactly what a human supervisor had already detected
and ignored. The gap in that case was enforcement, not perception. Keepout would
have added nothing.

This matters because it bounds the claim. Keepout closes the gap where **nobody
was looking**, which is the Knowsley conveyor case: a man alone, in front of a
camera with no viewer. It does nothing about the gap where somebody looked and
chose not to act. Those are different failures and only one of them is a computer
vision problem.

---

## 2. Users

**The person the product is for is the one who is alone.** Not a safety manager
reviewing a dashboard, and not a compliance auditor. The worker at Knowsley,
working alone, with nobody nearby to see or hear.

Three people actually touch it:

- **The maintenance technician or line supervisor** who installs it. Draws the
  danger zone once against a reference frame, confirms the guard is in place so it
  can be learned, and calibrates the machine-running threshold from one running
  clip and one stopped clip. Everything is a polygon on a still image; there is no
  configuration file to write.
- **Whoever is in the control room**, or carrying the phone the alerts go to. Sees
  a level, a reason in a sentence, and the frame that proves it. Presses
  acknowledge on the one state that will not clear itself.
- **The safety manager**, afterwards. Reads an incident log where every entry has a
  timestamp, a duration, a machine state and an image, rather than a count.

What none of them ever sees is a name. Keepout does not know who anybody is, by
construction (section 6).

---

## 3. Architecture

Diagrams in [architecture.md](architecture.md). In prose: frames in, incidents
with evidence out, and a hard stop in front of everything else that refuses to
answer from a view it cannot trust.

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
must be ONNX. That rules out loading YOLOv4's public-domain Darknet weights
directly, and the AGPL-3.0 licence on the Ultralytics family rules those out for a
hosted demo, because AGPL section 13 extends copyleft to network use.

**YOLOX-tiny** (Megvii, Apache-2.0), official ONNX export, at 416x416:

```python
net = cv2.dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_AUTO)
```

The decode is written out rather than taken from a library, because YOLOX is
anchor-free and the details are where the bugs live: centres are grid-relative and
scale by the stride, sizes are exponentiated, and the letterbox pads bottom-right
only, so un-scaling a box is a single divide with no offset. Getting the padding
convention wrong puts every box slightly in the wrong place, which for this
product means slightly the wrong side of a zone boundary.

Only the COCO `person` class is kept. Nothing else in the frame is of interest and
carrying 79 other classes through the tracker would be noise.

### 4.2 Tracking: OpenCV 5 removed the trackers, so we wrote one

`cv2.TrackerCSRT`, `cv2.TrackerKCF` and the whole `cv2.legacy` namespace are gone
from the main wheel in 5.0. `opencv-contrib-python` has them but conflicts with the
main wheel and doubles the image, so detections are associated directly.

The method is detection-to-track assignment by minimum total cost, solved exactly
with the **Hungarian algorithm** (Jonker-Volgenant shortest-augmenting-path form,
implemented in `track.py`), with a constant-velocity predictor so a person missed
for a frame or two keeps their identity. Cost blends IoU distance with centre
distance: IoU alone loses a fast walker whose boxes stop overlapping, centre
distance alone swaps two people who pass each other.

Exact rather than greedy, and the reason is specific to this product. Two workers
near a machine, one steps into the zone, and a greedy pass can hand the entering
box to the wrong track. That is an incident, with an evidence frame, attached to a
person who never entered. `tests/test_track.py` checks the solver against
brute-force enumeration on random matrices, and separately checks that two people
crossing keep their own identities.

It costs **0.05 ms per frame**, 0.2% of the budget.

### 4.3 Zone geometry

`cv2.pointPolygonTest` with `measureDist=True` gives signed distance in pixels,
positive inside, which is exactly what the hysteresis thresholds are expressed in.

The contact point is the **bottom-centre of the box**, not the centroid. A floor
polygon drawn on a floor is a statement about where feet may go; a person standing
beside a press with a centroid outside the zone and a hand inside it is a
different question, covered by an optional box-overlap mode.

Hysteresis is not optional. Entry and exit use different thresholds and exit
additionally requires a grace period, because a person whose feet oscillate two
pixels across a boundary would otherwise generate forty incidents in four seconds.
`tests/test_zones.py` runs exactly that oscillation for sixty frames and asserts
one entry.

Zone proposal uses `cv2.accumulate` over frame differences, `cv2.threshold`,
`cv2.morphologyEx`, `cv2.connectedComponentsWithStats` to take the largest moving
region, `cv2.dilate` to add a standoff margin, and `cv2.convexHull` plus
`cv2.approxPolyDP` for the polygon. It refuses, with a reason, when the scene is
still (the machine was not running), when the region is too small, or when most of
the frame is moving (the camera is not fixed).

### 4.4 Machine running, from motion energy

Mean absolute frame difference inside the machine polygon, on a blurred,
downscaled grey image so compression noise does not read as motion. Two pieces of
discipline:

**Person exclusion.** Detected person boxes are zeroed before measuring. Without
it, a worker walking past a stopped machine makes it read as running, precisely
when somebody is next to it. `tests/test_machine.py` builds that scene and asserts
both that exclusion suppresses it and that the scene really does trigger without
exclusion, so the test cannot pass vacuously.

**Hysteresis plus dwell.** Machinery has dead spots: an indexing table is still
between indexes. The state only changes after the signal has held for
`min_state_ms`.

The absolute value is meaningless across cameras, so `suggest_thresholds` fits
them from a labelled running clip and a labelled stopped clip, **and reports when
the two are not separable at all** from that angle. When they overlap the honest
answer is to wire in the machine's own run signal rather than infer it.

### 4.5 Guard checking, and a wrong assumption that a test caught

Two independent comparisons against a reference captured when an operator
confirmed the guard was in place, combined by taking the **weaker** of the two,
so a guard is called present only when both agree. Asymmetric on purpose: a false
"guard missing" costs somebody walking over to look; a false "guard present" costs
an arm.

- **Structure**: normalised correlation of Canny edge maps.
- **Appearance**: zero-mean normalised cross-correlation of the grey patch, the
  quantity `cv2.matchTemplate` computes for `TM_CCOEFF_NORMED`, done over an
  arbitrary mask so occluded pixels are excluded rather than averaged in.

The edge comparison was wrong in its first version, and a test found it. Dimming
the scene 38% with the guard still bolted in place collapsed edge correlation from
1.00 to **0.19**, and the checker called a present guard missing. Edges are only
lighting-invariant if the contrast survives the thresholding; below the Canny
thresholds they simply vanish. The fix is `cv2.createCLAHE` local contrast
equalisation before Canny.

Occlusion returns UNKNOWN and holds the previous state, because calling "guard
removed" every time somebody walks in front of it would destroy trust in a week.

### 4.6 Person down, and the measurement that changed the design

The highest level is "a person is down and not moving". It uses two signals a
detector already gives for free: the **aspect ratio** of the box, and **stillness**,
measured as peak centre displacement over a window in units of the box diagonal so
it is scale-invariant. Both must agree for a sustained period.

Then we measured whether the detector can see a person on the floor at all.
Rotating a matted person sprite through body angle:

| Body angle | 0 | 20 | 40 | **50** | 70 | 90 |
|---|---|---|---|---|---|---|
| Detection rate | 1.00 | 1.00 | 1.00 | **0.00** | 0.25 | 0.25 |
| Mean score | 0.91 | 0.90 | 0.68 | — | 0.30 | 0.31 |

Detection collapses between 40 and 50 degrees. A person lying on the floor is near
90. **If there is no box, there is no aspect ratio, and the product's highest
escalation level silently never fires for exactly the case it was built for.**

This is a property of COCO, which contains very few people lying down, not a bug.
It is also not absolute: on a floor with higher contrast against the subject the
same 80-degree collapse *is* detected, at score 0.64, which is why the `cell-down`
clip raises through the posture path. But an installation cannot promise contrast.

So Keepout stopped relying on seeing them. A confirmed track that was standing
well inside the zone and then stops being detected, without crossing the boundary,
is escalated on its own as `person_unaccounted`, at the same latching CRITICAL
level. It is the inference a human watching the monitor would make: they were
there, now there is nobody, and they cannot have left without walking out past a
line we were watching.

That rule then had to be tamed against real footage, which is in section 7.

**Upright first.** Real construction footage (Malta, Wikimedia Commons) showed the
posture path failing the other way. A bag or a stack of roof tiles on the bottom
edge of the frame was detected as a person. Its box was wider than tall and never
moved, and four seconds later it raised a latched critical. A fall needs somebody
who was standing, so a track now becomes eligible for "down" only after three
upright boxes. A fall usually breaks the track, so a new track inherits that
history from an upright track that went out of view within 3 s and one box height.
A box touching the frame edge is never judged for posture, because a person cut
off by the edge of the picture is a wide box too. The price is stated in the code
and here: somebody already lying down when the camera starts, and never seen
standing, does not raise `person_down`.

### 4.7 The honesty rails

Four checks before anything else, in `viewcheck.py`:

- **Camera moved**: `cv2.findTransformECC` with a Euclidean model against the
  reference the zone was drawn on, reporting the actual translation and rotation.
  Non-convergence is itself evidence and falls back to `cv2.phaseCorrelate`. We
  deliberately do **not** warp the zone into the new view: fitting a homography to
  a partly-changed scene is confident nonsense. A human re-draws it.
- **Lens blocked**: Canny edge fraction plus Laplacian standard deviation.
- **Too dark**: median luminance plus collapsed dynamic range, so night footage
  with working lights passes and a black frame does not.
- **Frozen feed**: rebuilt after measuring it. The obvious test keys on the
  *maximum* difference between consecutive frames, and it never fires, because
  real cameras deliver compressed video and an H.264 round trip of a genuinely
  frozen feed produces differences of up to **7 grey levels**. The mean separates
  frozen from live-but-static by about fifteen times where the maximum does not
  separate them at all. The test is now a sustained low mean.

Across 228 blind frames in the degraded clip, **zero** zone incidents were raised.

**The reference itself has to be checked.** A Commons time-lapse opens with a fade
from black. The black frame became the reference, and all 2,689 frames of the clip
were refused as `camera_moved`. A reference is now only taken from a frame that
passes the frame-level checks. It stays provisional until five frames agree with
it. If instead 25 consecutive frames disagree with it and agree with each other,
the reference was the odd one out. It is replaced, and a `reference_replaced` view
event goes into the record. A confirmed reference is never replaced
automatically, because a camera knocked out of position that then holds still
looks exactly like that run of frames, and it must keep refusing until a human
re-draws the zone. That full file now has 1,764 usable frames. It also exposed a
gap we have not closed: after a hard cut to a similar-looking second camera, ECC
sometimes converges on a small false shift, and 165 of the 1,085 frames after the
cut were judged usable (evaluation.md §10.4).

### 4.8 Other OpenCV 5 specifics

- `cv2.FontFace("sans")` for frame annotation, which is new in 5 and renders UTF-8
  through a TrueType engine.
- `VideoCapture.get()` returns **-1** for unsupported properties in 5 where 4.x
  returned 0. Every property read in `visioncore.imageio` treats anything below
  zero as missing.
- `cv2.FaceDetectorYN` is present in the main wheel and is used **only to destroy
  information** (section 6). `cv2.FaceRecognizerSF`, which would turn a face into
  an embedding, is never loaded.

---

## 5. AWS deployment

Image built for `linux/amd64` (App Runner has no arm64 runtime), pushed to ECR,
served by App Runner at 4 vCPU / 8 GB with managed TLS and a stable URL. Evidence
frames mirror to S3 under an instance role scoped to `s3:PutObject` on one prefix.
Logs are one JSON object per line to CloudWatch. Full inventory, prices and the
deployment log are in [costs.md](costs.md); it comes to about **$48/month**, or
$1.58 a day.

Two decisions worth defending.

**The service is in eu-west-2, not us-east-1.** The account is restricted to two
App Runner services per region and both us-east-1 slots hold unrelated live
services. ECR was duplicated into eu-west-2 because App Runner cannot pull across
regions; the S3 bucket stays in us-east-1 and the container is told so.

**Every bundled clip is analysed at image build time.** `keepout demo --bake` runs
the real pipeline over all seven clips during `docker build` and writes the results
and evidence frames into the image, so a cold container shows a real alert in under
two seconds. This is not a fixture: the same code runs, and "run this clip live"
re-executes it on the instance for anyone who wants to watch.

The reason it is necessary is a measurement. The same ONNX model in the same
`cv2.dnn` call runs at **14 ms/frame on a 22-thread workstation and 386 ms/frame
on App Runner's 4 vCPU**, a factor of 27, or about 5x per core. Anyone sizing CPU
inference from a laptop benchmark will be wrong by most of an order of magnitude.

---

## 6. Privacy, by construction

**Keepout does not know who anybody is.**

- No face recognition anywhere. `cv2.FaceRecognizerSF` is never loaded and no
  embedding is ever computed. Adding recognition would require a new dependency.
- Track numbers are per-run integers that reset when the process restarts and are
  never joined to a roster, a badge or an image of a face.
- Before an evidence frame is written, the head region of **every person detected
  in that frame** is blurred. That covers every track, every detection down to
  score 0.15, and a tiled low-threshold sweep of the raw frame. Every face
  `cv2.FaceDetectorYN` (YuNet, MIT) finds is blurred as well. The detector is used
  only to decide what to destroy. The image fetches and checks YuNet at build time
  and refuses to start without it. Elsewhere the head regions still get blurred,
  and the record says `face_detector: unavailable`. The limit, stated rather than
  implied: a person no detector finds is not blurred.
- **This used to be overstated.** Until 16 September this section said faces were
  blurred in every stored frame. On a crowded real construction clip, only the
  incident's own subject was blurred, and the workers beside him stayed sharp in
  the evidence. The pipeline had handed the blur only the subject's box, and the
  YuNet weights were not in the image. The new behaviour is covered by a test that
  puts several people in frame (tracked, untracked, below the tracking threshold,
  and found only by the sweep) and checks that every head region lost its detail.
  It was also checked by eye on the same Malta clip, locally and on the live
  service.
- Blurring is not a request parameter. A client that asks for it off is ignored.
- The UI shows the blur state on every evidence frame and in the sidebar, so
  nobody has to take it on trust.

Blur rather than a black box, because a redacted frame still has to let a
supervisor see what the person was doing, which is the whole point of evidence.

This is not a feature list. A camera watching workers is surveillance unless the
design makes it structurally incapable of being one, and the argument that it is
incapable has to be checkable in the source.

---

## 7. Evaluation

Full detail, method and reproduction in [evaluation.md](evaluation.md).
2,440 frames across seven clips; six carry frame-level ground truth generated at
the same moment the pixels were.

| | |
|---|---|
| Detection rate inside the zone | **819 / 819 frames**, 95% CI [0.9955, 1.0] |
| Time to alert | **0 ms** median and maximum, 4 of 4 labelled entries |
| Escalation level correct | **4 / 4** |
| False alerts, empty room | **0** in 0.0111 camera-hours |
| View problems detected | **4 / 4** kinds |
| Incidents raised while blind | **0** |
| Guard removal detected | **1.5 s** |
| Person down detected | **8.5 s** |
| Throughput | 26 ms/frame on 22 threads, 440 ms/frame on App Runner |

Time to alert of 0 ms is not a rounding: there is no smoothing window on entry, so
the frame that first satisfies the zone test is the frame that opens the incident.
The only quantisation is the frame interval. It does not include the latency of
getting a frame off a real camera, which on RTSP is another 100 to 400 ms.

**The measurement that argues against us.** On 40 seconds of unmodified, crowded
pedestrian footage, the first person-unaccounted rule produced three false
criticals, about 270 per camera-hour. A minimum track lifetime and an overlap test
brought that to one. Judging the overlap at the moment the track was lost, rather
than 2.5 s later, brought it to zero. Zero on one 40 s clip is not a rate, and the
rule still does not belong on a camera pointed at a crowd.

**Real footage.** Run over fixed-camera construction clips from Wikimedia Commons,
Keepout showed five defects: sharp bystander faces in evidence, a black first frame
that blinded a whole clip, a false person-down on a bag, 27 alerts for about nine
people as tracks fragmented, and false vanish criticals. All five are fixed, and
the before-and-after table is evaluation.md §10.3. On the Malta pump crew, alerts
went from 27 to 16 (3.0 to 1.8 per person) and false criticals from 1 to 0. On the
house time-lapse with the crane zone, false criticals went from 14 to 5. Every
critical in those runs was false, since nobody fell.

**People must be big enough.** One YOLOX-tiny pass at its 416 px input finds 96%
of people at 15-20% of frame height, 72% at 10-12% and 36% at 6-8%. Full frame plus
2x2 tiles finds 95% down to 8-10%, at five times the cost. Tiling is an option, and
without it a probe warns when the single pass is missing people. On the steady
Amazon dock clip the single pass tracked nobody and said so. Tiled, it tracked 15.

---

## 8. Limitations

1. **No outcome trial exists.** Not for Keepout and not for any camera-based
   workplace safety product in any domain we could source. If asked whether this
   has been shown to prevent an injury in the field, the honest answer is that
   nobody has published that evidence. The nearest precedent, a NIOSH pilot that
   fitted warning lights to three forklifts and asked nine employees whether they
   felt safer, is perceived benefit from nine people over four months in one
   warehouse. It is precedent, not proof.
2. **Detection and enforcement are different problems.** Section 1.3: in the trench
   collapse a supervisor saw the hazard and did nothing. Keepout closes the gap
   where nobody was looking, not the gap where somebody looked and ignored it.
3. **One service cannot keep up with one live camera.** 440 ms/frame on App Runner
   is about 0.19x real time at 12 fps. A real deployment runs the vision at the
   edge or on a GPU and uses a service like this for review.
4. **The evaluation set is composited.** Real people, rendered machine cell. It
   does not test factory lighting, steam, coolant, dust, high-visibility clothing,
   real machine occlusion, or more than two people in frame. Unlabelled real
   construction footage (section 7) has since covered high-visibility clothing and
   crews of eight to twelve. With no labels, it can count false alarms but not
   misses.
5. **The false-alert rate is not established.** 0.0111 camera-hours of empty-room
   footage bounds nothing useful.
6. **Prone-person detection depends on floor contrast** and cannot be promised.
   The vanish rule covers it, at the cost of the false-positive rate in section 7.
   Person-down also needs the person to have been seen upright first, so someone
   already on the floor when the camera starts is only caught if they vanish.
7. **Person-down takes 8.5 seconds**, deliberately, so that a worker crouching to
   clear a jam does not trigger it. An alarm that fires whenever somebody kneels
   gets switched off.
8. **Machine-running inference may not transfer.** Some camera angles cannot
   separate running from stopped by motion energy at all, and `suggest_thresholds`
   says so rather than emitting a threshold that is really a coin flip.
9. **A known guard false positive**: a person lying across the guard region can sit
   just under the occlusion threshold and produce a guard alert alongside the
   correct critical. Conservative, but noise at the worst moment.
10. **State is in memory.** A restart loses in-flight jobs and their
    acknowledgements. The baked runs survive because they are files.
11. **Keepout does not check PPE, deliberately.** OSHA's 2024 PPE rulemaking says
    the dominant real-world failure is PPE that is present and ill-fitting, which
    looks correct on camera and does not protect. A camera flags absence, not
    effectiveness, and selling one as the other would be dishonest.

12. **Small people are missed.** Below about 15% of frame height one detector pass
    is unreliable. Tiled detection reaches about 8% and is five times slower. The
    run warns when a probe finds people the single pass missed.
13. **A cut to a similar-looking camera can be partly missed.** ECC can converge on
    a small false shift against an unrelated view. On the one real example, 165 of
    1,085 frames after the cut were judged usable. A fix based on the ECC
    correlation is measured but not built (evaluation.md §10.4).
14. **Incident de-duplication inherits tracker mistakes.** A track id that the
    tracker hands from one person to the person beside them continues one
    incident. On the courtyard clip, one entry was folded into a neighbour's
    incident this way.
15. **Blur is generous, and in a crowd it is heavy.** Low-threshold boxes around
    groups blur large patches of a crowded evidence frame. That is the direction we
    chose to fail in, and it costs some of the evidence's detail.

---

## 9. Responsible use

**What this must not become.** A camera pointed at workers is surveillance unless
it is built not to be. Keepout counts events and measures geometry. It does not
identify, does not rank individuals, does not measure productivity, and does not
retain anything that could be joined to a person. That is enforced by what the
code does not contain, not by a policy document.

**The evidence is of a hazard, not of a person.** Every stored frame has the head
region of every detected person blurred before it is written. An incident record names a zone, a machine state, a
duration and a track number that means nothing tomorrow.

**It must not be used for discipline.** An incident log that becomes a
disciplinary record stops being a safety tool within a week, because the people it
is meant to protect will start working around it, and a system that is worked
around is worse than no system because it creates the appearance of cover. If a
site wants to deploy this, the deployment agreement should say so explicitly.

**Alerting is not guarding.** A detection is a request for a human to look, not a
substitute for a fixed guard, an interlock or an emergency stop. The Knowsley case
turned on the absence of a guard *and* the absence of an emergency stop; a camera
would have summoned help faster, and the guard would have prevented the injury.
Keepout is the last line, and the last line is the worst place to put your only
control.

**Say when it cannot see.** The four honesty rails exist because a safety system
that goes quiet when it goes blind is more dangerous than no system at all. That
is why the refusal state is a designed screen rather than a degraded one, and why
"incidents raised while blind" is a first-class number in the evaluation.

**Consent and notice.** Workers should be told the camera is there, what it
computes, what is stored, for how long, and that no identity is derived. The
30-day expiry on stored evidence is a privacy decision, not a billing one.
