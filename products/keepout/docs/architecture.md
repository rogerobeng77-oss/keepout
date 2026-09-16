# Architecture

Two diagrams: what happens to a frame, and what runs on AWS.

---

## 1. The vision pipeline

Every frame goes through the same seven steps, in this order, and the order is the
design. The view check is first because nothing below it means anything if the
camera has moved.

```mermaid
flowchart TD
    A[Frame from a fixed camera] --> B{View check<br/>viewcheck.py}

    B -->|camera moved<br/>lens blocked<br/>too dark<br/>frozen feed| X[Stop.<br/>Report the problem,<br/>raise nothing,<br/>ask for the zone to be re-drawn]
    B -->|usable| C[Person detection<br/>YOLOX-tiny in cv2.dnn<br/>COCO person class only]

    C --> D[Association into tracks<br/>track.py: Hungarian assignment<br/>+ constant-velocity prediction]

    D --> E[Machine state<br/>machine.py: motion energy in the<br/>machine polygon, person boxes excluded]
    D --> F[Guard state<br/>guard.py: CLAHE edge + appearance<br/>correlation vs a learned reference]
    D --> G[Posture<br/>posture.py: box aspect ratio<br/>+ stillness over a window]
    D --> H[Zone occupancy<br/>zones.py: signed distance from the<br/>polygon to the feet, with hysteresis]

    E --> I{Escalate<br/>escalate.py}
    F --> I
    G --> I
    H --> I
    D -->|a confirmed track vanished<br/>deep inside the zone| I

    I -->|in the zone, machine stopped| N[NOTE<br/>closes when they leave]
    I -->|guard no longer matches| GU[GUARD OPEN<br/>closes when it comes back]
    I -->|in the zone, machine running| AL[ALERT<br/>closes when they leave]
    I -->|down and motionless,<br/>or unaccounted for| CR[CRITICAL<br/>latched until a human acknowledges]

    N --> EV[Evidence frame<br/>privacy.py blurs faces<br/>before anything is written]
    GU --> EV
    AL --> EV
    CR --> EV

    EV --> OUT[Incident log + RunRecord JSON]
    X --> OUT

    style X fill:#2A2F36,stroke:#8A9198,color:#F2F4F5
    style CR fill:#FF6B35,stroke:#FF6B35,color:#17191C
    style AL fill:#FF6B35,stroke:#FF6B35,color:#17191C
    style EV fill:#20242A,stroke:#6FD08C,color:#F2F4F5
```

### Why the order matters

**The view check is first**, and it is the only step that can stop the pipeline.
A danger zone is a polygon in the pixel coordinates of one reference frame. Nudge
the camera and that polygon is a statement about a different piece of floor, but
the zone monitor carries on happily reporting "nobody in the zone". Silence that
looks like safety is the worst output this system has, so a frame that fails the
view check produces a named problem and no incidents at all.

We deliberately do not warp the zone into the new view. Fitting a homography to a
scene that has partly changed is exactly the silently-confident-nonsense failure
mode we are trying to avoid. A human re-draws it.

**Detection feeds everything else**, including the two checks that need to know
where people are in order to ignore them:

- machine motion energy excludes the person boxes, because otherwise a worker
  walking past a stopped machine makes it read as running, precisely when somebody
  is next to it;
- the guard check excludes them too, because otherwise anybody standing in front
  of a guard reads as a removed guard, and that alert gets switched off in a week.

**The vanish edge into escalation** is the one that is easy to miss. It is there
because YOLOX-tiny stops returning a box past about 50 degrees of body rotation,
so a person on the floor often produces no detection at all and posture can never
fire. A confirmed track that was standing well inside the zone and then stops
being detected, without crossing the boundary, is escalated on its own. See
`docs/evaluation.md` §5.

---

## 2. The escalation state machine

```mermaid
stateDiagram-v2
    [*] --> Clear

    Clear --> Note: person in the zone,<br/>machine stopped
    Clear --> Alert: person in the zone,<br/>machine running
    Clear --> GuardOpen: guard stops matching<br/>its reference

    Note --> Alert: the machine starts<br/>(upgrades in place,<br/>same incident)
    Note --> Clear: they leave

    Alert --> Clear: they leave
    GuardOpen --> Clear: the guard comes back

    Note --> Critical: down and motionless,<br/>or unaccounted for
    Alert --> Critical: down and motionless,<br/>or unaccounted for
    Clear --> Critical: unaccounted for

    Critical --> Acknowledged: a human presses<br/>acknowledge
    Acknowledged --> [*]

    note right of Critical
        Latched. The condition
        ending does not clear it.
        Only a person does, and
        the acknowledgement is
        recorded with its time.
    end note

    note right of Note
        Incidents upgrade in place
        and never downgrade, so one
        continuous presence is one
        story with one set of
        evidence frames.
    end note
```

---

## 3. AWS

```mermaid
flowchart LR
    subgraph Build["Build time, on a workstation"]
        SRC[Source + packages] --> DOCK[docker build<br/>linux/amd64]
        DOCK --> M1[Fetch YOLOX-tiny ONNX<br/>Apache-2.0, digest printed]
        M1 --> M2[Assert OpenCV major == 5]
        M2 --> M3[keepout demo --bake<br/>analyse all 7 clips,<br/>write results + evidence<br/>into the image]
        M3 --> IMG[(Image, 682 MB)]
    end

    IMG -->|infra/ecr.sh| ECR[(Amazon ECR<br/>opencv26/keepout<br/>lifecycle: keep last 10)]

    ECR -->|pull, via the<br/>ECR access role| AR

    subgraph Runtime["eu-west-2"]
        AR[AWS App Runner<br/>opencv26-keepout<br/>4 vCPU / 8 GB<br/>managed TLS, stable URL]
    end

    AR -->|assumes| IR[IAM instance role<br/>opencv26-keepout-instance<br/>s3:PutObject on one prefix only]
    IR --> S3[(Amazon S3<br/>opencv26-artifacts-<aws-account-id><br/>keepout/evidence/*<br/>private, versioned,<br/>30-day expiry)]

    AR --> LOGS[(CloudWatch Logs<br/>one JSON object per line)]

    J((Judge)) -->|HTTPS| AR
    AR -->|"/api/demo: baked run,<br/>served from disk in under 2 s"| J
    AR -->|"POST /api/jobs: the same<br/>pipeline, live on the instance"| J

    style AR fill:#F5C518,stroke:#F5C518,color:#17191C
    style S3 fill:#20242A,stroke:#6FD08C,color:#F2F4F5
    style IMG fill:#20242A,stroke:#A7AEB4,color:#F2F4F5
```

### What each piece is for

| Component | Why |
|---|---|
| **ECR** | Private image registry in the same region as the service. App Runner cannot pull cross-region. Lifecycle policy keeps the last ten images. |
| **App Runner** | A managed HTTPS URL with no load balancer. An ALB would cost about $16.20/month at zero traffic; App Runner's URL is included. There is no arm64 runtime, which is why the image is `linux/amd64`. |
| **IAM instance role** | Scoped to `s3:PutObject` on `keepout/evidence/*` and nothing else. The service never reads, lists or deletes. Created by `products/keepout/infra/instance-role.sh` because the shared `infra/apprunner.sh` only creates the ECR *access* role. |
| **S3** | Evidence frames that outlive the container. Private, versioned, encrypted, 30-day expiry. Failures are non-fatal and circuit-broken: evidence is already on local disk and served by the API. |
| **CloudWatch Logs** | App Runner ingests stdout. `servicekit` writes one JSON object per line with a request id, so there is no logging agent and no log files. |

### State, and the lack of it

The service holds jobs in memory, with a TTL and an eviction cap. That is a
deliberate limit, not an oversight: a restart loses in-flight jobs and their
acknowledgements. The baked runs are read-only files in the image, so they always
come back. A real deployment that needed acknowledgements to survive a restart
would write incidents to DynamoDB, and the shape of the `RunRecord` is already the
right shape for that.

### The region

The service is in **eu-west-2**, not `us-east-1` as originally planned, because
this AWS account is restricted to two App Runner services per region and both
`us-east-1` slots hold unrelated live services. The ECR repository was created in
`eu-west-2` to match, since App Runner cannot pull across regions. The S3 bucket
stays in `us-east-1` and the container is given `AWS_REGION=us-east-1` so its S3
client targets the right endpoint. Everything is tagged `Project=opencv26`.
