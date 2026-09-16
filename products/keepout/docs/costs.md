# What Keepout created on AWS, and what it costs

Account <aws-account-id>. Everything tagged `Project=opencv26`, `Product=keepout`.
Prices below are AWS list prices for the stated region, checked 16 September 2026.

**Live endpoint: <https://mqrmfp6mbi.eu-west-2.awsapprunner.com>**

---

## The inventory

| # | Resource | Region | Identifier | Created by |
|---|---|---|---|---|
| 1 | App Runner service | eu-west-2 | `opencv26-keepout` | `infra/apprunner.sh` |
| 2 | ECR repository | eu-west-2 | `opencv26/keepout` | `infra/ecr.sh` |
| 3 | ECR repository | us-east-1 | `opencv26/keepout` | `infra/ecr.sh` (first attempt) |
| 4 | IAM role | global | `opencv26-keepout-instance` | `products/keepout/infra/instance-role.sh` |
| 5 | IAM inline policy | global | `opencv26-keepout-evidence-write` | same |
| 6 | S3 prefix | us-east-1 | `opencv26-artifacts-<aws-account-id>/keepout/evidence/` | shared bucket, `infra/s3.sh` |
| 7 | CloudWatch log groups | eu-west-2 | `/aws/apprunner/opencv26-keepout/*` | App Runner, automatically |

Shared with the other four entries, not created by Keepout: the S3 bucket itself
and the `opencv26-apprunner-ecr-access` role.

---

## Running cost

### App Runner, the only thing that costs real money


`opencv26-keepout` runs at **4 vCPU / 8 GB**, always on, created 04:29 UTC on
16 September 2026.

App Runner bills two things separately. **Provisioned memory** is charged for
every hour the service exists, whether or not anybody is using it. **Active CPU**
is charged only while a request is actually being served.

eu-west-2 (London) list prices:

| Component | Rate | Hours/month | Monthly |
|---|---|---|---|
| Provisioned memory, 8 GB | $0.00778 per GB-hour | 730 | **$45.43** |
| Active CPU, 4 vCPU | $0.07112 per vCPU-hour | see below | variable |

Idle, the service costs **$45.43/month**, or about **$1.49/day**.

Active CPU is the part that depends on judging. Serving the baked demo is a file
read of a few tens of milliseconds, so browsing the UI is effectively free. A
*live* re-run is the expensive path: a 30 second clip takes about 160 seconds of
wall time on this instance, which is 0.178 vCPU-hours at 4 vCPU, or **$0.0127 per
live run**. A judge who runs every one of the seven bundled clips live costs about
nine cents.

Budgeting 200 live runs across the judging window adds **$2.54**. Round the whole
service to **$48/month**.

> If this had fitted in `us-east-1` it would have been slightly cheaper:
> $0.00694/GB-hour and $0.064/vCPU-hour, so $40.53/month idle. The extra $4.90 is
> the price of the region App Runner had capacity in. See "the region" below.

### ECR

| Repository | Stored | Rate | Monthly |
|---|---|---|---|
| eu-west-2 | 0.762 GB | $0.10/GB-month | **$0.08** |
| us-east-1 | 0.381 GB | $0.10/GB-month | **$0.04** |

Both repositories carry a lifecycle policy keeping the last ten images. The
us-east-1 repository holds one image from the first deployment attempt and is dead
weight; deleting it saves four cents a month and is left in place only so the
deployment history in this document matches what is actually in the account.

Data transfer out of ECR to App Runner in the same region is free. The first pull
crossed no region boundary.

### S3

Three evidence frames per live run at roughly 62 KB each. A hundred live runs is
about 19 MB.

| Component | Rate | Monthly |
|---|---|---|
| Storage, ~20 MB | $0.023/GB-month | **$0.0005** |
| PUT requests, ~300 | $0.005 per 1,000 | **$0.0015** |
| GET requests | $0.0004 per 1,000 | negligible |

**Under one cent per month.** The 30-day expiry rule on the bucket is there for
tidiness rather than thrift: evidence frames are images of people at work and they
should not accumulate indefinitely, which is a privacy argument, not a billing one.

### CloudWatch Logs

Structured JSON, one object per line, at roughly 40 lines per analysis. Ingest is
$0.50/GB and the first 5 GB of storage is free. Realistically **under $0.05/month**.

### IAM

Free.

---

## Total

| | Monthly |
|---|---|
| App Runner, idle | $45.43 |
| App Runner, ~200 live runs | $2.54 |
| ECR, both regions | $0.12 |
| S3 | $0.01 |
| CloudWatch Logs | $0.05 |
| **Total** | **≈ $48.15/month** |

About **$1.58 a day**, or **$0.066 an hour** for a service that is always on and
answers in under two seconds from cold.

The original plan across five products was $80/month total, which this single
service would have broken on its own. The instruction changed partway through to
size for a good judge experience rather than the cheapest option, and this is that
decision made visible: 4 vCPU / 8 GB always on, not 0.25 vCPU / 0.5 GB scaled to
zero. At 0.25 vCPU / 0.5 GB the same service would idle at **$2.53/month**, and a
live run of a 30 second clip would take roughly forty minutes instead of two and a
half, which is not a demo.

### Turning it off

```bash
AWS_REGION=eu-west-2 infra/apprunner.sh keepout --delete
```

Provisioned memory stops billing the moment the service is deleted. The image in
ECR (8 cents a month) and the evidence in S3 (under a cent) can stay.

---

## The region, and why it is not us-east-1

The brief specified `us-east-1`. The service is in `eu-west-2`. This is not a
preference; it is an account restriction, and the error message is worth quoting
because it is not a quota you can see in the console:

```
An error occurred (InvalidRequestException) when calling the CreateService
operation: Account <aws-account-id> is restricted and can support only two App
Runner services per region at the moment. To remove the restriction, contact
the AWS support team.
```

Both `us-east-1` slots were already held by unrelated services that are live and
in use (`ugjcs-backend` and `cairn-v2`), so deleting one to make room was not an
option I was willing to take unattended. `us-east-2` and `us-west-2` were also
full. `eu-west-2` had capacity.

Three consequences, all handled:

1. **The ECR repository had to be duplicated into eu-west-2.** App Runner cannot
   pull an image from another region. The us-east-1 repository still exists with
   the first image in it.
2. **The S3 bucket stayed in us-east-1**, because it is shared with the other four
   entries. The container is given `AWS_REGION=us-east-1` so its S3 client targets
   the right endpoint; the App Runner service itself is in eu-west-2. Cross-region
   PUTs of 62 KB images cost $0.02/GB in transfer, which at the volumes here is
   fractions of a cent.
3. **Everything is still tagged `Project=opencv26`**, so Cost Explorer can split
   the bill by product regardless of region.

There is an accidental appropriateness to it: the case this product is named for
is a UK Health and Safety Executive prosecution, and the service now runs in
London. That is a coincidence, not a design decision.

---

## Things that were created and then not used

**Nothing.** Both ECR repositories hold real pushed images and the us-east-1 one is
recorded above rather than quietly deleted. No EC2 instance, no load balancer, no
NAT gateway, no Graviton instance, no COOL Marketplace subscription. Keepout does
not use the COOL AMI: its workload is dominated by ONNX inference in `cv2.dnn`,
not by the `imgproc` functions COOL accelerates, so a COOL benchmark here would be
measuring the 3% of the frame budget that is not the bottleneck.

---

## Deployment log

| When (UTC) | What |
|---|---|
| 2026-09-16 04:15 | Image pushed to ECR us-east-1, tag `9a37fa8-dirty` |
| 2026-09-16 04:17 | App Runner create refused: two-service-per-region restriction |
| 2026-09-16 04:24 | Image pushed to ECR eu-west-2, tag `af577ed-dirty` |
| 2026-09-16 04:29 | App Runner `opencv26-keepout` created, 2 vCPU / 4 GB, healthy |
| 2026-09-16 04:40 | First live run: 605 ms/frame, 3 S3 uploads failed (no instance role) |
| 2026-09-16 04:45 | IAM instance role created and attached |
| 2026-09-16 04:50 | Image rebuilt with a fail-fast, circuit-broken S3 sink |
| 2026-09-16 04:55 | Service updated to 4 vCPU / 8 GB with the instance role |
| 2026-09-16 04:56 | Live run: 3 evidence frames mirrored to S3, 0 failures |
| 2026-09-16 06:05 | Build failed: the Dockerfile still baked one clip, and the evidence path had moved to `data/demo/<clip>/evidence` |
| 2026-09-16 06:11 | Image rebuilt with all seven clips baked, tag `f9f79f5-dirty` |
| 2026-09-16 06:20 | Service updated to that image, `KEEPOUT_INSTANCE_LABEL` set so the UI stops claiming 2 vCPU |
| 2026-09-16 06:24 | Verified live: all four view-refusal states reachable, `cell-down` opens on the latched critical |
