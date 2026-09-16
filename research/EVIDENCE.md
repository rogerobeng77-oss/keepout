# EVIDENCE — OpenCV AI Competition 2026

Hard, quotable evidence behind candidate problems. Every figure below was fetched
from a primary source during this session and is quoted verbatim. Fetched pages
are saved under `/tmp/ocv-evidence/`.

Compiled 2026-09-16. Companion document: `FINDINGS.md` (technical ground and
problem briefs) — written separately, not by this pass.

**Rule applied throughout:** if a number is not in a quote below, it is not
verified and must not be used in a video or a submission. Anything remembered but
not fetched is marked UNVERIFIED and should be treated as false until checked.

---

## 1. Summary table

| # | Problem | Human cost per year (verified) | Biggest verified money figure | Evidence |
|---|---|---|---|---|
| 1 | Intraoperative / obstetric blood loss underestimated, transfusion delayed | ~260,000 maternal deaths globally (2023, WHO); visual estimation underestimated blood loss in 90% of caesarean cases in one cohort | No named settlement fetched this pass | **Strong** on the failure mode, **moderate** on money |
| 2 | Bile duct injury in laparoscopic cholecystectomy / critical view of safety | 1.12% BDI rate across 41,044 Chinese patients (2022 meta-analysis); 0.4% in a 76,524-case paediatric review | No named verdict fetched this pass | **Strong** on incidence and failure mode |
| 3 | Retained surgical items | ~1 in 32,672 paediatric cases (national US study); other estimates 1/1,000–1/19,000 procedures | "medical and liability costs of >$200,000 per incident" | **Strong** on cost per incident, **moderate** on rate (estimates conflict by 20x) |
| 4 | Unsafe surgery generally (framing for 1–3) | "complications after inpatient operations occur in up to 25% of patients"; 3M+ deaths/yr from unsafe care (WHO) | Patient harm "reduces global economic growth by 0.7% a year" (WHO) | **Strong** |
| 5 | Powerline-ignited wildfire (utilities + disaster response) | 84 counts of involuntary manslaughter charged, Camp Fire 2018 (the often-cited "85 deaths" is NOT verified) | **$13.5 billion** Fire Victim Trust; guilty plea to **84 counts of involuntary manslaughter** | **Strong** — best money evidence in the whole set |
| 6 | Disaster response delay / damage assessment | Not verified | Not verified | **Thin** — nothing citable found; see §3.2–3.3 |
| 7 | Road, level crossings, work zones | 1.16M road deaths/yr globally (WHO); 857 US work-zone crash deaths 2020, 62 of 117 worker deaths struck-by-vehicle on foot | $1.91M FRA grant for crossing cameras; DOT VSL $12.5M/life (2022 USD, since withdrawn) | **Strong** on stats, **thin** on money |
| 8 | Workplace and industrial safety | 5,283 US fatal work injuries, 2023 (OSHA) | **GBP 650,000** single HSE fine (T Vaughan Ltd, 2026); $349,754 top US OSHA penalty fetched | **Strong** — most named, dated, dollar-verified cases of any domain |
| 9 | Water safety and drowning | ~300,000 drowning deaths/yr globally (WHO); ~4,000/yr US | No drowning verdict or settlement amount verified; VGBA is federal law | **Strong** on scale and on lifeguard failure mode; **thin** on money |
| 10 | Elderly and hospital inpatient falls | 684,000 fall deaths/yr globally + 37.3M falls needing medical care (WHO) | **$50.0 billion** US medical cost of older-adult falls, 2015 (Florence et al.); **$456.8M** in CMS nursing-home fines over 3 years | **Strong** |

All ten rows are now evidenced. Sections 2 and 3 cover the two chosen domains;
sections 4 to 7 cover road/rail, workplace, water and falls. Agriculture,
environment and retail/logistics were not researched — see §11.

---

## 2. Domain 1 — Surgical safety in the operating field

### 2.1 Blood loss is estimated by eye, and the eye is wrong

**Source:** Edilu R, et al. "Comparing visual estimation and hematocrit change in
the assessment of blood loss among women undergoing cesarean delivery in a
tertiary facility in northern Uganda." *Therapeutic Advances in Reproductive
Health*, 17 October 2024. PMID 39435121, PMCID PMC11492183.
DOI 10.1177/26334941241289552.

> "The visual estimation technique significantly underestimated blood loss in up
> to 90% of cases, particularly during emergency cesarean sections."

and, from the results:

> "Visual estimation underestimated blood loss in 90% of cases (n = 100), and 21%
> (n = 21) had undiagnosed PPH (>1000 ml blood loss). None of the respondents had
> PPH (>1000 ml blood loss) following vEBL."

That last sentence is the whole product case in one line: by eye, nobody
haemorrhaged; by measurement, one in five did.

**Corroborating source, different country, different design:** Tan AWM, et al.
"Accurate estimation of blood loss during cesarean deliveries: A secondary
analysis of a randomized controlled trial comparing visual, quantitative and
calculated approaches." *Acta Obstetricia et Gynecologica Scandinavica*,
November 2025 (Epub 25 Sep 2025). PMID 40999760. DOI 10.1111/aogs.70052.
KK Women's and Children's Hospital, Singapore — the country's largest maternity unit.

> "On average, vEBL was 249.7 mL (95% CI: -822.7-323.3) less than QBL, and 287.9
> mL (95% CI: -1143.9-568.0) less than cEBL."

> "vEBL appears to grossly underestimate actual blood loss when compared with QBL
> and cEBL methods... reliance solely on vEBL may lead to under-recognition and
> delayed management of PPH."

Note honestly: this second study says the differences were **not statistically
significant** and the confidence intervals are wide. Quote the direction, not a
precise millilitre figure, from this one.

**Third, independent confirmation that the profession already knows:**
Biller-Friedmann K, Bayerlein J. "[Visual estimation of blood losses: Known high
error rate — How can it be improved?]" *Die Anaesthesiologie* 74(6):384–394,
June 2025. DOI 10.1007/s00101-025-01517-6. The title itself is the finding: the
high error rate is "known".

**Scale of the stakes.** WHO, "Maternal mortality" fact sheet, 2023 data,
https://www.who.int/news-room/fact-sheets/detail/maternal-mortality

> "Every day in 2023, over 700 women died from preventable causes related to
> pregnancy and childbirth. A maternal death occurred almost every 2 minutes in
> 2023."

> "About 260 000 women died during and following pregnancy and childbirth in
> 2023."

> "Approximately 92% of all maternal deaths occurred in low- and lower-middle-income
> countries in 2023, and most could have been prevented."

**Failure mode.** Blood loss is assessed by a clinician glancing at swabs, drapes
and suction canisters during a procedure where their attention is on the
operation. The measured alternatives exist — gravimetric weighing of gauze
(QBL) and haematocrit-based calculation (cEBL) — and the Singapore paper says
plainly that they "remain underutilized" in the Ugandan setting and are
recommended but not routine in Singapore. The failure is not ignorance; it is
that measurement costs time and attention that the operating team does not have.

**Where a camera helps.** A camera over the field, or on the suction/swab
station, can estimate blood volume on gauze and in canisters continuously and
without asking anyone to stop and weigh anything. This is the closest thing in
the whole evidence set to a problem where vision replaces a task humans are
measurably bad at, with a ground truth (gravimetric weight) that can be used to
train and validate.

**What a camera cannot do.** It cannot see blood that is inside the patient
(concealed abruption, intra-abdominal bleeding), it cannot distinguish blood from
amniotic or irrigation fluid without help — the Singapore paper flags exactly
this contamination problem — and it does not decide to transfuse. It shortens
recognition, not treatment.

### 2.2 Bile duct injury and the critical view of safety

**Incidence.** Yang S, Hu S, Gu X, Zhang X. "Analysis of risk factors for bile
duct injury in laparoscopic cholecystectomy in China: A systematic review and
meta-analysis." *Medicine (Baltimore)* 101(37):e30365, 16 September 2022.
PMID 36123939, PMCID PMC9478294. DOI 10.1097/MD.0000000000030365.

> "The compilation of all data from a total of 19 case-control studies revealed
> that among 41,044 patients, 458 patients experienced bile duct injury in LC,
> accounting for the incidence rate of 1.12% for bile duct injury."

The same paper identifies the anatomical driver, which matters because it is
exactly what a camera looks at:

> "the anatomic variations of the gallbladder triangle (OR = 11.82, 95% CI:
> 6.32-22.09, P < .001)"

An eleven-fold odds ratio attached to *how the anatomy looks in the triangle* is
a strong argument that the decisive information is visual and is in the
laparoscope's field of view.

**A lower, more conservative rate** — use this to avoid overclaiming: Mattson A,
et al. "Laparoscopic cholecystectomy in children: A systematic review and
meta-analysis." *The Surgeon* 21(3):e133–e141, June 2023. PMID 36243605.
76,524 LC cases from 114 studies.

> "Major complications included bile duct injury (0.4%) and intra- or
> post-operative bleeding (0.9%)."

Be honest in the submission that published BDI rates range from roughly 0.4% to
1.12% depending on population and study design. Do not pick the big one and
present it as settled.

**The failure mode, named by surgeons.** Aguilera M, et al. "Critical view of
safety in laparoscopic cholecystectomy: a nationwide video-based benchmark of
resident-performed cases." *Surgical Endoscopy*, 3 August 2026 (online ahead of
print). DOI 10.1007/s00464-026-13243-0. Pontificia Universidad Católica de Chile.
Nationwide review of 548 operative videos.

> "CVS status was available for 513/548 recordings (93.6%), and CVS was achieved
> in 314/513 cases (61.2%)."

> "Achievement declined with complexity: OPRS 1, 66.5%; OPRS 2, 62.3%; OPRS 3,
> 56.3%; OPRS 4, 42.9%; and OPRS 5, 14.3% (p < 0.001)."

That gradient is the finding worth building on. The safety step that prevents the
injury is skipped most often precisely in the hardest cases, where it is needed
most — in the hardest tier, it is achieved in one case in seven.

On why CVS matters at all, from the Tokyo Guidelines commentary in the same
result set:

> "Achieving a CVS prevents the misidentification of the [structures before]
> dividing any structures."

and the honest limit, also from that source:

> "a critical view of safety (CVS) cannot be achieved because of the presence of
> nondissectable scarring or severe [inflammation]"

**Where a camera helps.** The laparoscope is already a camera, already recording
in many theatres, and CVS is defined by three visual criteria that a model can be
trained to score. A real-time "CVS not yet achieved" indicator is a well-posed
vision problem with an existing expert-rating ground truth, and the Chilean paper
shows asynchronous video review already works as a benchmark — so the labels
exist.

**What a camera cannot do.** It cannot achieve the view; only dissection does.
In the severe-inflammation cases where CVS is impossible, the correct action is a
bail-out (subtotal cholecystectomy or conversion), and a model that only says
"not achieved" adds nothing there unless it also recognises the bail-out
indication. It also cannot prevent injuries that occur before the model's
decision point.

### 2.3 Retained surgical items

**Cost per incident** — the strongest single money line in the surgical set.
Regenbogen SE, et al. (decision-analytic model of sponge-tracking technologies;
retrieved via PubMed, in the `retained surgical item` result set,
`/tmp/ocv-evidence/rsi_abs.txt`):

> "Given medical and liability costs of >$200,000 per incident, novel
> technologies can substantially reduce the incidence of RSS at an acceptable
> cost."

The same model gives the economics of the incumbent fixes, which is useful
because it prices the competition:

> "for an additional $95,000 per RSS averted. If RF were as effective as bar
> coding, it would cost $720,000 per additional RSS averted (versus standard
> [counting])... $1.1 to 1.4 million per RSS event prevented."

A camera-based count that costs nothing per case competes against alternatives
that a peer-reviewed model prices at $95,000 to $1.4 million per event averted.

**Incidence — and a genuine conflict in the literature, stated honestly.**
From a US national paediatric study in the same result set:

> "retained foreign bodies after surgery (incidence, 0.0031%)... The incidence of
> retained foreign body is 0.0031% or approximately 1 in 32,672 cases and is
> associated with an increased charge of $42,077 for this complication."

> "Mean [additional] charges for a patient with this complication are $56,683
> (95% confidence interval, $41,327-$72,039); mean length of stay is 10.5 days."

But a UK Health Foundation / socio-technical review in the same set says:

> "it has been estimated that incidences range between 1/1000 and 1/19,000
> procedures"

These differ by more than a factor of twenty. And the honest reason is given
directly:

> "Gossypiboma is under reported and the true incidence is largely unknown."

**Use this carefully.** The per-incident cost ($200,000+ liability, $42,077
additional charges, 10.5 extra bed-days) is solid. The rate is not. Lead with
cost per incident and the "never event" framing, not with a frequency.

**Where a camera helps.** Counting sponges in and out of a cavity is a visual,
countable task performed under time pressure by people also doing other things.
The literature explicitly blames "count recording" as a risk factor, and names
"emergency surgery, an unplanned [change in procedure]" as the conditions where
counts fail.

**What a camera cannot do.** It cannot see an item already inside the patient and
out of the optical path — which is the whole problem in an open abdomen packed
with laparotomy pads. RF-tagged sponges can; a camera cannot. Be upfront that a
camera system is a complement to, not a replacement for, RF detection.

### 2.4 The framing statistic for any surgical product

WHO, "Patient safety" fact sheet,
https://www.who.int/news-room/fact-sheets/detail/patient-safety

> "Around 1 in every 10 patients is harmed in health care and more than 3 million
> deaths occur annually due to unsafe care."

> "Above 50% of harm (1 in every 20 patients) is preventable"

> "Patient harm potentially reduces global economic growth by 0.7% a year. On a
> global scale, the indirect cost of harm amounts to trillions of US dollars each
> year."

WHO, "Safe surgery",
https://www.who.int/teams/integrated-health-services/patient-safety/research/safe-surgery

> "the reported crude mortality rate after major surgery is 0.5-5%; complications
> after inpatient operations occur in up to 25% of patients; in industralized
> countries, nearly half of all adverse events in hospitalized patients are
> related to surgical care; at least half of the cases in which surgery led to
> harm are considered preventable"

> "The Surgical Safety Checklist has been shown to reduce complications and
> mortality by over 30 percent. The Checklist is simple and can be completed in
> under 2 minutes"

That last quote is a gift for a pitch: the benchmark intervention in surgical
safety is a two-minute paper checklist, and it moved mortality by over 30%. It
sets a low, credible bar for what a software intervention has to beat, and it is
WHO's own number.

### 2.5 Wrong-site surgery and surgical fires — NOT VERIFIED

Joint Commission sentinel event data (jointcommission.org) returned **HTTP 403**
and the FDA surgical fire safety communication returned **HTTP 503** during this
session. No figure for wrong-site surgery frequency or surgical fire frequency
was obtained. **Do not use any number for these two sub-problems.** They are
plausible but currently unevidenced here.

---

## 3. Domain 2 — Natural disaster response

### 3.1 The one overwhelming verified case: Camp Fire / PG&E

This is the best-documented money-and-lives fact in the entire evidence set, and
it sits at the intersection of disaster response (§2) and utility vegetation
management (§9) — so it can anchor either product.

**Source:** PG&E Corporation and Pacific Gas and Electric Company, Form 10-K for
fiscal year ended 31 December 2020, filed with the U.S. Securities and Exchange
Commission 25 February 2021. Accession 0001004980-21-000007, document
`pcg-20201231.htm`. This is the company's own sworn filing, not journalism.

> "On March 17, 2020, the Utility entered into the Plea Agreement and Settlement
> (the "Plea Agreement") with the People of the State of California, by and
> through the Butte County District Attorney's office... to resolve the criminal
> prosecution of the Utility in connection with the 2018 Camp fire. Subject to
> the terms and conditions of the Plea Agreement, the Utility agreed to plead
> guilty to 84 counts of involuntary manslaughter in violation of Penal Code
> section 192(b) and one count of unlawfully causing a fire in violation of Penal
> Code section 452."

> "Per the Plea Agreement, the Utility was sentenced to pay the maximum total
> fine and penalty of approximately $3.5 million. The Utility also agreed to pay
> $500,000 to the Butte County District [Attorney's office]"

And the civil side, from the same filing:

> "$12.15 billion of the $13.5 billion liability as of June 30, 2020 was
> extinguished in the third quarter of 2020, and the remaining $1.35 billion will
> be paid out under the terms of the Tax Benefits Payment Agreement... On January
> 15, 2021, the Utility paid approximately $758 million of the $1.35 billion"

**The detail that makes it a vision problem.** From PG&E's own Q2 2020 press
release (SEC Exhibit 99.1, `pge-063020xpressrelease.htm`, filed 30 July 2020),
describing what the company did *after* the fire:

> "Situational awareness completion exceeds 30 percent, with 144 weather stations
> and 60 high definition cameras installed, despite some initial supply chain
> issues due to COVID-19 disruptions."

> "Enhanced vegetation management progress is at 70 percent. PG&E has reviewed
> more than 1,200 miles of distribution and lower-voltage transmission lines and
> taken necessary action to trim or remove hazards and expand rights-of-way."

So the remedy the utility itself adopted, under a criminal plea and a $13.5bn
settlement, was *cameras and vegetation inspection*. That is not our inference —
it is in the filing. It is the single most defensible "a camera would have
helped" claim available, because the defendant said it.

**Failure mode.** Vegetation and ageing hardware in a high-fire-threat corridor
were inspected on a cycle, by people, at a scale of thousands of circuit miles.
The filing quantifies the scale honestly: over 1,200 miles reviewed in a single
partial-year push. Human inspection does not cover that continuously.

**What a camera cannot do.** It cannot stop a worn C-hook from failing, it cannot
de-energise a line by itself, and 60 cameras across PG&E's service territory is
sparse detection, not coverage. A detection product shortens time-to-notice; it
does not remove the ignition source.

**Note on precision:** the 10-K says 84 counts of involuntary manslaughter. The
commonly cited death toll for the Camp Fire is 85 (one death was not charged as
manslaughter). **Quote "84 counts of involuntary manslaughter" — that is the
verified figure.** Do not say "85 deaths" unless you verify it separately;
cal fire's own incident page returned **HTTP 403 Access Denied** this session.

### 3.2 Earthquake entrapment and time-to-rescue — PARTIAL

From the PubMed entrapment/survival result set
(`/tmp/ocv-evidence/eq_abs.txt`), a burn-and-crush cohort:

> "Among the burn patients, 101 (52.9%) were rescued from the rubble 2-60 h after
> [the earthquake]"

and from a separate building-collapse case series (Zümrüt apartment, Konya,
2 February 2004):

> "92 out of a total of 121 [occupants were] rescued from the rubble"

with extrication times averaging in the range

> "(5-24 hours) on average"

**This is thin.** It establishes that extrication takes many hours, but I did not
verify a survival-versus-time curve, a "golden 72 hours" figure from an official
source, or any cost or inquiry finding. **Do not build a claim on this without
more work.**

### 3.3 Sources that failed this session — disaster domain

| Source | Result |
|---|---|
| cal fire incident page (fire.ca.gov) | HTTP 403 Access Denied |
| CPUC press release on the PG&E wildfire settlement | HTTP 404 (page moved) |
| Hawaii Attorney General Maui wildfire reports | HTTP 404 |
| WMO early warnings page | HTTP 404 |
| WHO floods fact sheet | fetched, but has no "Key facts" block to quote |
| GAO (gao.gov) | HTTP 403 across the site |

The disaster-response domain is **thin as fetched**, with the single enormous
exception of the PG&E case. If the team wants a disaster product, the honest
position tonight is: the wildfire-ignition framing has world-class evidence, and
the post-event damage-assessment framing does not yet.

---

## 4. Domain 3 — Road, rail level crossings and work zones

### 4.1 Global scale

**Source:** WHO, "Road traffic injuries" fact sheet, updated 20 July 2026.
https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries

> "Approximately 1.16 million people die each year as a result of road traffic
> crashes... More than half of all road traffic deaths are among vulnerable road
> users, including pedestrians, cyclists and motorcyclists... Road traffic crashes
> cost most countries 3% of their gross domestic product."

> "The risk of death for pedestrians hit by car fronts rises rapidly (4.5 times
> from 50 km/h to 65 km/h)."

That second quote cuts both ways and should be used honestly: above a certain
speed, earlier detection does not save the pedestrian. The limiting factor is
impact energy, not sensing.

### 4.2 Valhalla, New York grade crossing — a full NTSB investigation

**Source:** NTSB, *Highway-Railroad Grade Crossing Collision, Commerce Street,
Valhalla, New York, February 3, 2015*, Railroad Accident Report NTSB/RAR-17/01,
adopted 25 July 2017. https://www.ntsb.gov/investigations/AccidentReports/Reports/RAR1701.pdf

> "Five passengers died and nine passengers and the engineer were injured, all in
> the lead railcar. The driver of the vehicle also died."

Probable cause, verbatim:

> "the probable cause of the accident was the driver of the sport-utility vehicle,
> for undetermined reasons, moving the vehicle on to the tracks while the Commerce
> Street highway-railroad grade crossing warning system was activated, into the
> path of Metro-North Railroad train 659."

The camera finding, which is the reason this report matters to us:

> "The lead railcar (Metro-North 4333) did not have an outward-facing (track
> image) video recorder. Federal regulations do not require outward-facing video
> recorders. However, Metro-North was in the process of installing them, but had
> not yet equipped this locomotive."

And the money that followed:

> "the FRA awarded $1.91 million to the MTA to install cameras to record movements
> at 43 identified grade crossings within Metro-North territory in New York to
> investigate specific incidents and analyze grade crossing/traffic operations for
> targeted modifications to improve safety. (MTA 2016)"

**What a camera could not have done here.** The engineer already saw the SUV and
applied emergency braking. Nothing about earlier detection stops a train inside
85 feet. The NTSB frames the funded cameras as forensic and site-analysis tools,
not as a real-time prevention mechanism — quote them that way or the claim
collapses under scrutiny.

### 4.3 Chester, Pennsylvania — Amtrak strikes a work crew at 99 mph

**Source:** NTSB, *Amtrak Train Collision with Maintenance-of-Way Equipment,
Chester, Pennsylvania, April 3, 2016*, NTSB/RAR-17/02, adopted 14 November 2017.

> "Two roadway workers were killed, and 39 other people were injured. Amtrak
> estimated property damages to be $2.5 million."

Probable cause, verbatim:

> "the unprotected fouled track that was used to route a passenger train at maximum
> authorized speed; the absence of supplemental shunting devices, which Amtrak
> required but the foreman could not apply because he had none; and the inadequate
> transfer of job site responsibilities between foremen during the shift change"

**Honest reading:** this is a handoff and signal-redundancy failure, not a
visibility failure. A camera checking whether foul-time was transferred is a
compliance tool. It does not stop a train already committed at 99 mph. If we
pitch a rail work-zone product, this report is the argument *against* framing it
as collision prevention.

### 4.4 Work zone fatalities — the numbers, in a federal rulemaking

**Source:** FHWA, "Work Zone Safety and Mobility and Temporary Traffic Control
Devices," Federal Register, published 20 September 2023, doc 2023-19701.

> "In 2020 (the latest year for which data are available), the National Highway
> Traffic Safety Administration (NHTSA) reports that 857 individuals lost their
> lives in 774 fatal work zone crashes. In 2020, 117 workers at road construction
> sites experienced a fatal occupational injury, 62 of which involved a worker on
> foot being struck by a motor vehicle."

62 of 117 — over half of fatal work-zone worker injuries are a worker on foot
struck by a vehicle. That is precisely the geometry a proximity-detection camera
addresses, and it is stated by the regulator in its own rulemaking.

### 4.5 Value of a statistical life — use with a caveat

**Source:** CPSC, "Notice of Availability of Final Guidance for Estimating Value
per Statistical Life," Federal Register, 18 April 2024, doc 2024-08300, Table 2
("U.S. Federal Departments' VSLs [2022 dollars]"):

> "EPA DOT HHS $11.0 million $12.5 million $12.3 million"

DOT's VSL is **USD $12.5 million in 2022 dollars**. Two caveats that must travel
with the figure: it was fetched from a CPSC cross-agency table, not DOT's own
page (transportation.gov returned Access Denied), and a later Federal Register
notice (2026-02-24, doc 2026-03655) withdrew that guidance — the federal VSL
framework is currently unsettled.

### 4.6 Litigation — named and dated, but NOT dollar-verified

CourtListener's v4 search API identified these; its opinion and cluster detail
endpoints returned **401 Unauthorized** without an API key, and the court PDF
hosts were blocked, so **no verdict or settlement amount was verified**:

- *Texas Department of Transportation and Vulcan Materials Company v. Kristina
  Smith*, Tex. App. 6th Dist., decided 31 October 2024 — work-zone fatality.
- *JTL Group, Inc. d/b/a Knife River v. Tangney Gray-Dockham*, Wyo. Sup. Ct.,
  decided 7 June 2022 — work-zone fatality.
- *Watson v. BNSF Railway Company*, Okla. Sup. Ct., decided 15 October 2024.

**Do not cite a dollar figure for any of these.** They are leads, not evidence.

---

## 5. Domain 4 — Workplace and industrial safety

### 5.1 OSHA's own enforcement priorities map onto vision

**Source:** OSHA, "Commonly Used Statistics," https://www.osha.gov/data/commonstats
(fetched 16 September 2026).

> "There were 5,283 fatal work injuries in 2023 (3.5 fatalities per 100,000
> full-time equivalent workers)."

> "Worker deaths in America are down—on average, from about 38 worker deaths a day
> in 1970 to 15 a day in 2023."

The Top 10 most-cited standards, FY2024, quoted verbatim from OSHA:

> "Fall Protection, general requirements (29 CFR 1926.501); Hazard Communication,
> general industry (29 CFR 1910.1200); Control of Hazardous Energy (lockout/tagout),
> general industry (29 CFR 1910.147); Ladders, construction (29 CFR 1926.1053);
> Respiratory Protection, general industry (29 CFR 1910.134); Powered Industrial
> Trucks, general industry (29 CFR 1910.178); Fall Protection Training, construction
> (29 CFR 1926.503); Scaffolding, construction (29 CFR 1926.451); Eye and Face
> Protection, construction (29 CFR 1926.102); Machine Guarding, general industry
> (29 CFR 1910.212)."

This is the best top-of-funnel item in the whole dossier: the regulator's own
enforcement priority list maps almost one-to-one onto visually detectable
hazards — missing fall protection, unguarded machinery, forklift proximity,
absent PPE.

### 5.2 Named, dated OSHA penalties — six employers, 2026

All quoted directly from OSHA's own news releases, not from press coverage.

| Employer | Hazard | Penalty (USD) | Release date |
|---|---|---|---|
| Orchids Builders LLC (Rockledge, FL) | Fall protection | $349,754 | 23 Jul 2026 |
| Blazey Construction Services LLC (Alvin, TX) | Excavation collapse | $343,797 | 15 Jul 2026 |
| D L Bandy Constructors Inc. (San Antonio, TX) | Crawl-space entrapment, fatal | $276,399 | 13 Jul 2026 |
| FleetPride Inc. (Corpus Christi, TX) | Confined space, fatal asphyxiation | $264,380 | 15 Jul 2026 |
| Max Home Services LLC (Fort Lauderdale, FL) | Fatal fall from roof | $172,324 | 24 Apr 2026 |
| Breland Homes Inc. (Huntsville, AL) | Trench collapse, fatal | $115,855 | 16 Apr 2026 |

The repeat-offender line from the Orchids Builders release is the quotable one:

> "OSHA cited the employer for two willful and four repeat violations and proposed
> $349,754 in penalties. Orchids Builders LLC has been inspected seven times since
> 2023 and all the cases included fall protection violations."

Seven inspections, every one of them a fall-protection violation. Inspection as
a control does not work when it is periodic.

And FleetPride, on confined space:

> "an employee asphyxiated while inspecting a tanker trailer. OSHA cited the
> company for 16 serious and three other-than-serious safety violations... OSHA has
> proposed $264,380 in penalties."

### 5.3 UK HSE prosecutions — larger fines, and one devastating camera anecdote

**Trench collapse, fatal.** HSE press release, 9 September 2026, Old Bailey
conviction. https://press.hse.gov.uk/2026/09/09/two-companies-and-an-individual-sentenced-after-worker-crushed-by-two-tonnes-of-soil-in-trench-collapse/

> "T Vaughan Limited was fined £650,000 and ordered to pay costs of £40,000.
> Appledorn Developments Limited was fined £500,000."

A contracts manager, Anthony O'Connor, "was sentenced to 10 months' imprisonment,
suspended for 18 months." HSE:

> "Gheorghita Arsene's death was as horrifying as it was preventable. He lost his
> life because basic precautions were not in place to protect workers from the
> well-known risks that come with excavation work."

**Honest note:** O'Connor had already seen another worker in the same unsupported
excavation earlier that day and did nothing. A camera would have detected exactly
what a human supervisor already detected and ignored. The gap here is
enforcement, not vision.

**The single most important quote in this domain — a camera that was there and
did not help.** Factory Services UK Limited, Knowsley, September 2026.
https://press.hse.gov.uk/2026/09/09/manufacturer-fined-after-worker-suffers-life-changing-injuries-in-conveyor-incident/

> "his arm was pulled into the moving machinery. There was no guard in place and no
> emergency stop button in the area. Working alone at the time, there was nobody
> nearby to see or hear what had happened. In an effort to raise the alarm, he
> repeatedly waved at a CCTV camera in the hope that someone monitoring the system
> would spot him and come to his aid, nobody did."

A man waved at a camera with his remaining arm and nobody was watching. That is
the case for automated detection over passive CCTV, made by a regulator, in a
sentence no one who hears it will forget. **This is the strongest single fact in
the entire dossier for a workplace-safety pitch.**

**Forklift / pedestrian.** Clearaway Recycling Limited, sentenced 13 August 2026.
https://press.hse.gov.uk/2026/08/18/waste-and-recycling-company-fined-400000-after-woman-crushed/

> "The company was fined £400,000 and ordered to pay costs of £10,259... HSE
> statistics show that fatal incidents in the waste and recycling sector are ten
> times more likely than the all-industry average."

> "The waste and recycling industry saw six workers killed during 2025/26."

**Forklift load.** Smith & Nephew Medical Ltd, Hull, sentenced 14 September 2026.

> "The company was fined £230,000 and ordered to pay £111,000 in costs... There was
> no assessment of the suitability of the company's two-tonne forklift truck in
> relation to the characteristics of the load... The unloading operation was not
> meaningfully controlled or supervised."

All figures above are GBP, as stated. No currency conversion has been applied.

### 5.4 Forklift and warehouse injury data

**Source:** Putz Anderson V, Schulte PA, Novakovich J, Pfirman D, Bhattacharya A.
"Wholesale and retail trade sector occupational fatal and nonfatal injuries and
illnesses from 2006 to 2016." *Am J Ind Med* 2020;63(2):121–134. PMID 31709592.
NIOSH authorship.

> "In 2016, 553,100 injuries and illnesses and 461 fatalities occurred among WRT
> [wholesale and retail trade] workers... From 2006 through 2016, wholesale sector
> fatality rates (4.9/100,000 FTE) exceeded private industry rates (3.8/100,000
> FTE). The largest causal fatal factors were transportation in wholesale and
> violence in retail."

Older but forklift-specific: Janicak CA, Deal GA. "Occupational fatalities
involving forklifts." *J Trauma* 1999;47(6):1084–7. PMID 10608537.

> "Forklift rollovers account for approximately 18% of all fatalities involving
> forklifts and many of the more serious injuries in the workplace."

**Caveat:** that is 1999 data. No more recent forklift-specific fatality count was
obtainable — BLS, NHTSA and CDC WONDER are all blocked from this account. Do not
substitute a remembered figure.

### 5.5 The closest thing to a precedent — and why it is weak

**Source:** Bobick TG, Hause M, Socias-Morales C, Gwilliam M, Decker T (NIOSH).
"Forklift Safety — Pilot Study Evaluation of Retrofit Lights." *Professional
Safety* 2020;65(12):41–45. PMCID PMC11119981.

> "Blue and red lights were retrofitted onto three forklifts and used for four
> months in a warehouse environment to increase the awareness of approaching
> vehicles... Feedback indicated that all nine employees thought the addition of
> the lights increased the visibility of the forklifts and improved safety by
> making the vehicles more conspicuous."

Nine employees, self-report, four months, one warehouse. This shows *perceived*
benefit from visibility technology at the forklift-pedestrian interface. It is
**not** measured incident reduction. Cite it as precedent, never as proof.

### 5.6 PPE — and the limit of detecting presence

**Source:** OSHA, "Personal Protective Equipment in Construction" final rule,
Federal Register, 12 December 2024, doc 2024-29220.

> "If PPE does not fit properly, it can make the difference between an employee
> being safely protected, having inadequate protection, or being dangerously
> exposed. In some cases, ill-fitting PPE may not protect an employee at all... The
> issue of improperly fitting PPE is particularly important for smaller construction
> workers, including some women, who may not be able to use currently existing
> standard-size PPE."

**The honest limit:** a camera reliably flags *absence* — no hard hat, no harness.
OSHA's own rulemaking says the dominant real-world failure is PPE that is present
but ill-fitting, which looks correct on camera and does not protect. Do not sell
presence-detection as effectiveness-detection.

### 5.7 Confined space deaths — UNVERIFIED

OSHA's confined-space page gives only qualitative language:

> "Confined and enclosed space operations have a greater likelihood of causing
> fatalities, severe injuries, and illnesses than any other type of shipyard work."

The Federal Register OMB renewal notices for the permit-required confined space
standard contain only paperwork-burden counts (221,852 establishments with permit
spaces; 1,505,672 permit-space entrants) and **no fatality figure**. The commonly
cited "about 100 confined-space deaths a year" could not be sourced. **Do not use
it.**

An adjacent, properly quantified slice does exist, for agriculture: Nour MM, et
al., *J Agric Saf Health* 2021;27(2):105–122, PMID 34350740 —

> "The overall fatality rate was 57%... asphyxiations accounting for 42% of all
> cases... the Purdue Agricultural Confined Space Incident Database (PACSID),
> which contained over 2,400 individual U.S. cases"

---

## 6. Domain 5 — Water safety and drowning

### 6.1 Global and US scale

**Source:** WHO, "Drowning" fact sheet.
https://www.who.int/news-room/fact-sheets/detail/drowning

> "There are around 300 000 annual drowning deaths worldwide."

> "Drowning is the fourth leading cause of death for children aged 1–4 years and
> the third leading cause of death for children aged 5–14 years. Ninety-two percent
> of drowning deaths occur in low- and middle-income countries."

> "Children and young adults aged 0–29 years account for over half (57%) of all
> drowning deaths... The risk of drowning increases when children interact with
> water outside of active adult supervision."

**US, and a worsening trend.** Clemens T, et al. "Vital Signs: Drowning Death
Rates — United States, 2019–2023." *MMWR Morb Mortal Wkly Rep* 2024 May
23;73(20):467–473. PMID 38781109, PMCID PMC11115434. (Retrieved via PubMed
Central; cdc.gov itself is blocked.)

> "Drowning is the cause of approximately 4,000 U.S. deaths each year and
> disproportionately affects some age, racial, and ethnic groups."

> "Compared with an overall unintentional drowning death rate of 1.2 per 100,000
> persons in 2019, rates were significantly higher in 2020 (1.4; 10.5% increase),
> 2021 (1.4; 13.7%), and 2022 (1.3; 9.1%)... Drowning death rates were highest
> among children aged 1–4 years in all years and increased significantly in 2021
> (3.1; 28.9%) and 2022 (3.1; 28.3%) compared with 2019."

And the pool-specific share, from CDC's NAFIS report (*MMWR Surveill Summ* 2016
May 20;65(5):1–26, PMID 27199095):

> "Drowning is the leading cause of injury deaths in children aged 1-4 years, and
> approximately half of fatal drownings in this age group occur in swimming pools."

### 6.2 The failure mode — lifeguards cannot scan fast enough

**Source:** Smith, Obine, Talbot et al. "An Investigation of the 10:20 Protection
Rule for Detecting Aquatic Hazards." *Europe's Journal of Psychology*
2025;21(3):194–207. PMID 41727263, PMCID PMC12923190.

> "the stark reality persists that incidents of drowning continue to occur within
> zones overseen by trained lifeguards."

The central finding:

> "it is not possible for lifeguards to scan the full zone every 10 seconds,
> despite explicit instructions to do so, and thus the 10:20 protection rule should
> be carefully considered if agencies are advocating it as an effective scanning
> strategy."

And, remarkably, on the evidence base for the worldwide standard:

> "the evidence that underpins the efficacy of this guidance is limited and the
> authors are unable to access any peer-reviewed studies supporting this system"

— this of a technique the same paper calls "the most widely used scanning
technique used by lifeguards in the world," endorsed by the UK Health and Safety
Executive. The mechanism, citing Sharpe et al.:

> "high task demands through increased bather numbers and decreased 'drown'
> duration negatively influenced [lifeguard] performance, as attentional lapses are
> likely to increase when the task at hand is objectively more challenging."

This is a better foundation than the manikin-detection studies originally sought:
it is recent, peer-reviewed, and directly about scanning failure.

### 6.3 What the technology literature actually claims

**Source:** Jalalifar S, Belford A, Erfani E, et al. "Enhancing Water Safety:
Exploring Recent Technological Approaches for Drowning Detection." *Sensors
(Basel)* 2024;24(2):331. PMID 38257424, PMCID PMC10820385.

On lifeguards, from the review's own comparison table:

> "Limited physical endurance and potential for fatigue; Restricted field of view
> and potential for human error"

On image processing:

> "Continuous monitoring and real-time data collection; Accuracy and effectiveness
> in recognising and differentiating drowning incidents"

against limitations including "Maintenance and calibration requirements; Limited
underwater communication," and:

> "the image-processing approach requires substantial resources and sophisticated
> MLAs, making it costly and complex to implement."

**The gap to disclose:** this is a 2024 engineering review, not an outcome trial.
No peer-reviewed study measuring real-world lives saved or false-negative rates
for a deployed pool camera system was found. Say so rather than implying the
effectiveness question is settled.

### 6.4 Law and litigation

**Virginia Graeme Baker Pool and Spa Safety Act.** Federal Register, 85 FR 58263,
18 September 2020.

> "The VGBA, 15 U.S.C. 8001 et seq., took effect on December 19, 2008. The VGBA's
> purpose is to prevent drain entrapment and child drowning in swimming pools and
> spas."

CPSC's 2019 entrapment report found "six injuries (and no deaths) due to hair
entrapment in the years 2014 to 2018" under the current standard. **Honesty
point:** VGBA solved a narrow mechanism — drain entrapment — with mandatory
hardware, not with sensing. It is not a precedent for camera regulation.

**A case that argues against us, which is why it is worth knowing.** *Estate of
Sthella Feliciano v. Rivertree Landings Apartments, LLC*, Fla. 2d DCA,
No. 2D2023-0561, 10 May 2024 (full opinion retrieved). A six-year-old autistic
child "exited their apartment unnoticed," walked past a fenced pool to an
unfenced adjacent river, and drowned. The court **affirmed summary judgment for
the property owner**. The incident was on video — "Rivertree surveillance video
depicts the child walking..." — and nobody was watching it in real time, and it
happened outside any zone a pool camera would cover. Two lessons: recorded video
is not detection, and perimeter scope matters more than model accuracy.

Also identified: *Sta-Rite Industries, Inc. v. Levey*, 909 So.2d 901 (Fla. 3d DCA
2004) — suction-entrapment wrongful-death product liability, predating VGBA.
Damages figure **UNVERIFIED**.

**No drowning settlement or verdict amount was verified.** The money evidence in
this domain is thin.

---

## 7. Domain 6 — Elderly falls and hospital inpatient falls

### 7.1 Global scale

**Source:** WHO, "Falls" fact sheet.
https://www.who.int/news-room/fact-sheets/detail/falls

> "Falls are the second leading cause of unintentional injury deaths worldwide.
> Each year an estimated 684 000 individuals die from falls globally of which over
> 80% are in low- and middle-income countries. Adults older than 60 years of age
> suffer the greatest number of fatal falls... Though not fatal, approximately 37.3
> million falls severe enough to require medical attention occur each year."

> "falls are responsible for over 38 million DALYs... and result in more years
> lived with disability than transport injury, drowning, burns and poisoning
> combined."

WHO also names the institutional setting directly, listing "Ensure adequate
staff-to-resident ratios in residential care facilities" among its recommended
prevention measures — i.e. the acknowledged control is staffing, which is
expensive and in short supply.

### 7.2 The money — the landmark US cost figure

**Source:** Florence CS, Bergen G, Atherly A, et al. "Medical Costs of Fatal and
Nonfatal Falls in Older Adults." *J Am Geriatr Soc* 2018 Apr;66(4):693–698.
PMID 29512120, PMCID PMC6089380.

> "In 2015, the estimated medical costs attributable to fatal and nonfatal falls
> was approximately $50.0 billion. For nonfatal falls, Medicare paid approximately
> $28.9 billion, Medicaid $8.7 billion, and private and other payers $12.0 billion.
> Overall medical spending for fatal falls was estimated to be $754 million."

Sample: "Fatal falls from the 2015 NVSS (N=28,486)."

**$50.0 billion is the largest verified healthcare money figure in this dossier.**

### 7.3 Hospital inpatient fall rates — a range, not a number

Three independent primary studies, all fetched:

- **UK NHS dementia ward:** baseline 5.4 falls per 1,000 occupied bed days,
  reduced to "1.4 falls per 1000 occupied bed days" by a QI intervention — a 74%
  reduction. Sorlie et al., *BMJ Open Qual* 2026 Jun 1;15(2):e003988, PMID 42225373.
- **Chinese tertiary hospital, 325,377 admissions:** "fall rate per 1000
  patient-days (0.22‰ vs. 0.29‰, p = 0.000)". Liao, Guo, Li, Liu, *Sci Rep* 2026
  Mar 13;16(1):9560, PMID 41826360.
- **NYU Langone inpatient rehabilitation:** "Among the 6238 unique patient
  admissions, a total of 40 falls were identified. The rate of falling was 0.43
  falls per 1000 patient days. The majority of falls occurred because of buckling
  (47.5%) and during gait training (40.0%)." Camillieri et al., *Phys Ther* 2025
  Aug 5;105(8):pzaf096.

**Cite the range, roughly 0.2 to 5.4 falls per 1,000 patient-days depending on
unit acuity, not a single number.** Rehab and dementia wards run an order of
magnitude above general medical wards.

Note the NYU mechanism breakdown: nearly half of falls were *buckling*, and 40%
happened *during gait training* — that is, while a therapist was already present
and holding on. A camera does not prevent a knee giving way.

### 7.4 The regulator has already made falls a financial liability

**Source:** CMS, "Medicaid Program; Payment Adjustment for Provider-Preventable
Conditions Including Health Care-Acquired Conditions," Federal Register,
6 June 2011. (The official HAC category list, quoted from 75 FR 50084–50085.)

> "Foreign Object Retained After Surgery. Air Embolism. Blood Incompatibility.
> Stage III and IV Pressure Ulcers. Falls and Trauma. + Fractures. + Dislocations.
> + Intracranial Injuries. + Crushing Injuries. + Burns. + Electric Shock."

and CMS's rationale for extending it to Medicaid:

> "certain Medicare HACs, such as Foreign Objects Retained After Surgery, Air
> Embolism, Blood Incompatibility, Stage 3 and 4 Pressure Ulcers, Falls and
> Trauma... are clinically applicable to all Medicaid populations."

This confirms the commercial premise exactly: a fall with injury on a hospital's
watch is a Hospital-Acquired Condition, and the hospital gets **no additional
payment** for treating what it caused. The buyer has a budget line already.

Note the pleasing overlap: "Foreign Object Retained After Surgery" is on the same
list as "Falls and Trauma" — one CMS non-payment policy covers two of our
candidate products.

### 7.5 Nursing home penalties — current, computable, large

**Source:** CMS Care Compare "Penalties" dataset, id `g6vv-u9sr`,
https://data.cms.gov/provider-data/dataset/g6vv-u9sr, most recent release
26 August 2026. CMS's own description:

> "A list of the fines and payment denials received by nursing homes in the last
> three years."

Computed directly from the downloaded CSV (15,696 rows, preserved at
`/tmp/ocv-evidence/NH_Penalties.csv`): **13,256 fine records totalling
$456,752,787**, plus 2,440 separate payment-denial penalties, over the trailing
three-year window.

**Caveat that must travel with this number:** the file has no deficiency-type or
F-tag column, so the fall-specific share cannot be isolated. This is *total*
nursing-home fine exposure, of which falls are one substantial contributor, not
the whole. Present it as the size of the enforcement pool, never as "fines for
falls."

### 7.6 Litigation — real cases, shaky amounts

- *Conrad ex rel. Conrad v. Alderwood Manor*, 78 P.3d 177 (Wash. Ct. App. Div. 3,
  2003). The court's own words: **"This is a claim for injuries to and the death
  of an elderly nursing home patient. The claim resulted in a multimillion dollar
  jury verdict."** The exact figure is UNVERIFIED — "multimillion dollar" is the
  court's characterisation and is all that could be retrieved.
- *Cleveland Nursing & Rehabilitation, LLC v. Estate of Gully*, Miss. Sup. Ct.,
  20 October 2016: "the jury... awarded the Estate of Annie Mae Gully... $1,000,000".
  The verdict amount and citation are solid; **whether the injury was fall-related
  is UNVERIFIED** (the Mississippi courts site was down at fetch time). Do not
  present this as a fall verdict.

CourtListener's HTML opinion pages sit behind an AWS WAF bot challenge, which is
why full text could not be pulled for either.

### 7.7 The privacy problem, stated honestly

**Source:** Tham, Brady, Ziefle, Dinsmore. "Barriers and Facilitators to Older
Adults' Acceptance of Camera-Based Active and Assisted Living Technologies: A
Scoping Review." *Innov Aging* 2024 Nov 29;9(2):igae100. PMID 39968358.

> "Dominant barriers concerned the technology's privacy-invasive, obtrusive, and
> stigmatizing qualities"

**Source:** Maidhof, Offermann, Ziefle. "Eyes on privacy: acceptance of
video-based AAL impacted by activities being filmed." *Front Public Health* 2023
Jul 4;11:1186944. PMID 37469701.

> "identified distinct evaluation patterns for 25 activities of daily living with
> very high (e.g., changing clothes, showering) and very low privacy needs (e.g.,
> gardening, eating, and drinking)... The strongest barrier perception was found
> for intimate activities and mainly regarded privacy concerns."

**This is the domain's core contradiction and it must be in the pitch, not hidden
from it.** Bathing, toileting and dressing are where a large share of falls
happen, and they are exactly the activities where camera acceptance is lowest by
a wide margin. The coverage gap and the acceptance gap point at the same rooms.
A credible product answers this with on-device processing and pose abstraction
(never storing or transmitting imagery), and says so up front.

## 8. Most quotable single facts

Ranked by how well each would open a five-minute video.

1. **"In an effort to raise the alarm, he repeatedly waved at a CCTV camera in the
   hope that someone monitoring the system would spot him and come to his aid,
   nobody did."** — UK HSE press release, 9 September 2026, on the Factory
   Services UK conveyor case. A man whose arm had been pulled into unguarded
   machinery waved at a camera with the arm he had left, and nobody was watching.
   It is the case for automated detection over passive CCTV, made by a regulator,
   in one sentence.

2. **"None of the respondents had PPH (>1000 ml blood loss) following [visual
   estimation]."** — while 21% actually had it. *Ther Adv Reprod Health*,
   17 October 2024, PMID 39435121. By eye nobody haemorrhaged; by measurement one
   in five did.

3. **PG&E "agreed to plead guilty to 84 counts of involuntary manslaughter,"** paid
   into a **$13.5 billion** Fire Victim Trust — and then installed **"144 weather
   stations and 60 high definition cameras."** PG&E Corp Form 10-K, SEC,
   25 February 2021. The defendant's own remedy was cameras.

4. **"Orchids Builders LLC has been inspected seven times since 2023 and all the
   cases included fall protection violations."** OSHA news release, 23 July 2026,
   with $349,754 in proposed penalties. Periodic inspection does not work as a
   control.

5. **"it is not possible for lifeguards to scan the full zone every 10 seconds,
   despite explicit instructions to do so"** — and the authors "are unable to
   access any peer-reviewed studies supporting this system," the most widely used
   lifeguard scanning technique in the world. *Europe's Journal of Psychology*,
   2025, PMID 41727263.

6. **"CVS was achieved in 314/513 cases (61.2%)"** — falling to **14.3%** in the
   hardest cases. *Surgical Endoscopy*, 3 August 2026. The safety step that
   prevents bile duct injury is skipped most where it is needed most.

7. **"In 2015, the estimated medical costs attributable to fatal and nonfatal falls
   was approximately $50.0 billion."** Florence et al., *J Am Geriatr Soc*, April
   2018. The largest verified healthcare figure in this dossier.

8. **"62 of [117 fatal work-zone worker injuries] involved a worker on foot being
   struck by a motor vehicle."** FHWA, Federal Register, 20 September 2023. Over
   half of the deaths are one geometry.

9. **"Given medical and liability costs of >$200,000 per incident"** for a retained
   sponge — against alternatives priced at "$95,000" to "$1.4 million per RSS
   event prevented."

10. **"The Surgical Safety Checklist has been shown to reduce complications and
    mortality by over 30 percent... completed in under 2 minutes."** WHO. Sets the
    bar low, credibly, and WHO set it.

11. **"Each year an estimated 684 000 individuals die from falls globally"** plus
    "approximately 37.3 million falls severe enough to require medical attention."
    WHO falls fact sheet.

12. **"Every day in 2023, over 700 women died from preventable causes related to
    pregnancy and childbirth. A maternal death occurred almost every 2 minutes."**
    WHO, 2023 data.

---

## 9. Where the evidence did not hold up

Drop, de-prioritise, or never quote these.

**Dropped outright — no verified numbers at all:**

- **Wrong-site surgery.** Joint Commission sentinel event data returned HTTP 403.
  No frequency verified.
- **Surgical fires.** FDA safety communication returned HTTP 503. Nothing verified.
- **Confined-space annual death toll.** OSHA's own page is qualitative only, and
  the Federal Register notices contain paperwork-burden counts but no fatality
  figure. The remembered "about 100 a year" could not be sourced — **do not use
  it.**
- **Liberty Mutual Workplace Safety Index.** Host returned 403. **Do not cite any
  dollar figure from it.**

**Looked strong going in, did not hold up:**

- **Post-disaster damage assessment and search triage.** This was one of the two
  domains chosen in advance, and it produced nothing citable: no deaths-from-delay
  figure, no inquiry quote naming the delay, no cost. cal fire, CPUC, the Hawaii AG
  reports, WMO and GAO were all blocked or missing. The earthquake entrapment
  literature yielded only extrication times of hours, with no survival-versus-time
  curve. **The wildfire *ignition* framing is world-class; the *response* framing
  is not.** If a disaster product is wanted, pivot to ignition detection.

**Use, but with a stated caveat:**

- **Retained surgical item frequency.** Estimates span 1/1,000 to 1/32,672 — a
  factor of thirty — and the literature says "the true incidence is largely
  unknown." Lead with cost per incident, never a rate.
- **"85 deaths in the Camp Fire."** Commonly cited, not verified here. The verified
  figure is **84 counts of involuntary manslaughter**.
- **The Singapore blood-loss study.** Found its underestimation "not statistically
  significant" with very wide confidence intervals. Quote its direction and
  conclusion, never its millilitre point estimates.
- **The $456.8M CMS nursing-home penalty total.** Real and computed from the raw
  CMS file, but it has no deficiency-type column, so the fall-specific share cannot
  be isolated. It is the size of the enforcement pool, not "fines for falls."
- **Forklift rollover share (18%).** Peer-reviewed but from 1999. No current
  forklift-specific figure was obtainable with BLS, NHTSA and CDC WONDER blocked.
- **DOT's $12.5M value of a statistical life.** Fetched from a CPSC cross-agency
  table rather than DOT's own page, and a later Federal Register notice withdrew
  the guidance. The federal VSL framework is currently unsettled.
- **The NIOSH forklift-lights pilot.** Nine employees, self-report, one warehouse,
  four months. Perceived benefit, not measured incident reduction. Precedent, not
  proof.

**Litigation leads with no verified dollar amount** — named and dated, but do not
attach a number to any of them: *TxDOT and Vulcan Materials v. Kristina Smith*
(2024); *JTL Group v. Gray-Dockham* (2022); *Watson v. BNSF* (2024); *Conrad v.
Alderwood Manor* (2003, "multimillion dollar" is the court's own word and all
that could be retrieved); *Sta-Rite Industries v. Levey* (2004). *Cleveland
Nursing v. Estate of Gully* has a solid $1,000,000 verdict but **unverified
fall causation** — do not call it a fall verdict.

**Two cases that argue against us, and are worth knowing for that reason:**

- *Estate of Feliciano v. Rivertree Landings Apartments* (Fla. 2d DCA, 10 May
  2024). The drowning was on surveillance video, nobody was watching it, and it
  happened outside any zone a pool camera would cover. Summary judgment for the
  owner was affirmed. Recorded video is not detection.
- The HSE trench case (9 September 2026). The contracts manager had already seen a
  worker in the same unsupported excavation earlier that day and did nothing. A
  camera would have detected exactly what a human already detected and ignored.
  That gap is enforcement, not vision.

---

## 10. Method and reproducibility

All pages fetched with `curl` and a browser User-Agent (web tools are blocked on
this account). SEC requires a *declared* User-Agent containing contact details —
a browser UA is rejected with "Your Request Originates from an Undeclared
Automated Tool"; `-A "OpenCV-Competition-Research <email>"` works.

**Working hosts confirmed:** osha.gov, press.hse.gov.uk, ntsb.gov (including
report PDFs), who.int, federalregister.gov (API and full-text HTML),
data.cms.gov, data.cdc.gov, fema.gov, eutils.ncbi.nlm.nih.gov (PubMed and PMC
full text), europepmc.org, fda.gov, epa.gov, ec.europa.eu/eurostat,
efts.sec.gov (EDGAR full-text search), courtlistener.com **search** API v4.

**Blocked:** cdc.gov main site, wonder.cdc.gov, bls.gov, nhtsa.gov, gao.gov,
cpsc.gov, jointcommission.org, fire.ca.gov, transportation.gov,
business.libertymutual.com, railroads.dot.gov, the FDA surgical-fire page, and
the specific CPUC and Hawaii AG pages tried. `ops.fhwa.dot.gov` and
`hse.gov.uk/statistics/fatalinjuries.htm` are 404 — pages removed.
`safetydata.fra.dot.gov` is marked "FRA - Retired."

**CourtListener's limit, which shaped this dossier.** The v4 *search* API works
without a key and finds cases. The *opinion* and *cluster* detail endpoints return
**401 Unauthorized** without an API key, and the HTML opinion pages sit behind an
AWS WAF bot challenge. That is why the litigation entries here have names and
dates but almost no verified dollar amounts. **Getting a free CourtListener API
key is the single highest-value next step** if named verdict figures matter to the
submission.

**Three routes that worked unusually well and should be reused:**

1. **EDGAR full-text search** for corporate liability — a company's own 10-K
   states settlements and criminal pleas in quotable prose:
   `https://efts.sec.gov/LATEST/search-index?q=%22PHRASE%22&forms=10-K`
2. **The Federal Register full-text API** as a back door to blocked agencies —
   NHTSA and BLS figures appear verbatim inside FHWA and OSHA rulemakings, and
   CMS's official HAC list is quoted inside a Medicaid rule.
3. **PubMed Central via eutils** as a mirror for blocked CDC content — the MMWR
   drowning Vital Signs report was unreachable on cdc.gov and trivially available
   through PMC.

---

## 11. Open items

- **Three domains were never researched:** agriculture and food (crop disease,
  produce grading, contamination recalls), environmental (illegal dumping,
  pipeline and tailings failures, air quality and smoke, methane emissions), and
  retail/logistics/insurance (cargo and parcel damage disputes, vehicle handover
  disputes, counterfeit and expired goods). Nothing here speaks to them either
  way.
- **Utilities and infrastructure** (domain 9) is covered only through the PG&E
  case in §3.1, which is strong enough to carry a product on its own. Bridge and
  rail inspection were not researched.
- **Surgical malpractice money** remains the biggest hole in the two chosen
  domains: no named verdict or settlement for bile duct injury or retained items
  was verified. CourtListener with an API key is the fix.
- **A deployed-system outcome trial** could not be found for any camera safety
  product in any domain — pool drowning detection, fall detection, or workplace
  monitoring. Every technology citation here is an engineering review or a small
  pilot. If a judge asks "has this been shown to save lives in the field," the
  honest answer today is that nobody has published that evidence. Better to say so
  than to be caught implying otherwise.
