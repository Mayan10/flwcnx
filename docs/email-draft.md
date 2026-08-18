# Draft: data request to the StarNet authors

Send from Mayan's own account. Adjust the project description to match how the
work is actually being framed for assessment.

**To:** zikunliu2@illinois.edu
**Cc:** deepakv@illinois.edu
**Subject:** StarNet dataset links expired - request for access (CoNEXT 2025)

---

Dear Zikun Liu and Prof. Vasisht,

I am a student working on a predictive bandwidth allocation system for Starlink
access links, building directly on your CoNEXT 2025 paper, "Vivisecting
Starlink Throughput: Measurement and Prediction".

I am writing because the three OneDrive links in the repository README (US,
Germany and Canada) now return "Sorry, the link has expired." I checked the
repository for a mirror and the ACM DL entry for an artifact, and could not
find an alternative source.

Would it be possible to share the `dataset_tp_sat.pkl` files for the three
locations, or a refreshed link? A time-limited or read-only share would be
completely fine.

Briefly, what I am doing with it, in case it is useful context. I am
reproducing StarNet as the forecasting backbone, and the contribution I am
attempting sits on top of it: a regime-conditioned calibration layer that turns
a point throughput forecast into a safe lower bound whose overestimation rate
is controlled within each operating regime, where regime is defined by
covariates the terminal has at prediction time (15 second phase, serving
satellite elevation and distance, candidate count). The motivation is the
conditional risk failure that Xie et al. report for a single global quantile:
risk held on average, and lost precisely in the low-capacity regime. Your
serving-satellite resolution is what makes those regimes observable at all,
which is why this dataset specifically is the one I need.

I am happy to cite the dataset however you prefer, and to share back anything
that comes out of it.

Thank you for making the tooling and the models public in the first place.

Best regards,
Mayan Sharma
mayan25sharma@gmail.com
