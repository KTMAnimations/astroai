# External M67 observations

`gruner2023_m67.csv` is the 47-star Table C.2 from Gruner, Barnes & Weingrill
(2023), [New insights into the rotational evolution of near-solar age stars from
the open cluster M67](https://arxiv.org/abs/2305.16997). It was retrieved from the
[pinned gyro-interp copy](https://github.com/lgbouma/gyro-interp/blob/33538508939352186bf00cd77907d1dafe22b9dc/gyrointerp/data/literature/Gruner_Barnes_Weingrill_2023_M67_tableC2.csv).
`gruner2023_manifest.json` records the exact URL and checksum.

These are the authors' effectively single main-sequence stars. The table directly
supplies dereddened Gaia DR3 BP-RP, so no temperature-to-color approximation is
used. Their reddening correction is E(BP-RP) = 0.054 mag; it differs in procedure
from the original catalogue's 3D dust-map correction. HPS indicates half-period
sequence objects, whose periods have already been doubled in this published
table. The pipeline retains the published values unchanged.

The fixed color domain contains 28 rows. Removing the three whose Gaia IDs occur
anywhere in the entire original 8,412-entry catalogue leaves 25 external stars.
No period-residual or outlier cut is applied. The external selection ledger
records all 47 rows and both membership-in-original and color-domain indicators.
All M67 stars are excluded from both discovery and validation in the frozen
models used for this external test.

The main external reference is 4,000 Myr, with 3,870 Myr recorded as a sensitivity.
The paper also discusses an isochrone fit near 3,600 Myr; none of these ages is an
exact truth. The age scale and selection differences limit interpretation. This
is an additional sample in one old cluster, not multiple new clusters or a field
star validation. Observations and authorship belong to the cited researchers.
