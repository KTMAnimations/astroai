# Research Plan: Graph Neural Network for Open-Cluster Chemical Homogeneity

**Proposal 5 from top_5_proposals.md**

This document is a step-by-step guide from an empty folder to a completed paper. Every architectural choice, data query, and evaluation metric is explained without specialist vocabulary.

---

## What is being created that is new

The naive version — "stack SAGEConv layers from PyTorch Geometric and classify clusters" — is an application of standard tools to a new dataset. Publishable at A&A as-is, but weak.

The novel contribution has three parts that make this a methods-AND-science paper:

1. **Noise-aware message passing (UncertaintySAGEConv).** Standard GraphSAGE averages neighbor features with equal weight. But each star's XP metallicity has a known error bar, and errors vary by a factor of 10 across stars in the same cluster (bright stars have small errors, faint stars have large errors). The novel convolution weights neighbors by the inverse of their measurement variance during aggregation: `agg = sum(h_i / σ_i²) / sum(1/σ_i²)` instead of `agg = mean(h_i)`. This is a one-line change to the forward pass but it is a NEW convolution variant that is physically motivated and architecturally novel. Name it `UncertaintySAGEConv`. Compare it to standard SAGEConv in the ablation — if it outperforms, that is both a methods result and evidence that measurement uncertainty matters for graph-based astronomical inference.

2. **Attention-based interpretability.** When using GATv2, extract the learned attention weights after training and analyze which stars the network considers most informative for the homogeneity decision. Are they the brightest stars (smallest errors)? The most chemically extreme stars? The ones at the cluster center? This turns the model from a black box into a scientific instrument — it tells astronomers which observations to prioritize for follow-up.

3. **Calibrated uncertainty via MC-Dropout.** Instead of outputting a single homogeneity score, apply dropout at inference time (50 forward passes with dropout active) and report the mean and standard deviation of the homogeneity probability. Report calibration: does the network's stated 90% confidence correspond to 90% empirical accuracy? If not, apply temperature scaling to fix it. Calibrated uncertainties are essential for the catalog to be usable downstream.

---

## Phase 1: Environment setup

**Time: 1-2 hours**

1. Create a Python environment:
   ```
   python -m venv gnn_cluster_env
   source gnn_cluster_env/bin/activate
   ```

2. Install core packages:
   ```
   pip install torch numpy pandas matplotlib scipy astropy astroquery
   ```

3. Install PyTorch Geometric (the graph neural network library). Follow the official install instructions at `pyg.org` — the exact command depends on your PyTorch and CUDA versions:
   ```
   pip install torch-geometric
   pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.x.x+cu12x.html
   ```
   If you are on CPU only, use the CPU wheel URL instead.

4. Install the Gaia XP coefficient package:
   ```
   pip install GaiaXPy
   ```

5. Create your project structure:
   ```
   mkdir -p project/{data,graphs,models,results,figures,notebooks}
   ```

---

## Phase 2: Download the cluster catalog

**Time: 2-3 hours**

The cluster catalog is from Hunt & Reffert. Use the **updated 2024 version** (J/A+A/686/A42), which supersedes the 2023 version.

### Option A: Download from VizieR using astroquery

```python
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1  # no row limit

# Download cluster properties table
clusters = Vizier.get_catalogs("J/A+A/686/A42/clusters")[0]
clusters.write("project/data/clusters.fits", format="fits")

# Download member stars table (this is large: ~1.3 million rows)
members = Vizier.get_catalogs("J/A+A/686/A42/members")[0]
members.write("project/data/members.fits", format="fits")
```

### Option B: Bulk download from CDS FTP

```
cd project/data
wget -r https://cdsarc.cds.unistra.fr/ftp/J/A+A/686/A42/
```

### What you now have

- **Cluster table:** ~7,000 rows. Each row is one cluster with properties: center position (RA, Dec), mean proper motion, mean parallax, estimated age, number of members.
- **Member table:** ~1.3 million rows. Each row is one star with: Gaia DR3 source ID, cluster ID, membership probability, position (RA, Dec), proper motions (pmRA, pmDec), parallax, G-band magnitude, BP-RP color.

### Quality check

Pick 3 clusters. For each, select its members from the member table. Plot their sky positions (RA vs Dec). They should form a tight clump. Plot their proper motions (pmRA vs pmDec). They should also cluster tightly. If either plot shows widely scattered points, filter to membership probability > 0.7.

---

## Phase 3: Cross-match members with Gaia XP spectra

**Time: 4-8 hours (mostly waiting for queries)**

Each star in the member table has a Gaia DR3 `source_id`. You need the XP spectrum coefficients for each star.

### What XP coefficients are

Gaia observed each star's light spread across wavelength (a spectrum) using two low-resolution prisms called BP (Blue Photometer) and RP (Red Photometer). The raw data is compressed into 55 BP coefficients and 55 RP coefficients = 110 numbers per star. These 110 numbers are the "features" for each node in the graph.

### How to get them

**Option A: Gaia Archive ADQL query (recommended for large batches)**

Go to `https://gea.esac.esa.int/archive/` and submit an ADQL query. Because 1.3M stars is too many for a single query, batch by cluster:

```sql
SELECT m.source_id, xp.bp_coefficients, xp.rp_coefficients
FROM gaiadr3.xp_continuous_mean_spectrum AS xp
JOIN user_uploaded_table AS m ON xp.source_id = m.source_id
```

In practice, upload source IDs for each cluster (or groups of clusters) as a user table, then join. The Gaia Archive supports table uploads up to ~3M rows.

**Option B: Use GaiaXPy for smaller batches**

```python
from gaiaxpy import calibrate
# For a list of source_ids
calibrated_spectra, sampling = calibrate(source_ids_list)
```

**Option C: Use the pre-computed Andrae et al. 2023 metallicities**

Instead of raw XP coefficients, download the Andrae et al. 2023 metallicity catalog from VizieR. This gives you a single `[M/H]` value per star with its error, computed from XP by machine learning. This is simpler than using raw XP coefficients, and sufficient for a first version.

```python
andrae = Vizier.get_catalogs("J/ApJS/XXX/andrae2023")[0]
```

### What to store

For each star in the member table, you need:
- Gaia source_id
- Cluster ID
- Position: RA, Dec
- Kinematics: pmRA, pmDec, parallax
- XP features: either 110 raw coefficients OR the Andrae [M/H] value with error
- Membership probability

Save as a single file per cluster (a dictionary or HDF5 group) for easy graph construction later.

**Not every star has XP spectra.** Gaia DR3 published XP for about 220M of 1.8B sources. Expect ~70-80% match rate. Stars without XP should be excluded from the graph but noted in the paper.

---

## Phase 4: Cross-match with APOGEE DR17 for ground truth

**Time: 2-3 hours**

APOGEE is a high-resolution infrared spectrograph that measured detailed chemical abundances for ~700,000 stars. A subset of these are in open clusters.

### Download APOGEE DR17

From the SDSS website, download the `allStar` file:
```
wget https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits
```
This is ~1.5 GB. It contains one row per star with: position, Teff, log(g), [Fe/H], [Mg/Fe], [Si/Fe], [Ca/Fe], and 20+ other abundance ratios.

### Cross-match to cluster members

Match APOGEE stars to cluster members by Gaia source_id. The APOGEE catalog includes a `GAIAEDR3_SOURCE_ID` column for this purpose.

```python
import astropy.io.fits as fits

apogee = fits.open("allStar-dr17-synspec_rev1.fits")[1].data
apogee_ids = apogee["GAIAEDR3_SOURCE_ID"]

# For each cluster, find members that appear in APOGEE
for cluster_id in cluster_list:
    member_ids = get_member_source_ids(cluster_id)
    overlap = np.intersect1d(member_ids, apogee_ids)
    if len(overlap) >= 3:  # need at least 3 stars for a meaningful homogeneity test
        # Store APOGEE [Fe/H] and other abundances for these stars
        ...
```

### What you get

Expect 200-400 clusters with at least 3 APOGEE members. For each of these clusters, you know:
- The "true" [Fe/H] of each star (from APOGEE, accurate to ~0.05 dex)
- The "true" chemical dispersion within the cluster (standard deviation of APOGEE [Fe/H])
- Whether the cluster is genuinely homogeneous (σ_[Fe/H] < 0.05 dex) or heterogeneous (σ_[Fe/H] > 0.05 dex)

This is your labeled training set. The remaining ~6,500 clusters without APOGEE are the unlabeled set you will predict on.

### Define the training labels

For each cluster with APOGEE data:
- **Mean metallicity:** average of APOGEE [Fe/H] values for cluster members
- **Homogeneity label:** binary, 1 if σ_[Fe/H] < 0.05 dex (homogeneous), 0 if σ_[Fe/H] > 0.05 dex (heterogeneous)
- **Continuous dispersion target:** the actual σ_[Fe/H], so the GNN can also be trained for regression

---

## Phase 5: Build the graph structure for each cluster

**Time: 1-2 days**

### How to build one graph

Each cluster becomes one graph. Every star in the cluster with XP data is a node. Edges encode which stars are "neighbors."

**Node features (per star):**
- 110 XP coefficients (or fewer if using Andrae metallicities: [M/H], [M/H]_err, Teff, logg)
- Gaia astrometry: parallax, pmRA, pmDec, G magnitude, BP-RP color
- Total: ~115-120 features per node if using raw XP, or ~8 features if using Andrae-derived parameters

**Edge construction:**
Build a k-nearest-neighbor graph in 6D phase space: (RA, Dec, pmRA, pmDec, parallax, G_mag). Normalize each dimension to zero mean and unit variance before computing distances. Use k=10 neighbors. This means each star is connected to its 10 nearest neighbors in phase space.

```python
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

# For one cluster
node_features = torch.tensor(xp_features, dtype=torch.float)  # (N_stars, 115)
positions_6d = torch.tensor(phase_space_coords, dtype=torch.float)  # (N_stars, 6)

edge_index = knn_graph(positions_6d, k=10, loop=False)

graph = Data(
    x=node_features,
    edge_index=edge_index,
    y_metallicity=torch.tensor(apogee_mean_feh, dtype=torch.float),  # cluster-level label
    y_homogeneous=torch.tensor(is_homogeneous, dtype=torch.long),   # cluster-level label
    y_dispersion=torch.tensor(apogee_sigma_feh, dtype=torch.float), # cluster-level label
)
```

### Handling variable-size clusters

Clusters range from ~20 to ~5,000 members. Graph neural networks handle variable-size graphs natively — you do not need to pad or truncate. PyTorch Geometric's `DataLoader` batches multiple graphs into a single disconnected graph internally.

### Save all graphs

```python
torch.save(all_graphs, "project/graphs/all_cluster_graphs.pt")
```

---

## Phase 6: Build the graph neural network

**Time: 3-5 days**

### Architecture

The network has three stages:

**Stage 1: Node encoder (per-star)**
A small fully-connected network that maps the raw node features (115 dimensions) to a 64-dimensional embedding:
```
Linear(115 → 128) → ReLU → Linear(128 → 64) → ReLU
```

**Stage 2: Message-passing layers (graph-level)**
3 layers of GraphSAGE or GATv2 convolutions. Each layer updates each node's representation by aggregating information from its neighbors:

```
For each layer i = 1, 2, 3:
    For each node v:
        neighbor_messages = AGGREGATE({h_u for u in neighbors(v)})
        h_v_new = UPDATE(h_v, neighbor_messages)
```

In standard GraphSAGE, `AGGREGATE` is a mean over neighbor embeddings and `UPDATE` is a linear layer + ReLU. In GATv2, `AGGREGATE` uses learned attention weights.

**The novel variant — UncertaintySAGEConv:** Each node carries a measurement uncertainty σ_i (from the XP pipeline). During aggregation, weight neighbors by 1/σ_i² instead of equal weight. This is a precision-weighted mean — the same principle used in classical astronomy for combining heteroskedastic measurements, now embedded inside a learned message-passing layer.

Implementation with PyTorch Geometric:
```python
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

class UncertaintySAGEConv(MessagePassing):
    """GraphSAGE convolution that weights neighbors by inverse measurement variance."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=None)  # custom aggregation
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, uncertainty):
        # uncertainty: per-node measurement error, shape (N,)
        # Compute inverse-variance weights
        inv_var = 1.0 / (uncertainty ** 2 + 1e-8)  # (N,)
        return self.propagate(edge_index, x=x, inv_var=inv_var)

    def message(self, x_j, inv_var_j):
        # x_j: neighbor features, inv_var_j: neighbor inverse variances
        return x_j * inv_var_j.unsqueeze(-1)  # weight each neighbor

    def aggregate(self, messages, index, inv_var_j):
        # Precision-weighted mean: sum(x_j / σ_j²) / sum(1 / σ_j²)
        numerator = scatter(messages, index, dim=0, reduce='sum')
        denominator = scatter(inv_var_j.unsqueeze(-1), index, dim=0, reduce='sum')
        return numerator / (denominator + 1e-8)

    def update(self, aggr_out, x):
        return torch.relu(self.lin_self(x) + self.lin_neigh(aggr_out))


class ClusterGNN(torch.nn.Module):
    def __init__(self, in_dim=115, hidden_dim=64, num_layers=3, use_uncertainty=True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.ReLU()
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if use_uncertainty:
                self.convs.append(UncertaintySAGEConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(0.2)  # for MC-Dropout uncertainty at inference

        # Cluster-level output heads
        self.metallicity_head = nn.Linear(hidden_dim, 1)
        self.dispersion_head = nn.Linear(hidden_dim, 1)
        self.homogeneity_head = nn.Linear(hidden_dim, 2)

    def forward(self, data, mc_dropout=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        uncertainty = data.uncertainty  # per-node error bars, shape (N,)

        x = self.encoder(x)

        for conv in self.convs:
            if self.use_uncertainty:
                x = conv(x, edge_index, uncertainty)
            else:
                x = conv(x, edge_index)
            x = torch.relu(x)
            if mc_dropout or self.training:
                x = self.dropout(x)

        cluster_embedding = global_mean_pool(x, batch)

        met = self.metallicity_head(cluster_embedding)
        disp = self.dispersion_head(cluster_embedding)
        homo = self.homogeneity_head(cluster_embedding)

        return met, disp, homo
```

The ablation comparing `UncertaintySAGEConv` vs standard `SAGEConv` (setting `use_uncertainty=False`) is MANDATORY. If the uncertainty-aware version outperforms, it validates the architectural contribution. If it does not, report this honestly — it would mean the GNN already learns to ignore noisy nodes from the features alone, which is also interesting.

**MC-Dropout inference for calibrated uncertainty:** At test time, run the model 50 times with `mc_dropout=True`. The mean of the 50 homogeneity logits is the prediction; the standard deviation is the uncertainty. Report calibration curves (predicted confidence vs actual accuracy).

### Key architectural decision: GraphSAGE vs GATv2

Try both and compare. GATv2 adds attention (the network learns which neighbors to weight more heavily) but is slightly slower. If the results are similar, use GraphSAGE for simplicity and report both.

### Loss function

The total loss combines three terms:
```python
loss = (
    mse_loss(predicted_metallicity, true_metallicity)          # regression
    + 0.5 * mse_loss(predicted_dispersion, true_dispersion)    # regression
    + 0.5 * cross_entropy_loss(predicted_homogeneity, true_label)  # classification
)
```

The weights (0.5) can be tuned. Start with equal weights and adjust if one task dominates.

---

## Phase 7: Training and evaluation

**Time: 1-2 weeks**

### Train/validation/test split

Split the ~300 clusters with APOGEE ground truth into:
- 70% training (~210 clusters)
- 15% validation (~45 clusters)
- 15% test (~45 clusters)

Split by cluster, not by star. All stars in a cluster go to the same split.

### Training loop

```python
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)

model = ClusterGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    for batch in train_loader:
        met, disp, homo = model(batch)
        loss = compute_loss(met, disp, homo, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch}: val RMSE(met)={val_metrics['met_rmse']:.3f}, "
              f"val acc(homo)={val_metrics['homo_acc']:.3f}")
```

### Evaluation metrics

On the test set, report:
- **Metallicity RMSE:** difference between GNN-predicted [Fe/H] and APOGEE [Fe/H], in dex
- **Dispersion RMSE:** difference between GNN-predicted σ_[Fe/H] and APOGEE σ_[Fe/H]
- **Homogeneity accuracy:** fraction of clusters correctly classified as homogeneous/heterogeneous
- **Homogeneity precision and recall:** to check for bias toward one class
- **Confusion matrix:** 2×2 (homogeneous vs heterogeneous, predicted vs true)

### The baseline to compare against

**Weighted mean + chi-squared test (no ML):**

For each cluster, compute the weighted mean XP metallicity and the chi-squared statistic:
```python
weights = 1 / errors**2
mean_feh = np.average(feh_values, weights=weights)
chi2 = np.sum(weights * (feh_values - mean_feh)**2)
p_value = scipy.stats.chi2.sf(chi2, df=len(feh_values)-1)
is_homogeneous = (p_value > 0.05)  # not significantly different from uniform
```

Run this baseline on the same test set and report the same metrics. The GNN should beat this baseline because it uses the full XP spectrum (not just the derived metallicity) and it can learn non-linear relationships.

---

## Phase 8: Apply to the full catalog

**Time: 2-3 days**

Once the model is trained and validated:

1. Build graphs for ALL ~7,000 clusters (including those without APOGEE data)
2. Run the trained GNN on each cluster
3. Store the predictions: (cluster_id, predicted_metallicity, predicted_dispersion, predicted_homogeneity_score)
4. The homogeneity score is a probability between 0 and 1. Classify as homogeneous if > 0.5.

### Create the main science output

A catalog of all ~7,000 clusters with:
- Hunt-Reffert properties (age, position, number of members)
- GNN-predicted mean metallicity
- GNN-predicted dispersion
- GNN-predicted homogeneity probability
- For clusters with APOGEE: the ground-truth values for comparison

Release this catalog as a supplementary table with the paper.

### Science analysis

With the full catalog:
- Plot the fraction of homogeneous clusters vs cluster age. Theory predicts that older clusters should be more homogeneous because they have had time for radial mixing. Is this what you see?
- Plot predicted metallicity vs Galactic position (galactocentric radius). There should be a metallicity gradient — clusters closer to the center of the galaxy should be more metal-rich.
- Identify the most heterogeneous clusters in the sample. Are these known to have unusual properties (merger remnants, tidal tails, contamination)?

---

## Phase 9: Ablation experiments

**Time: 3-5 days**

### Ablation 1: Graph connectivity

Compare k=5, k=10, k=20, and a fully-connected graph (all pairs). Report homogeneity accuracy vs k. If performance is similar across k, the network is robust. If it peaks at a specific k, discuss why.

### Ablation 2: Number of message-passing layers

Compare 1, 2, 3, 4 layers. More layers let information propagate further across the graph but risk over-smoothing (all node embeddings converge to the same vector). Report metrics vs depth.

### Ablation 3: Node features

Compare three feature sets:
- Raw XP coefficients (110 features per star)
- Andrae-derived parameters ([M/H], [M/H]_err, Teff, logg — 4 features per star)
- Gaia photometry only (G, BP-RP, parallax — 3 features per star)

This tells you whether the full XP spectrum adds value over the derived parameters, and whether the network can work with minimal features.

### Ablation 4: GraphSAGE vs GATv2

Run both architectures with the same hyperparameters and compare. Report whether attention helps.

---

## Phase 10: Verification and sanity checks

**Time: 2-3 days**

1. **Overfit check:** Train on 5 clusters only (with APOGEE ground truth) for 1000 epochs. The training loss should drop to near zero. If it does not, there is a bug in the model or data pipeline.

2. **Random label check:** Shuffle the homogeneity labels randomly and retrain. The model should achieve ~50% accuracy (random chance). If it still performs well on shuffled labels, the model is memorizing cluster structure rather than learning chemistry.

3. **Single-star check:** For clusters with only 1-2 APOGEE members, the ground truth is unreliable. Report test metrics separately for clusters with 3-5 APOGEE members and for clusters with 6+ APOGEE members. Performance should be better for the latter.

4. **Known-cluster check:** M67 and the Hyades are two of the best-studied open clusters. Check that the GNN's metallicity prediction for these clusters matches published literature values ([Fe/H] ≈ 0.0 for M67, [Fe/H] ≈ +0.13 for Hyades).

---

## Phase 11: Paper writing plan

### Full A&A / MNRAS version (12-15 pages)

**Title:** "Testing Open-Cluster Chemical Homogeneity with Graph Neural Networks and Gaia XP Spectra"

**Sections:**

1. **Abstract** (200-250 words)
   - Problem: testing chemical homogeneity in open clusters normally requires expensive high-resolution spectroscopy
   - Method: a graph neural network that combines noisy Gaia XP per-star features via learned message passing within a phase-space-defined cluster graph
   - Training: supervised on ~300 clusters with APOGEE ground truth
   - Result: the GNN outperforms the weighted-mean chi-squared baseline by X% on homogeneity classification; applied to 7,000+ clusters to produce the first large-scale homogeneity catalog from XP spectra alone
   - Released: catalog of cluster metallicities and homogeneity scores

2. **Introduction** (1.5 pages)
   - Open clusters as probes of star formation: the homogeneity question
   - Current methods for measuring cluster chemistry (high-resolution spectroscopy, limitations)
   - Gaia XP spectra: 220 million stars at low resolution, noisy per-star metallicities
   - The problem: per-star noise > intra-cluster chemical spread
   - Why GNNs: message passing is a natural way to combine noisy measurements across a structured group
   - This paper's contribution

3. **Data** (2 pages)
   - 3.1 Hunt & Reffert cluster catalog: sample, member selection, membership probabilities
   - 3.2 Gaia DR3 XP spectra: coefficient extraction, coverage rate, quality cuts
   - 3.3 APOGEE DR17 ground truth: cross-match procedure, number of clusters with ≥3 APOGEE members, definition of homogeneity label
   - Table: properties of the labeled cluster sample (age range, metallicity range, size range)

4. **Methods** (2.5 pages)
   - 4.1 Graph construction: node features, edge construction (k-NN in 6D phase space), normalization
   - 4.2 GNN architecture: node encoder, message-passing layers (GraphSAGE and GATv2), global pooling, output heads
   - Architecture diagram (figure)
   - 4.3 Loss function: multi-task (metallicity regression + dispersion regression + homogeneity classification)
   - 4.4 Training procedure: optimizer, learning rate, early stopping, train/val/test split
   - 4.5 Baseline: weighted mean + chi-squared test

5. **Results** (3 pages)
   - 5.1 Test-set performance: RMSE, accuracy, confusion matrix (table + figure)
   - 5.2 GNN vs baseline comparison
   - 5.3 Ablation: graph connectivity, depth, features, architecture
   - 5.4 Full-catalog predictions: metallicity distribution, homogeneity fraction vs age
   - 5.5 Metallicity gradient with Galactic position
   - 5.6 Notable heterogeneous clusters: any surprises?

6. **Discussion** (1.5 pages)
   - What the GNN learns: are there interpretable attention patterns?
   - Limitations: small labeled sample, APOGEE selection biases, XP noise floor
   - Comparison to previous cluster chemistry studies (Niu et al. 2025, etc.)
   - Implications for future surveys (SDSS-V, 4MOST, WEAVE)

7. **Conclusions** (0.5 page)
   - Summary of the method and main result
   - Catalog released as supplementary material
   - Future work: extend to individual abundance ratios ([Mg/Fe], [Si/Fe]) not just [Fe/H]

8. **References** (~35-45 references)

9. **Appendix**
   - Full catalog description (column definitions for the released table)
   - Additional ablation plots
   - List of clusters classified as heterogeneous (for follow-up)

### ML4PS / ML4Astro workshop version (4 pages)

For the workshop paper, compress Sections 1-6 into 4 pages. Focus on the method (Sections 3-4), the main comparison result (Section 5.1-5.2), and one interesting science finding (Section 5.4 or 5.6). Omit the full ablation and the catalog release (those go in the journal paper).

---

## Timeline

| Week | Activity |
| --- | --- |
| 1 | Environment setup, download Hunt-Reffert catalog, explore cluster data |
| 2 | Cross-match cluster members with Gaia XP (write ADQL queries, run GaiaXPy) |
| 3 | Cross-match with APOGEE DR17, define labeled sample, compute ground-truth dispersions |
| 4 | Build graph structures for all clusters, implement baseline (weighted mean + chi-squared) |
| 5 | Implement GNN architecture (GraphSAGE + GATv2 variants), train on labeled sample |
| 6 | Evaluate on test set, run ablation experiments |
| 7 | Apply trained model to full catalog (~7,000 clusters), generate science plots |
| 8 | Sanity checks (M67, Hyades, random-label test, overfit test) |
| 9 | Write the paper: methods + results first, then intro + discussion |
| 10 | Revise, advisor feedback, generate camera-ready figures, submit |
