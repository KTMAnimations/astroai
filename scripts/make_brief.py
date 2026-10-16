#!/usr/bin/env python3
"""Build a four-page, landscape research brief from the completed results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from astrogyro.data import ROOT

INK = colors.HexColor("#25344b")
BLUE = colors.HexColor("#226cb5")
GRAY = colors.HexColor("#677486")
W,H = landscape(letter)
M = 36


def main():
    result=ROOT/"project/results/research_v1"
    figures=ROOT/"project/figures/research_v1"
    target=ROOT/"output/pdf/stellar_rotation_research_brief.pdf"
    target.parent.mkdir(parents=True,exist_ok=True)
    summary=json.loads((result/"observational_summary.json").read_text())
    diagnostics=json.loads((result/"selection_diagnostics.json").read_text())
    counts=pd.read_csv(result/"recovery_counts.csv")
    example=counts[(counts.truth=="interaction")&(counts.design=="catalog")&(counts.noise_dex==.05)].iloc[0]
    scores=summary["mean_cluster_rmse_dex"]
    interval=summary["paired_cluster_intervals"]["pysr_validation_minus_power_law"]
    fonts=Path(matplotlib.get_data_path())/"fonts/ttf"
    pdfmetrics.registerFont(TTFont("Study",str(fonts/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("StudyBold",str(fonts/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("Study",normal="Study",bold="StudyBold",italic="Study",boldItalic="StudyBold")
    c=canvas.Canvas(str(target),pagesize=(W,H))
    c.setTitle("Can an equation search learn stellar rotation?")
    c.setAuthor("AstroAI research project")
    body=ParagraphStyle("body",fontName="Study",fontSize=10.5,leading=15,textColor=INK,alignment=TA_LEFT)
    small=ParagraphStyle("small",parent=body,fontSize=9,leading=12,textColor=GRAY)
    def para(text,x,y,width,style=body):
        p=Paragraph(text,style);_,height=p.wrap(width,H);p.drawOn(c,x,y-height);return y-height
    def heading(text,y=555):
        c.setFillColor(INK);c.setFont("StudyBold",23);c.drawString(M,y,text)
    def subheading(text,x,y):
        c.setFillColor(BLUE);c.setFont("StudyBold",12);c.drawString(x,y,text);return y-13
    def picture(name,x,y_top,width):
        path=figures/f"{name}.png"
        with Image.open(path) as im:height=width*im.height/im.width
        c.drawImage(str(path),x,y_top-height,width=width,height=height,mask="auto")
        return y_top-height
    def footer(page):
        c.setStrokeColor(colors.HexColor("#dce2ea"));c.line(M,32,W-M,32)
        c.setFillColor(GRAY);c.setFont("Study",8)
        c.drawString(M,20,"ASTROAI  |  15 October 2026  |  Observational data: Van-Lane et al. (2025)")
        c.drawRightString(W-M,20,f"{page} / 4")
    def finish(page):footer(page);c.showPage()

    heading("Can an equation search learn stellar rotation?")
    para("A controlled test of symbolic regression, sampling and independent validation",M,533,W-2*M)
    text=(f"Stars provide a test for automated equation discovery: rotation changes with age, but measured stars show substantial scatter. "
          f"This study ran <b>180 controlled searches</b> and <b>48 cluster-holdout searches</b>. "
          f"For a known nonlinear law with moderate noise and catalogue sampling, validation recovered the function in "
          f"<b>{int(example.pysr_validation)}/10 trials</b>, compared with <b>{int(example.pysr_default)}/10</b> for default selection. "
          f"On real stars, the fitted power law scored {scores['power_law']:.3f} dex and validation-selected symbolic regression scored "
          f"{scores['pysr_validation']:.3f} dex. The next pages show the full comparison and its limits.")
    y=para(text,M,503,W-2*M)
    bottom=picture("sample",M,y-13,W-2*M)
    para("Sample: 721 stars in 16 clusters, with age at least 100 Myr, 0.65 &lt; Gaia BP-RP &lt; 1.5, "
         "membership score at least 0.8, quality flags passed and at least 10 selected stars per cluster. "
         "No upper rotation-period cut or model-residual cut was applied.",M,bottom-10,W-2*M,small)
    finish(1)

    heading("1. Test the search when the answer is known")
    para("Three known functions generate synthetic rotation data. Each cell contains ten independent trials. "
         "Both selectors choose from the same discovered equations.",M,530,W-2*M)
    bottom=picture("recovery_counts",M,487,W-2*M)
    y=para("<b>Definition of recovery:</b> less than 0.02 dex error against the exact function on a separate grid "
           "across the age/color domain. This is about a 4.7% period factor. It tests predictions, not literal algebraic identity. "
           "The grid never influences model selection.",M,bottom-12,W-2*M)
    para(f"Across all 180 trials, the default selector missed recovery in {diagnostics['default_selection_misses_despite_accurate_candidate']} "
         f"trials even though an accurate candidate was present. In {diagnostics['no_candidate_recovers']} trials, no candidate met the threshold. "
         "Overall recovery was 76/180 for default selection, 157/180 for validation and 164/180 for the quadratic baseline. "
         "These totals depend on the chosen functions. Two invalid validation-selected functions remain failures. "
         "Candidate availability is a truth-based diagnostic, not a selector available for real observations.",M,y-10,W-2*M,small)
    finish(2)

    heading("2. Predict clusters that were withheld")
    para("Leave one cluster out for testing. Select three other clusters for validation, and fit every method "
         "on the remaining discovery clusters. Repeat with three split/search seeds.",M,530,W-2*M)
    bottom=picture("model_comparison",M,484,W-2*M)
    y=para(f"Validation-selected symbolic regression minus the fitted power law: <b>{interval['mean_difference']:+.4f} dex</b>, "
           f"with a descriptive paired 95% interval of <b>[{interval['ci95_low']:+.4f}, {interval['ci95_high']:+.4f}] dex</b>. "
           "Negative differences favor symbolic regression. Scores average seeds within a cluster first, then give every cluster equal weight.",M,bottom-13,W-2*M)
    para("Error is measured in log10(rotation period). For scale, 0.1 dex corresponds to a factor of 1.26. "
         "The intervals resample the 16 observed clusters; they do not include systematic errors in cluster ages, "
         "all effects of overlapping training sets, or the selection of the catalogue. An interval containing zero does not prove equivalence.",M,y-10,W-2*M,small)
    finish(3)

    heading("3. Interpret the result within its scope")
    col=(W-2*M-30)/2
    y=subheading("What was tested",M,517)
    y=para("The controlled experiment measures whether a search and selection procedure can recover a known function "
           "under different sampling and noise conditions. The observational experiment measures rotation-period "
           "prediction for a defined stellar sample at supplied cluster ages.",M,y,col)
    y=subheading("Limits",M,y-27)
    y=para("<b>Sixteen age groups.</b> Hundreds of stars do not provide hundreds of independent age calibrations. "
           "The oldest cluster, M67, is an extrapolation test.<br/><br/>"
           "<b>Selected population.</b> Membership and color cuts define the sample. They do not remove all binaries, "
           "contaminants or fast rotators. Sensitivity tables test alternative training cuts on the same test stars.<br/><br/>"
           "<b>Finite search budget.</b> Results describe the specified operators, transformations and search budget. "
           "Another configuration may behave differently.<br/><br/>"
           "<b>Scientific interpretation.</b> Stable equations and good prediction alone do not establish a new physical law, "
           "a magnetic-braking transition or calibrated individual-star ages.",M,y,col)
    x=M+col+30
    y=subheading("Reproduce and inspect",x,517)
    y=para("The repository contains the protocol, source-data checksum, exact dependency versions, random seeds, "
           "all candidate equations, discovery/validation/test cluster lists and per-star test predictions.<br/><br/>"
           "<font name='Courier' size='9'>bash scripts/setup_environment.sh<br/>"
           ".venv/bin/python -m pytest<br/>"
           ".venv/bin/python scripts/run_study.py</font><br/><br/>"
           "The README links the full tables, all-cluster error figure, M67 holdout plot and training-sample sensitivity analysis. "
           "The research log records pilots and follow-up checks.",x,y,col)
    y=para("<b>Fresh-seed follow-up:</b> 30 additional high-noise, catalogue-sampled searches gave 21/30 recoveries "
           "for validation and 0/30 for default selection. No invalid functions occurred, so a separate domain guard "
           "changed no choices. Its added benefit remains unconfirmed on fresh trials.",x,y-12,col,small)
    y=subheading("Sources",x,y-27)
    y=para("Van-Lane et al. (2025), <a href='https://arxiv.org/abs/2412.12244' color='#226cb5'>ChronoFlow</a>: "
           "catalogue and probabilistic rotation model.<br/><br/>"
           "Cranmer (2023), <a href='https://arxiv.org/abs/2305.01582' color='#226cb5'>PySR</a>: symbolic-regression software.<br/><br/>"
           "Barnes et al. (2016), <a href='https://arxiv.org/abs/1603.09179' color='#226cb5'>M67 rotation</a>; "
           "Dungee et al. (2022), <a href='https://arxiv.org/abs/2211.01377' color='#226cb5'>cool stars in M67</a>.<br/><br/>"
           "This revision used AI assistance for code, experiment design and writing. The measurements come from published work. "
           "A student presenter should be able to explain and independently inspect each step.",x,y,col,small)
    finish(4)
    c.save();print(target)


if __name__=="__main__":main()
