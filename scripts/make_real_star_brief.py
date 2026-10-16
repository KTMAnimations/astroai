#!/usr/bin/env python3
"""Create the four-page observational follow-up handout."""
import json
from pathlib import Path

import matplotlib
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from astrogyro.data import ROOT


def main():
    root = ROOT/"project/results/real_stars_v2"
    result = json.loads((root/"summary.json").read_text())
    external = json.loads((root/"external_summary.json").read_text())
    primary = result["paired_intervals"]["mixture_quadratic_minus_gaussian_power_nll"]
    output = ROOT/"output/pdf/real_star_research_brief.pdf"
    fontdir = Path(matplotlib.get_data_path())/"fonts/ttf"
    for name, filename in [("Study", "DejaVuSans.ttf"), ("StudyBold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fontdir/filename)))
    pdfmetrics.registerFontFamily("Study", normal="Study", bold="StudyBold", italic="Study", boldItalic="StudyBold")
    W, H = landscape(letter);M = 36
    ink = colors.HexColor("#25344b")
    body = ParagraphStyle("body", fontName="Study", fontSize=10.5, leading=15, textColor=ink)
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#677486"))
    c = canvas.Canvas(str(output), pagesize=(W, H))
    c.setTitle("Predicting real stellar rotation and cluster ages")
    c.setAuthor("AstroAI research project")
    def para(text, y, style=body, x=M, width=W-2*M):
        p = Paragraph(text, style);_, height = p.wrap(width, H);p.drawOn(c, x, y-height);return y-height
    def heading(text):
        c.setFont("StudyBold", 23);c.setFillColor(ink);c.drawString(M, 555, text)
    def picture(name, top, width=720):
        path = ROOT/"project/figures/real_stars_v2"/f"{name}.png"
        with Image.open(path) as im:height = width*im.height/im.width
        c.drawImage(str(path), (W-width)/2, top-height, width=width, height=height, mask="auto")
        return top-height
    def finish(page):
        c.setStrokeColor(colors.HexColor("#dce2ea"));c.line(M, 32, W-M, 32)
        c.setFont("Study", 8);c.setFillColor(colors.HexColor("#677486"))
        c.drawString(M, 20, "ASTROAI  |  Observational follow-up  |  15 October 2026  |  Protocol and complete tables in the repository")
        c.drawRightString(W-M, 20, f"{page} / 4");c.showPage()

    heading("Predicting real stellar rotation and cluster ages")
    y = para("721 observed stars. 16 withheld clusters. 25 additional external stars.", 530)
    y = para("A single equation describes an average, but stars of similar age and color can rotate at different rates. "
             "This study fits a narrow slow-rotation component and a broader, faster component. Six model families "
             "are compared on the same 48 discovery/validation/test splits. Every test cluster is excluded from fitting and selection.", y-12)
    bottom = picture("distribution_scores", y-13)
    y = para(f"<b>Primary result:</b> the quadratic mixture improves negative log likelihood by "
             f"{abs(primary['mean_difference']):.3f} nats versus a Gaussian power law; paired difference "
             f"{primary['mean_difference']:+.3f}, descriptive 95% interval [{primary['ci95_low']:+.3f}, {primary['ci95_high']:+.3f}]. "
             "It improves this score in 15 of 16 cluster averages. CRPS also improves, by about 10.3%.", bottom-12)
    para("Lower values are better for both proper distribution scores. These results show better probability predictions. "
         "Mean-log-period RMSE does not improve across the primary sample. Intervals resample clusters and do not include all "
         "shared-training dependence or systematic age error.", y-10, small)
    finish(1)

    heading("1. Predict the observed spread")
    y = para("Prediction intervals should contain the stated fraction of new observations while remaining informative. "
             "The comparison uses every selected star, including fast rotators and difficult clusters.", 530)
    bottom = picture("coverage", y-17)
    y = para("The quadratic mixture's 68% and 90% intervals contain <b>68.4% and 90.0%</b> of held-out observations "
             "after equal weighting by cluster. The power-law Gaussian contains 80.7% and 92.1%. "
             "The mixture's mean 68% interval is narrower: <b>0.291 versus 0.458 dex</b>.", bottom-15)
    para("The intervals represent population variation in rotation period, not uncertainty on the mean curve or on an "
         "individual star's age. Good average coverage does not establish calibration in every cluster or color range. "
         "A 0.1-dex difference corresponds to a factor of approximately 1.26 in period.", y-12, small)
    finish(2)

    heading("2. Recover the age of an unseen cluster")
    y = para("Use the withheld cluster's periods and colors to maximize the fitted likelihood over 80-5000 Myr. "
             "Its catalogue age is used only afterward for evaluation. Every cluster remains in the comparison.", 530)
    bottom = picture("age_recovery", y-12, width=690)
    y = para("Median absolute fractional age error changes from <b>18.3%</b> for the power law to <b>12.5%</b> for the "
             "quadratic mixture and <b>9.9%</b> for the model family chosen on validation data. "
             "Three inferred log ages are averaged within each test cluster.", bottom-10)
    para("Some clusters still have large errors; one selected fit reaches the age-grid boundary. The paired interval on "
         "the change in mean absolute log-age error spans zero. These are cluster point estimates with uncertain reference "
         "ages, not a claim of 10%-accurate individual-star ages.", y-10, small)
    finish(3)

    heading("3. Test additional published observations")
    y = para("Twenty-five Gruner et al. (2023) M67 stars are absent from the entire original catalogue. "
             "All M67 stars are excluded from the three frozen models used to predict this external sample.", 530)
    bottom = picture("external_m67", y-8, width=540)
    y = para("At a nominal age of 4.0 Gyr, the quadratic mixture reduces mean-prediction RMSE from "
             "<b>0.1537 to 0.1174 dex</b>, approximately <b>23.6%</b>. The ordinary quadratic Gaussian is competitive "
             "at 0.1249 dex and has lower median absolute error than the mixture. All 25 stars are included.", bottom-9)
    y = para("This is one external cluster sample, not field-star validation. Its single-star selection differs from the "
             "primary population, and its 68% mixture intervals overcover at 92%. The source's age and reddening remain uncertain. "
             "No external errors choose the fitted models.", y-8, small)
    para("<b>Sources:</b> <a href='https://arxiv.org/abs/2412.12244' color='#226cb5'>Van-Lane et al. (2025)</a>; "
         "<a href='https://arxiv.org/abs/2303.08830' color='#226cb5'>Bouma et al. (2023)</a>; "
         "<a href='https://arxiv.org/abs/2305.16997' color='#226cb5'>Gruner et al. (2023)</a>. "
         "Code, candidate fits, predictions, data hashes and tests are saved. AI assistance supported research, implementation and writing.", y-8, small)
    finish(4)
    c.save();print(output)


if __name__ == "__main__": main()
