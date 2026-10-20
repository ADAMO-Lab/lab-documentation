"""Draw one example figure per breast imaging dataset for datasets/breast-imaging/<slug>.qmd.

Openly licensed datasets (M-SYNTH, CMMD, TOMPEI-CMMD, CBIS-DDSM, CDD-CESM, DMID) show one real breast with its
label overlays and a panel of that breast's labels. EMBED, VinDr-Mammo and INbreast forbid redistribution, so they
get a drawn schematic of the label types instead. Nothing from those three is read.

Usage: python scripts/make_dataset_figures.py [--root /data/1TB/breasts] [--only cmmd,dmid]
Needs: scripts/requirements-figures.txt. Writes datasets/breast-imaging/figures/<slug>.png. Run locally; CI has no data.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "datasets" / "breast-imaging" / "figures"
LESION = "#ff3b30"
BENIGN = "#34c759"
BOX = "#ffcc00"
DISPLAY_H = 1100  # px; images are downsampled to this height before drawing


# ------------------------------------------------------------------ image helpers
def read_dicom(path):
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    if a.ndim == 3:  # RGB secondary capture with grey in every channel
        a = a[..., 0]
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    return a


def window(a):
    """Scale to 0-1 over the 1st-99.5th percentile of the breast (pixels above 5% of the 99.9th percentile)."""
    fg = a[a > 0.05 * np.percentile(a, 99.9)]
    lo, hi = np.percentile(fg if fg.size else a, [1, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def shrink(a):
    """Downsample to DISPLAY_H rows. Returns the image and the scale applied to coordinates."""
    s = DISPLAY_H / a.shape[0]
    if s >= 1:
        return a, 1.0
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((round(a.shape[1] * s), DISPLAY_H), Image.LANCZOS)
    return np.asarray(im) / 255.0, s


def mask_outline(ax, mask, s, color, label=None):
    m = np.asarray(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
        (round(mask.shape[1] * s), round(mask.shape[0] * s)), Image.NEAREST)) > 0
    ax.contour(m, levels=[0.5], colors=[color], linewidths=1.6)
    if label:
        ys, xs = np.nonzero(m)
        ax.text(xs.max() + 8, ys.min(), label, color=color, fontsize=8, va="top", fontweight="bold")


# ------------------------------------------------------------------ figure layout
def label_panel(ax, heading, labels):
    ax.set_axis_off()
    ax.text(0, 1, heading, fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)
    y = 0.92
    for k, v in labels:
        v = str(v)
        ax.text(0, y, k, fontsize=8.5, color="#555", va="top", transform=ax.transAxes)
        ax.text(0, y - 0.035, v, fontsize=9.5, va="top", transform=ax.transAxes)
        y -= 0.05 + 0.035 * (v.count("\n") + 1)


def figure(panels, labels, title, credit):
    """panels: list of (view title, image 0-1, draw(ax) callback). labels: list of (name, value)."""
    widths = [p[1].shape[1] / p[1].shape[0] for p in panels] + [0.75]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6.2 * sum(widths), 6.6),
                             gridspec_kw={"width_ratios": widths}, facecolor="white")
    for ax, (name, img, draw) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if draw:
            draw(ax)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    label_panel(axes[-1], "Labels for this breast", labels)
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.01, credit, fontsize=7.5, color="#666")
    return fig


def save(fig, slug):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{slug}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{slug}.png")


# ------------------------------------------------------------------ open datasets
def msynth(root):
    case = root / "msynth/images/ld1.0/dose100/r5mm/dense/001"
    info = json.loads((case / "info.json").read_text())
    a = np.asarray(Image.open(case / "image.png")).astype(np.float32)  # rows x cols as in the DICOM
    w = a.shape[1]
    a = np.rot90(a)  # chest wall (row 0) to the left; (r, c) -> (y, x) = (w - 1 - c, r)
    r, c = info["lesion_rc"]
    y, x = w - 1 - c, r
    img, s = shrink(window(a))
    rad = info["lesion_radius_mm"] / info["pixel_spacing_mm"] * s

    def draw(ax):
        ax.add_patch(Circle((x * s, y * s), rad * 1.6, fill=False, ec=LESION, lw=1.6))
        ax.text(x * s + rad * 1.8, y * s, "lesion", color=LESION, fontsize=8, fontweight="bold")

    labels = [("Breast density", info["breast_density"]), ("Lesion", "spiculated mass, present"),
              ("Lesion density", info["lesion_density"]), ("Lesion radius", f'{info["lesion_radius_mm"]} mm'),
              ("Dose", f'{info["dose_pct"]}% of clinical'), ("Lesion position", "known (circle)"),
              ("Lesion mask", "known (per-pixel segmentation)"), ("Pixel size", f'{info["pixel_spacing_mm"]} mm')]
    save(figure([("Simulated DM", img, draw)], labels, f'M-SYNTH: phantom {info["phantom_id"]}',
                "Sizikova et al. 2023, M-SYNTH (FDA VICTRE). CC0."), "msynth")


def cmmd(root):
    clin = pd.read_excel(root / "cmmd/metadata/CMMD_clinicaldata_revision.xlsx")
    row = clin[(clin.classification == "Malignant") & clin.subtype.notna()].iloc[0]
    man = pd.read_csv(root / "cmmd/manifest.csv")
    series = root / "cmmd" / man[man.PatientID == row.ID1].iloc[0].path
    panels = []
    for f in sorted(series.glob("*.dcm")):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if getattr(ds, "ImageLaterality", row.LeftRight) != row.LeftRight:
            continue
        meaning = " ".join(i.CodeMeaning for i in ds.get("ViewCodeSequence", []))
        view = {"cranio-caudal": "CC", "medio-lateral oblique": "MLO"}.get(meaning, meaning)
        panels.append((f"{row.LeftRight} {view}", shrink(window(read_dicom(f)))[0], None))
    labels = [("Side", row.LeftRight), ("Age", int(row.Age)), ("Abnormality", row.abnormality),
              ("Pathology (biopsy)", row.classification), ("Molecular subtype", row.subtype),
              ("Lesion location", "not given (see TOMPEI-CMMD)")]
    save(figure(panels, labels, f"CMMD: patient {row.ID1}",
                "Cui et al. 2021, CMMD, TCIA doi:10.7937/tcia.eqde-4b16. CC BY 4.0."), "cmmd")


def tompei_sheet(root):
    """Imaging Diagnosis Details sheet: two header rows (group, field) -> 'group / field' columns."""
    raw = pd.read_excel(root / "tompei-cmmd/TOMPEI-CMMD_clinical_data_v01_20250121.xlsx",
                        sheet_name="Imaging Diagnosis Details Sheet", header=None)
    group = raw.iloc[0].ffill()
    cols = [f"{g} / {f}".replace("\n", " ") if isinstance(f, str) else str(g).replace("\n", " ")
            for g, f in zip(group, raw.iloc[1])]
    return pd.DataFrame(raw.iloc[2:].values, columns=cols)


def tompei(root):
    recs = json.loads((root / "tompei-cmmd/derived_records.json").read_text())
    sheet = tompei_sheet(root)
    shape_col = next(c for c in sheet.columns if c.startswith("Mass / Shape"))
    for rec in recs:
        rows = sheet[(sheet.ID == rec["patient_id"]) & (sheet.LeftRight == rec["laterality"])]
        if (len(rec["lesions"]) == 1 and rec["multiplicity"] == "Single" and len(rows)
                and pd.notna(rows.iloc[0][shape_col])):
            break
    row = rows.iloc[0]
    a = read_dicom(root / "cmmd" / rec["image"])
    if rec["flip"]:
        a = a[:, ::-1]
    img, s = shrink(window(a))
    poly = json.loads(rec["lesions"][0]["poly"])
    pts = np.c_[poly["all_points_x"], poly["all_points_y"]] * s

    def draw(ax):
        ax.add_patch(Polygon(pts, closed=True, fill=False, ec=LESION, lw=1.6))
        ax.text(pts[:, 0].max() + 8, pts[:, 1].min(), "lesion outline", color=LESION, fontsize=8, fontweight="bold")

    def get(prefix):
        c = next((c for c in sheet.columns if c.startswith(prefix)), None)
        v = row[c] if c else None
        return v if pd.notna(v) else "-"

    labels = [("Side / view", f'{rec["laterality"]} MLO'), ("Breast density", get("Breast density")),
              ("Mass shape", get("Mass / Shape")), ("Mass margin", get("Mass / Margin")),
              ("Mass density", get("Mass / Density")), ("BI-RADS", get("BI-RADS")),
              ("Pathology (CMMD biopsy)", get("classification")), ("Lesion outline", "polygon, MLO only")]
    save(figure([(f'{rec["laterality"]} MLO', img, draw)], labels, f'TOMPEI-CMMD: patient {rec["patient_id"]}',
                "TOMPEI-CMMD annotations, TCIA doi:10.7937/WEZW-BH22, on CMMD images. CC BY 4.0."), "tompei-cmmd")


def cbis(root):
    meta = pd.read_csv(root / "cbis-ddsm/metadata/mass_case_description_test_set.csv")
    pid, side = meta.iloc[0].patient_id, meta.iloc[0]["left or right breast"]
    rows = meta[(meta.patient_id == pid) & (meta["left or right breast"] == side)]
    panels = []
    for _, r in rows.drop_duplicates("image view").iterrows():
        full = read_dicom(next((root / "cbis-ddsm/dicom" / r["image file path"].split("/")[0]).rglob("*.dcm")))
        # The ROI series holds the cropped patch and the full-size mask; the CSV's file order is unreliable.
        roi_dir = root / "cbis-ddsm/dicom" / r["ROI mask file path"].split("/")[0]
        mask = next(m for m in (read_dicom(p) for p in roi_dir.rglob("*.dcm")) if m.shape == full.shape)
        img, s = shrink(window(full))
        panels.append((f"{side} {r['image view']}", img, lambda ax, m=mask, s=s: mask_outline(ax, m, s, LESION, "mass")))
    r = rows.iloc[0]
    labels = [("Side", side), ("Breast density (ACR)", r.breast_density), ("Abnormality", r["abnormality type"]),
              ("Mass shape", r["mass shape"]), ("Mass margins", r["mass margins"]), ("BI-RADS", r.assessment),
              ("Pathology", r.pathology), ("Subtlety (1-5)", r.subtlety), ("Mass mask", "ROI mask per view")]
    save(figure(panels, labels, f"CBIS-DDSM: patient {pid}",
                "Lee et al. 2017, CBIS-DDSM, TCIA doi:10.7937/K9/TCIA.2016.7O02S9CY. CC BY 3.0."), "cbis-ddsm")


def cdd(root):
    ann = pd.read_excel(root / "cdd-cesm/metadata/Radiology-manual-annotations.xlsx", sheet_name="all")
    seg = pd.read_csv(root / "cdd-cesm/metadata/Radiology_hand_drawn_segmentations_v2.csv")
    seg["image"] = seg["#filename"].str.replace(".jpg", "", regex=False)
    dm = ann[(ann.Type == "DM") & ann.Findings.str.contains("mass", case=False, na=False)
             & ann.Image_name.isin(seg.image)]
    pid, side = dm.iloc[0].Patient_ID, dm.iloc[0].Side
    rows = ann[(ann.Patient_ID == pid) & (ann.Side == side) & (ann.Type == "DM")].sort_values("View")
    panels = []
    for _, r in rows.iterrows():
        a = np.asarray(Image.open(root / f"cdd-cesm/images/low_energy/{r.Image_name}.jpg").convert("L"), np.float32)
        img, s = shrink(window(a))
        polys = [json.loads(x) for x in seg[seg.image == r.Image_name].region_shape_attributes]

        def draw(ax, polys=polys, s=s):
            for p in polys:
                if p.get("name") == "polygon":
                    ax.add_patch(Polygon(np.c_[p["all_points_x"], p["all_points_y"]] * s,
                                         closed=True, fill=False, ec=LESION, lw=1.6))
                elif p.get("name") == "ellipse":
                    ax.add_patch(Ellipse((p["cx"] * s, p["cy"] * s), 2 * p["rx"] * s, 2 * p["ry"] * s,
                                         fill=False, ec=LESION, lw=1.6))
                elif p.get("name") == "circle":
                    ax.add_patch(Circle((p["cx"] * s, p["cy"] * s), p["r"] * s, fill=False, ec=LESION, lw=1.6))

        panels.append((f"{side} {r.View} (low energy)", img, draw))
    r = rows.iloc[0]
    labels = [("Side", side), ("Age", r.Age), ("Breast density (ACR)", r["Breast density (ACR)"]),
              ("BI-RADS", str(r.BIRADS).replace("$", ", ")), ("Findings", str(r.Findings).replace("$", "; ")),
              ("Tags", r.Tags), ("Pathology / follow-up", r["Pathology Classification/ Follow up"]),
              ("Lesion outline", "hand drawn, per view"), ("Also", "radiology report, CESM subtracted image")]
    save(figure(panels, labels, f"CDD-CESM: patient P{pid}",
                "Khaled et al. 2021, CDD-CESM, TCIA doi:10.7937/29kw-ae92. CC BY 4.0."), "cdd-cesm")


def dmid(root):
    recs = json.loads((root / "dmid/derived_records.json").read_text())
    rec = next(r for r in recs if {les["pathology"] for les in r["lesions"]} == {"MALIGNANT", "BENIGN"})
    name = Path(rec["image"]).stem
    a = np.asarray(Image.open(root / "dmid" / rec["image"]).convert("L"), np.float32)
    img, s = shrink(window(a))
    report = (root / f"dmid/raw/rep/Reports/Img{name[3:]}.txt").read_text(errors="replace")
    birads = re.search(r"BIRADS:\s*(\S+)", report)
    comp = re.search(r"Breast Composition:\s*\n?(.+)", report)

    def draw(ax):
        for les in rec["lesions"]:
            color = LESION if les["pathology"] == "MALIGNANT" else BENIGN
            mask_outline(ax, np.asarray(Image.open(les["mask"]).convert("L")), s, color, les["pathology"].lower())

    labels = [("Side / view", f'{rec["laterality"]} {rec["view"]}'),
              ("Breast composition", comp.group(1).strip().rstrip(".") if comp else "-"),
              ("BI-RADS (report)", birads.group(1) if birads else "-"),
              ("Lesions", "; ".join(f'{les["dmid_code"]}, {les["pathology"].lower()}' for les in rec["lesions"])),
              ("Lesion mask", "per-pixel, per lesion"), ("Also", "radiology report, lesion center + radius")]
    save(figure([(f'{rec["laterality"]} {rec["view"]}', img, draw)], labels, f"DMID: image {name}",
                "Oza et al. 2024, DMID, figshare 24522883. CC BY 4.0."), "dmid")


# ------------------------------------------------------------------ restricted datasets: schematics only
def breast_shape(ax):
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.add_patch(Polygon(np.c_[0.62 * np.cos(t), 1.1 * np.sin(t)], closed=True, fc="#d9d9d9", ec="#888", lw=1))
    ax.add_patch(Polygon([[0, 1.1], [0.42, 0.75], [0, 0.35]], closed=True, fc="#bdbdbd", ec="none"))  # pectoralis


def schematic(slug, title, marks, labels, note):
    fig, (ax, side) = plt.subplots(1, 2, figsize=(7.5, 5.6), gridspec_kw={"width_ratios": [1, 1.15]},
                                   facecolor="white")
    breast_shape(ax)
    for kind, xy, wh, text in marks:
        if kind == "box":
            ax.add_patch(Rectangle(xy, *wh, fill=False, ec=BOX, lw=1.8))
        else:
            ax.add_patch(Ellipse((xy[0] + wh[0] / 2, xy[1] + wh[1] / 2), *wh, fill=False, ec=LESION, lw=1.8))
        ax.text(xy[0] + wh[0] + 0.03, xy[1] + wh[1], text, fontsize=8, va="top", fontweight="bold",
                color=BOX if kind == "box" else LESION)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Schematic, not a dataset image", fontsize=10)
    label_panel(side, "Labels per breast", labels)
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.01, note, fontsize=7.5, color="#666")
    save(fig, slug)


def embed(_root):
    schematic("embed", "EMBED: label types",
              [("box", (0.2, 0.05), (0.22, 0.25), "radiologist\nlesion box")],
              [("Breast density", "tissueden 1-4 (BI-RADS A-D), per exam"), ("BI-RADS", "assessment per finding (asses)"),
               ("Pathology", "linked biopsy result and severity,\nwhen there is one"), ("Lesion box", "ROI coordinates on 2D images"),
               ("Views", "L/R CC and MLO, plus spot and magnification"), ("Also", "device, age, race, screening or diagnostic")],
              "Images not shown: the EMBED license forbids distributing or publishing any part of the data.")


def vindr(_root):
    schematic("vindr-mammo", "VinDr-Mammo: label types",
              [("box", (0.18, -0.15), (0.2, 0.22), "finding box\n+ category"),
               ("box", (0.32, 0.25), (0.12, 0.12), "finding box")],
              [("Breast density", "ACR A-D, per breast"), ("BI-RADS", "1-5, per breast"),
               ("Findings", "box + category (mass, calcification,\nasymmetry, distortion, ...) + BI-RADS"),
               ("Views", "L/R CC and MLO"), ("Split", "training / test")],
              "Images not shown: the PhysioNet Restricted Health Data License forbids sharing the data.")


def inbreast(_root):
    schematic("inbreast", "INbreast: label types",
              [("outline", (0.2, -0.1), (0.2, 0.2), "lesion\noutline")],
              [("Breast density", "ACR 1-4, per image"), ("BI-RADS", "per image"),
               ("Lesion outlines", "mass, calcification, asymmetry,\ndistortion (Osirix ROI / XML)"),
               ("Pectoral muscle", "boundary outline (MLO)"), ("Also", "medical report per patient")],
              "Images not shown: INbreast is distributed under an access agreement, not an open license.")


MAKERS = {"msynth": msynth, "cmmd": cmmd, "tompei-cmmd": tompei, "cbis-ddsm": cbis, "cdd-cesm": cdd,
          "dmid": dmid, "embed": embed, "vindr-mammo": vindr, "inbreast": inbreast}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/1TB/breasts"))
    ap.add_argument("--only", default=",".join(MAKERS))
    args = ap.parse_args()
    for slug in args.only.split(","):
        MAKERS[slug](args.root)
