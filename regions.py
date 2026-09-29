"""
Convert regions saved by sam_region_picker.py into a DXF (and SVG) for CAD import.

Usage examples:
  python regions_to_dxf.py processed/img1_regions.json --longest-mm 70
  python regions_to_dxf.py processed/img1_regions.json --width-mm 70
  python regions_to_dxf.py processed/img1_regions.json --height-mm 70
  python regions_to_dxf.py processed/img1_regions.json --mm-per-px 0.25
  python regions_to_dxf.py processed/img1_regions.json --ref-px 812 --ref-mm 60

Scale (pick one). The size options apply to the bounding box of ALL regions in the file:
  --longest-mm N   longest side of the bounding box becomes N mm
  --width-mm N     bounding-box width becomes N mm
  --height-mm N    bounding-box height becomes N mm
  --mm-per-px F    millimetres per image pixel
  --ref-px/--ref-mm  a known feature: N pixels long in the image = M mm

Other options:
  --tol-mm T   simplification tolerance in mm (default 0.1)
  --spline     closed splines instead of polylines
  --out PATH   output base name

The shape is placed with its bounding-box lower-left corner at (0, 0).
Install: pip install ezdxf numpy opencv-python
"""
import argparse, json, os
import numpy as np
import cv2
import ezdxf


def simplify(pts, tol):
    c = pts.astype(np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(c, tol, True).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--longest-mm", type=float)
    ap.add_argument("--width-mm", type=float)
    ap.add_argument("--height-mm", type=float)
    ap.add_argument("--mm-per-px", type=float)
    ap.add_argument("--ref-px", type=float)
    ap.add_argument("--ref-mm", type=float)
    ap.add_argument("--tol-mm", type=float, default=0.1)
    ap.add_argument("--spline", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    with open(a.json_path) as f:
        data = json.load(f)

    all_pts = np.array([p for r in data["regions"] for poly in r["polygons"] for p in poly], dtype=np.float64)
    if len(all_pts) == 0:
        raise SystemExit("No polygons found in JSON.")
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    bw, bh = xmax - xmin, ymax - ymin  # bounding box in pixels

    if a.longest_mm: s = a.longest_mm / max(bw, bh)
    elif a.width_mm: s = a.width_mm / bw
    elif a.height_mm: s = a.height_mm / bh
    elif a.mm_per_px: s = a.mm_per_px
    elif a.ref_px and a.ref_mm: s = a.ref_mm / a.ref_px
    else:
        raise SystemExit("Give a scale: --longest-mm, --width-mm, --height-mm, --mm-per-px, or --ref-px + --ref-mm")

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    svg_paths = []

    for r in data["regions"]:
        layer = f"REGION_{r['id']}"
        doc.layers.add(layer)
        for poly in r["polygons"]:
            p = np.array(poly, dtype=np.float64)
            pts = np.column_stack([(p[:, 0] - xmin) * s, (ymax - p[:, 1]) * s])  # flip Y, origin at corner
            pts = simplify(pts, a.tol_mm)
            if len(pts) < 3:
                continue
            if a.spline:
                msp.add_spline([tuple(q) for q in pts] + [tuple(pts[0])], degree=3, dxfattribs={"layer": layer})
            else:
                msp.add_lwpolyline([tuple(q) for q in pts], close=True, dxfattribs={"layer": layer})
            svg_paths.append("M " + " L ".join(f"{x:.3f},{bh*s - y:.3f}" for x, y in pts) + " Z")

    base = a.out or os.path.splitext(a.json_path)[0].replace("_regions", "")
    doc.saveas(base + ".dxf")
    with open(base + ".svg", "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{bw*s:.2f}mm" height="{bh*s:.2f}mm" '
                f'viewBox="0 0 {bw*s:.2f} {bh*s:.2f}">'
                + "".join(f'<path d="{d}" fill="none" stroke="black" stroke-width="0.1"/>' for d in svg_paths)
                + "</svg>")
    print(f"Wrote {base}.dxf and {base}.svg | {len(svg_paths)} contours | "
          f"size {bw*s:.2f} x {bh*s:.2f} mm ({s:.4f} mm/px)")


if __name__ == "__main__":
    main()