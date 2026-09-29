"""
Interactive region selector using SAM2 + OpenCV, with file picker and output folder.

Run:
  python sam_region_picker.py                 # opens a file dialog (multi-select)
  python sam_region_picker.py img1.jpg img2.jpg
  python sam_region_picker.py path/to/folder

Outputs go to a "processed" folder created next to the images (or use --out DIR):
  processed/<name>_regions.json   polygons per region
  processed/<name>_labels.png     label map (uint16, 0 = background)
  processed/<name>_overlay.png    image with region borders drawn

Controls:
  Left click     positive point        Right click   negative point
  Left drag      box prompt            u  undo last prompt
  Enter / Space  accept region         r  reset current prompts
  s              save this image       n  save + next image
  p              save + previous       o  open other files
  q / Esc        save + quit
"""
import argparse, json, os, sys
import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_ID = "facebook/sam2.1-hiera-small"   # tiny / small / base-plus / large
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WIN = "SAM region picker"
PALETTE = [(0, 200, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (200, 120, 0)]


# ---------------- file selection ----------------
def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in IMG_EXT)


def pick_images_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    exts = " ".join(f"*{e}" for e in sorted(IMG_EXT))
    files = filedialog.askopenfilenames(title="Select images",
                                        filetypes=[("Images", exts), ("All files", "*.*")])
    if not files:  # nothing chosen -> offer a whole folder
        folder = filedialog.askdirectory(title="...or select a folder of images")
        files = list_images(folder) if folder else []
    root.destroy()
    return list(files)


def resolve_inputs(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += list_images(a)
        elif os.path.isfile(a):
            paths.append(a)
    return paths


# ---------------- picker ----------------
class Picker:
    def __init__(self, paths, out_dir=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_ID} on {device}...")
        self.predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=device)
        self.out_dir_override = out_dir
        self.set_files(paths)

    def set_files(self, paths):
        self.paths = paths
        self.idx = 0
        base_dir = os.path.dirname(os.path.abspath(paths[0]))
        self.out_dir = self.out_dir_override or os.path.join(base_dir, "processed")
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"{len(paths)} image(s). Output folder: {self.out_dir}")
        self.load_image(0)

    def load_image(self, i):
        self.idx = i
        self.path = self.paths[i]
        self.bgr = cv2.imread(self.path)
        if self.bgr is None:
            raise FileNotFoundError(self.path)
        self.predictor.set_image(cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB))  # heavy step
        self.points, self.labels, self.box, self.mask = [], [], None, None
        self.regions = []
        self._drag_start = self._drag_now = None

    # ---- SAM ----
    def predict(self):
        if not self.points and self.box is None:
            self.mask = None
            return
        kw = {}
        if self.points:
            kw["point_coords"] = np.array(self.points, dtype=np.float32)
            kw["point_labels"] = np.array(self.labels, dtype=np.int32)
        if self.box is not None:
            kw["box"] = np.array(self.box, dtype=np.float32)
        with torch.inference_mode():
            masks, _, _ = self.predictor.predict(multimask_output=False, **kw)
        self.mask = masks[0].astype(np.uint8)

    @staticmethod
    def contours_of(mask, eps_frac=0.002):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in cnts:
            if cv2.contourArea(c) < 20:
                continue
            out.append(cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True))
        return out

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = self._drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start:
            self._drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start:
            x0, y0 = self._drag_start
            if abs(x - x0) > 5 and abs(y - y0) > 5:
                self.box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            else:
                self.points.append([x, y]); self.labels.append(1)
            self._drag_start = self._drag_now = None
            self.predict()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y]); self.labels.append(0)
            self.predict()

    # ---- drawing ----
    def draw_regions(self, img):
        for i, r in enumerate(self.regions):
            cv2.drawContours(img, r["contours"], -1, PALETTE[i % len(PALETTE)], 2)

    def render(self):
        img = self.bgr.copy()
        self.draw_regions(img)
        if self.mask is not None:
            overlay = img.copy()
            overlay[self.mask > 0] = (255, 144, 30)
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            cv2.drawContours(img, self.contours_of(self.mask), -1, (0, 255, 255), 2)
        for (x, y), l in zip(self.points, self.labels):
            cv2.circle(img, (x, y), 5, (0, 255, 0) if l else (0, 0, 255), -1)
            cv2.circle(img, (x, y), 5, (255, 255, 255), 1)
        if self.box:
            cv2.rectangle(img, tuple(self.box[:2]), tuple(self.box[2:]), (255, 0, 255), 2)
        if self._drag_start and self._drag_now:
            cv2.rectangle(img, self._drag_start, self._drag_now, (255, 0, 255), 1)
        txt = f"[{self.idx + 1}/{len(self.paths)}] {os.path.basename(self.path)}  regions: {len(self.regions)}"
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        return img

    # ---- actions ----
    def accept(self):
        if self.mask is None:
            return
        self.regions.append({"contours": self.contours_of(self.mask), "mask": self.mask.copy()})
        self.points, self.labels, self.box, self.mask = [], [], None, None

    def undo(self):
        if self.points:
            self.points.pop(); self.labels.pop()
        elif self.box:
            self.box = None
        elif self.mask is None and self.regions:
            self.regions.pop()
        self.predict()

    def reset(self):
        self.points, self.labels, self.box, self.mask = [], [], None, None

    def save(self):
        self.accept()  # include a pending mask
        if not self.regions:
            return
        stem = os.path.splitext(os.path.basename(self.path))[0]
        base = os.path.join(self.out_dir, stem)
        data = [{"id": i + 1, "polygons": [c.reshape(-1, 2).tolist() for c in r["contours"]]}
                for i, r in enumerate(self.regions)]
        with open(base + "_regions.json", "w") as f:
            json.dump({"image": os.path.abspath(self.path), "regions": data}, f)
        label_map = np.zeros(self.bgr.shape[:2], np.uint16)
        for i, r in enumerate(self.regions, 1):
            label_map[r["mask"] > 0] = i
        cv2.imwrite(base + "_labels.png", label_map)
        overlay = self.bgr.copy()
        self.draw_regions(overlay)
        cv2.imwrite(base + "_overlay.png", overlay)
        print(f"Saved {len(self.regions)} region(s) for {stem} -> {self.out_dir}")

    def go(self, step):
        self.save()
        j = self.idx + step
        if 0 <= j < len(self.paths):
            self.load_image(j)
        else:
            print("No more images in that direction.")

    def open_other(self):
        self.save()
        paths = pick_images_dialog()
        if paths:
            self.set_files(paths)

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 32): self.accept()
            elif k == ord("u"): self.undo()
            elif k == ord("r"): self.reset()
            elif k == ord("s"): self.save()
            elif k == ord("n"): self.go(+1)
            elif k == ord("p"): self.go(-1)
            elif k == ord("o"): self.open_other()
            elif k in (ord("q"), 27):
                self.save()
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="image files and/or folders")
    ap.add_argument("--out", help="output folder (default: <image dir>/processed)")
    a = ap.parse_args()

    paths = resolve_inputs(a.inputs) if a.inputs else pick_images_dialog()
    if not paths:
        sys.exit("No images selected.")
    Picker(paths, a.out).run()