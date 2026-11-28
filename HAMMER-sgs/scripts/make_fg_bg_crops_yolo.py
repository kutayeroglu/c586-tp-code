import os, json, cv2, numpy as np, argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ----------------- helpers -----------------
def resolve_image_path(ipath: str, img_root: str) -> str:
    p = Path(ipath)
    if p.is_absolute():
        return str(p)
    full = Path(img_root) / p
    # Flattened datasets (e.g. our DGM4plus copy) keep every image directly under img_root,
    # while the JSON still says data/manipulation/<type>/<name>.jpg
    flat = Path(img_root) / p.name
    if not full.exists() and flat.exists():
        return str(flat.resolve())
    return str(full.resolve())

def center_fallback(im):
    h, w = im.shape[:2]
    s = int(0.6 * min(h, w))
    y0 = max(0, (h - s) // 2); x0 = max(0, (w - s) // 2)
    fg = im[y0:y0+s, x0:x0+s].copy()
    bg = im.copy(); bg[y0:y0+s, x0:x0+s] = 0
    return fg, bg

def mask_to_fg_bg(im, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return center_fallback(im)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    fg = im[y1:y2+1, x1:x2+1].copy()
    m2 = mask[y1:y2+1, x1:x2+1]
    fg[m2 == 0] = 0
    bg = im.copy(); bg[mask > 0] = 0
    return fg, bg

def smooth_mask(mask, close_px=1, dilate_px=2):
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_px+1, 2*close_px+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if dilate_px > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k2)
    return mask

# ----------------- masks (--save_masks) -----------------
# Each saved mask is aligned with its crop: 255 where the crop shows image content, 0 where it
# was blacked out. They mirror center_fallback / mask_to_fg_bg exactly, so the crops don't change.
def center_fallback_masks(im):
    h, w = im.shape[:2]
    s = int(0.6 * min(h, w))
    y0 = max(0, (h - s) // 2); x0 = max(0, (w - s) // 2)
    fg_m = np.full((s, s), 255, np.uint8)
    bg_m = np.full((h, w), 255, np.uint8); bg_m[y0:y0+s, x0:x0+s] = 0
    return fg_m, bg_m

def mask_to_fg_bg_masks(mask):
    ys, xs = np.where(mask > 0)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    fg_m = np.where(mask[y1:y2+1, x1:x2+1] > 0, 255, 0).astype(np.uint8)
    bg_m = np.where(mask > 0, 0, 255).astype(np.uint8)
    return fg_m, bg_m

def check_crop_mask(crop, crop_mask, name):
    """Fail loudly if a mask doesn't fit its crop: same size, and nothing outside it but black."""
    if crop.shape[:2] != crop_mask.shape:
        raise RuntimeError(f"{name}: crop {crop.shape[:2]} vs mask {crop_mask.shape}")
    if crop[crop_mask == 0].any():
        raise RuntimeError(f"{name}: non-black pixels outside the mask")

def union_person_masks(result, img_shape, person_idxs, thr=0.5):
    """Union binary masks (prefer masks.data over polygons)."""
    H, W = img_shape[:2]
    union = np.zeros((H, W), np.uint8)
    if result.masks is None or result.masks.data is None:
        return union
    m = result.masks.data  # torch.Tensor (N, Hm, Wm); with retina_masks=True it's in image size
    n = m.shape[0]
    keep = [i for i in person_idxs if 0 <= i < n]
    if not keep:
        return union
    sel = m[keep].sum(dim=0) > thr
    union[sel.cpu().numpy()] = 255
    return union

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Single FG crop covering ALL persons; BG is negation (YOLOv8-seg union).")
    ap.add_argument("--data", required=True, help="JSON with {id,image,text}")
    ap.add_argument("--img_root", required=True, help="Folder that contains 'origin' and 'manipulation'")
    ap.add_argument("--out_dir", default="crops", help="Output dir (creates fg/ and bg/)")
    ap.add_argument("--model", default="yolov8s-seg.pt", help="Ultralytics segmentation model (e.g., yolov8n-seg.pt, yolov8s-seg.pt)")
    ap.add_argument("--imgsz", type=int, default=896, help="Inference size for better masks (CPU ok, slower if larger)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--topk", type=int, default=3, help="Max persons to include (by box area)")
    ap.add_argument("--min_area_frac", type=float, default=0.01, help="Discard tiny person detections (<1% of image area)")
    ap.add_argument("--close", type=int, default=1, help="Mask closing pixels")
    ap.add_argument("--dilate", type=int, default=2, help="Mask dilation pixels")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing crops")
    ap.add_argument("--check", action="store_true", help="Only validate paths; no outputs")
    ap.add_argument("--save_masks", action="store_true",
                    help="Also write masks/{fg,bg}/<id>.png aligned with each crop (255 = image content) "
                         "and masks/meta.csv; the crops themselves are unchanged")
    args = ap.parse_args()

    out_fg = Path(args.out_dir) / "fg"
    out_bg = Path(args.out_dir) / "bg"
    out_mfg = Path(args.out_dir) / "masks" / "fg"
    out_mbg = Path(args.out_dir) / "masks" / "bg"
    if not args.check:
        out_fg.mkdir(parents=True, exist_ok=True)
        out_bg.mkdir(parents=True, exist_ok=True)
        if args.save_masks:
            out_mfg.mkdir(parents=True, exist_ok=True)
            out_mbg.mkdir(parents=True, exist_ok=True)
    meta_rows = []  # one per written crop pair, for masks/meta.csv

    model = None
    if not args.check:
        model = YOLO(args.model)  # e.g., yolov8s-seg.pt (more reliable for multiple people)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    ok = missed = skipped = 0
    seen = {}  # resolved path -> JSON path, to catch file-name collisions from the flattened fallback
    for row in tqdm(data):
        iid = row.get("id"); rel = row.get("image", "")
        if iid is None or not rel:
            missed += 1; continue
        rel = rel.replace("\\","/")
        ipath = resolve_image_path(rel, args.img_root)
        if not os.path.exists(ipath):
            print(f"[warn] missing image for id={iid}: {ipath}")
            missed += 1; continue
        if seen.setdefault(ipath, rel) != rel:
            print(f"[warn] id={iid}: '{rel}' and '{seen[ipath]}' both resolve to {ipath}")

        if args.check:
            ok += 1; continue

        fgp = out_fg / f"{iid}.png"; bgp = out_bg / f"{iid}.png"
        if fgp.exists() and bgp.exists() and not args.overwrite:
            skipped += 1; continue

        im = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if im is None:
            missed += 1; continue
        H, W = im.shape[:2]; img_area = H * W

        try:
            res = model.predict(source=im, verbose=False, imgsz=args.imgsz, conf=args.conf, retina_masks=True)[0]

            # select up to topk persons by bbox area, filter tiny ones
            person = []
            if res.boxes is not None and len(res.boxes):
                bxyxy = res.boxes.xyxy.cpu().numpy()
                bcls  = res.boxes.cls.cpu().numpy()
                for idx, (b, c) in enumerate(zip(bxyxy, bcls)):
                    if int(c) == 0:  # person
                        x1, y1, x2, y2 = b.astype(int).tolist()
                        area = max(1, (x2-x1) * (y2-y1))
                        if area / img_area >= args.min_area_frac:
                            person.append((area, idx))
            person.sort(reverse=True)
            person_idxs = [idx for _, idx in person[:max(1, args.topk)]]

            fallback = ""
            if person_idxs:
                mask = union_person_masks(res, im.shape, person_idxs, thr=0.5)
                if mask.sum() == 0:
                    fg, bg = center_fallback(im); fallback = "empty_mask"
                else:
                    mask = smooth_mask(mask, close_px=args.close, dilate_px=args.dilate)
                    fg, bg = mask_to_fg_bg(im, mask)
            else:
                fg, bg = center_fallback(im); fallback = "no_person"

            cv2.imwrite(str(fgp), fg); cv2.imwrite(str(bgp), bg); ok += 1
        except Exception as e:
            print(f"[fallback] id={iid}: {e}")
            fg, bg = center_fallback(im); fallback = "exception"; person_idxs = []
            cv2.imwrite(str(fgp), fg); cv2.imwrite(str(bgp), bg); ok += 1

        if args.save_masks:
            fg_m, bg_m = center_fallback_masks(im) if fallback else mask_to_fg_bg_masks(mask)
            check_crop_mask(fg, fg_m, f"id={iid} fg"); check_crop_mask(bg, bg_m, f"id={iid} bg")
            cv2.imwrite(str(out_mfg / f"{iid}.png"), fg_m); cv2.imwrite(str(out_mbg / f"{iid}.png"), bg_m)
            meta_rows.append({
                "id": iid, "image": rel, "height": H, "width": W,
                "n_persons": 0 if fallback else len(person_idxs),
                "fallback": fallback or "none",
                # share of the image treated as foreground (the person mask, or the center square)
                "fg_area_frac": f"{int((bg_m == 0).sum()) / img_area:.4f}",
            })

    if args.save_masks and meta_rows:
        import csv
        with open(Path(args.out_dir) / "masks" / "meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader(); w.writerows(meta_rows)
        n_fb = sum(r["fallback"] != "none" for r in meta_rows)
        print(f"[masks] wrote {len(meta_rows)} mask pairs; center-square fallback for {n_fb}")

    print(f"[done] OK:{ok}  Skipped:{skipped}  Missed/failed:{missed}  Total:{len(data)}")

if __name__ == "__main__":
    main()
