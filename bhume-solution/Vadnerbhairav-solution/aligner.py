"""
bhume/aligner.py  — optimized with cv2 resizing
"""

from __future__ import annotations
import numpy as np
import cv2
from skimage.filters import sobel
from skimage.morphology import dilation, erosion
try:
    from skimage.morphology import footprint_rectangle
    def _square(n): return footprint_rectangle((n, n))
except ImportError:
    from skimage.morphology import square as _square

# ── Constants ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.72
PATCH_SIZE           = 192
BUFFER_M             = 90.0
SEARCH_STEPS_COARSE  = 9
SEARCH_STEPS_FINE    = 7
MIN_BOUNDARY_DENSITY = 0.004
MIN_AREA_SQM         = 300.0

# ── Preloaded raster arrays ────────────────────────────────────────────────
_img_array = _img_tf = _bnd_array = _bnd_tf = None

def init_rasters(img_array, img_transform, bnd_array, bnd_transform):
    global _img_array, _img_tf, _bnd_array, _bnd_tf
    _img_array = img_array
    _img_tf    = img_transform
    _bnd_array = bnd_array
    _bnd_tf    = bnd_transform

# ── Pixel helpers ──────────────────────────────────────────────────────────
def _geo_window(minx, miny, maxx, maxy, tf, H, W):
    c0 = int(np.floor((minx - tf.c) / tf.a))
    r0 = int(np.floor((maxy - tf.f) / tf.e))
    c1 = int(np.ceil ((maxx - tf.c) / tf.a))
    r1 = int(np.ceil ((miny - tf.f) / tf.e))
    return max(0,r0), min(H,r1), max(0,c0), min(W,c1)

def _extract_img_patch(bounds, size=PATCH_SIZE):
    minx, miny, maxx, maxy = bounds
    r0,r1,c0,c1 = _geo_window(minx,miny,maxx,maxy,_img_tf,_img_array.shape[1],_img_array.shape[2])
    if r1<=r0 or c1<=c0: return None
    raw = _img_array[:, r0:r1, c0:c1]   # (3, h, w)
    # Resize each band with cv2
    out = np.stack([cv2.resize(raw[b], (size,size), interpolation=cv2.INTER_LINEAR) for b in range(3)])
    return out

def _extract_bnd_patch(bounds, size=PATCH_SIZE):
    minx, miny, maxx, maxy = bounds
    r0,r1,c0,c1 = _geo_window(minx,miny,maxx,maxy,_bnd_tf,_bnd_array.shape[0],_bnd_array.shape[1])
    if r1<=r0 or c1<=c0: return None
    raw = _bnd_array[r0:r1, c0:c1]
    return cv2.resize(raw, (size,size), interpolation=cv2.INTER_NEAREST)

# ── Signal ─────────────────────────────────────────────────────────────────
def _compute_signal(img_patch, bnd_patch):
    gray = (0.299*img_patch[0] + 0.587*img_patch[1] + 0.114*img_patch[2]).astype(np.float32)/255.0
    edges = sobel(gray)
    emax = edges.max()
    if emax > 0: edges /= emax
    if bnd_patch is None:
        return edges.astype(np.float32)
    bnd_f = (bnd_patch > 128).astype(np.float32)
    bnd_d = dilation(bnd_f, _square(3))
    combined = 0.40*edges + 0.60*bnd_d
    cmax = combined.max()
    if cmax > 0: combined /= cmax
    return combined.astype(np.float32)

# ── Geometry mask ──────────────────────────────────────────────────────────
def _geom_to_mask(geom, bounds, size=PATCH_SIZE):
    from rasterio.transform import from_bounds as tfrom
    from rasterio.features import rasterize
    minx,miny,maxx,maxy = bounds
    tf = tfrom(minx,miny,maxx,maxy,size,size)
    return rasterize([(geom.__geo_interface__,1)],out_shape=(size,size),transform=tf,fill=0,dtype='uint8')

# ── Score one shift ────────────────────────────────────────────────────────
def _score(geom, dx, dy, bounds, signal, size=PATCH_SIZE):
    from shapely.affinity import translate
    shifted = translate(geom, xoff=dx, yoff=dy)
    mask = _geom_to_mask(shifted, bounds, size)
    try:
        eroded = erosion(mask, _square(3))
    except Exception:
        return 0.0
    perim = np.clip(mask.astype(np.int16) - eroded.astype(np.int16), 0, 1).astype(np.float32)
    psum = perim.sum()
    if psum < 4: return 0.0
    return float((perim * signal).sum()) / float(psum)

# ── Search grid ────────────────────────────────────────────────────────────
def _grid(cx, cy, radius, steps):
    xs = np.linspace(cx-radius, cx+radius, steps)
    ys = np.linspace(cy-radius, cy+radius, steps)
    return np.array([(x,y) for x in xs for y in ys])

# ── Confidence ────────────────────────────────────────────────────────────
def _confidence(best_score, all_scores, area_sqm, density, dx, dy, prior_dx, prior_dy):
    f_peak    = float(np.clip(best_score/0.35, 0, 1))
    mean_s    = float(np.mean(all_scores)) if len(all_scores)>1 else best_score
    sharpness = (best_score/mean_s) if mean_s>1e-6 else 1.0
    f_sharp   = float(np.clip((sharpness-1.0)/3.0, 0, 1))
    f_density = float(np.clip(density/0.05, 0, 1))
    f_area    = float(np.clip(np.log10(max(area_sqm,100))/np.log10(50_000), 0, 1))
    dev_m     = ((dx-prior_dx)**2+(dy-prior_dy)**2)**0.5
    f_dev     = float(np.clip(1.0-dev_m/120.0, 0, 1))
    return float(np.clip(0.35*f_peak+0.25*f_sharp+0.15*f_density+0.10*f_area+0.15*f_dev, 0, 1))

# ── Public API ─────────────────────────────────────────────────────────────
def align_parcel(geom, prior_dx=0.0, prior_dy=0.0, area_sqm=10_000.0):
    result = dict(dx=prior_dx, dy=prior_dy, confidence=0.0, score=0.0, method_note='no_signal')

    if area_sqm < MIN_AREA_SQM:
        result['method_note'] = 'too_small_flagged'
        return result

    bounds    = geom.buffer(BUFFER_M).bounds
    img_patch = _extract_img_patch(bounds)
    if img_patch is None:
        result['method_note'] = 'out_of_raster'
        return result

    bnd_patch = _extract_bnd_patch(bounds)
    signal    = _compute_signal(img_patch, bnd_patch)
    density   = float((signal > 0.2).mean())

    if density < MIN_BOUNDARY_DENSITY:
        result['method_note'] = 'low_boundary_density'
        return result

    # Coarse search ±60 m around prior
    cands_c  = _grid(prior_dx, prior_dy, 30.0, SEARCH_STEPS_COARSE)
    scores_c = np.array([_score(geom,dx,dy,bounds,signal) for dx,dy in cands_c])
    bi       = int(np.argmax(scores_c))
    bdx, bdy = cands_c[bi]

    # Fine search ±12 m around coarse best
    cands_f  = _grid(bdx, bdy, 6.0, SEARCH_STEPS_FINE)
    scores_f = np.array([_score(geom,dx,dy,bounds,signal) for dx,dy in cands_f])
    fi       = int(np.argmax(scores_f))
    if scores_f[fi] >= scores_c[bi]:
        bdx, bdy = cands_f[fi]
        best_score = scores_f[fi]
        all_scores = scores_f
    else:
        best_score = scores_c[bi]
        all_scores = scores_c

    shift_m = ((bdx-prior_dx)**2+(bdy-prior_dy)**2)**0.5
    conf    = _confidence(best_score, all_scores, area_sqm, density,
                          bdx, bdy, prior_dx, prior_dy)
    result.update(
        dx=bdx, dy=bdy,
        confidence=round(conf, 4),
        score=round(float(best_score), 4),
        method_note=f'local_align score={best_score:.3f} density={density:.3f} shift={shift_m:.1f}m',
    )
    return result
