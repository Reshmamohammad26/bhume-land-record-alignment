"""
aligner_mal.py — Malatavadi v4 (final)

Key findings from score analysis:
- Plot 1177: truth score=0.171, but local search finds ~0.249 (wrong field!)
  → truth shift is only 0.7m from zero — effectively already aligned
  → Fix: search centred on ZERO also, pick best of (prior-centred, zero-centred)
- Plot 1763: truth score=0.114, prior score=0.107 — very close, hard to distinguish
  → Prior is nearly as good as truth here; apply prior, flag with moderate conf
- Plot 1966: truth score=0.369, prior score=0.135 — big gap, local can find it
  → Local search works well here

Strategy v4:
  1. Run two grids: centred on prior AND centred on zero-shift
  2. Pick global best across both grids
  3. Must beat both anchors (zero, prior) by LOCAL_MUST_BEAT to trust local
  4. Otherwise pick better of zero vs prior
"""
import numpy as np
import cv2
from skimage.filters import sobel
from skimage.morphology import dilation, erosion
try:
    from skimage.morphology import footprint_rectangle
    def _square(n): return footprint_rectangle((n, n))
except ImportError:
    from skimage.morphology import square as _square

CONFIDENCE_THRESHOLD = 0.68
PATCH_SIZE           = 192
BUFFER_M             = 45.0
SEARCH_RADIUS_C      = 18.0
SEARCH_RADIUS_F      = 4.0
SEARCH_STEPS_C       = 7      # 7×7 per grid, two grids = 98 total
SEARCH_STEPS_F       = 5
MIN_BOUNDARY_DENSITY = 0.002
MIN_AREA_SQM         = 120.0
LOCAL_MUST_BEAT      = 0.04

_img_array=_img_tf=_bnd_array=_bnd_tf=None

def init_rasters(img_array,img_transform,bnd_array,bnd_transform):
    global _img_array,_img_tf,_bnd_array,_bnd_tf
    _img_array=img_array;_img_tf=img_transform
    _bnd_array=bnd_array;_bnd_tf=bnd_transform

def _geo_window(minx,miny,maxx,maxy,tf,H,W):
    c0=int(np.floor((minx-tf.c)/tf.a));r0=int(np.floor((maxy-tf.f)/tf.e))
    c1=int(np.ceil((maxx-tf.c)/tf.a));r1=int(np.ceil((miny-tf.f)/tf.e))
    return max(0,r0),min(H,r1),max(0,c0),min(W,c1)

def _extract_img(bounds):
    minx,miny,maxx,maxy=bounds
    r0,r1,c0,c1=_geo_window(minx,miny,maxx,maxy,_img_tf,_img_array.shape[1],_img_array.shape[2])
    if r1<=r0 or c1<=c0: return None
    raw=_img_array[:,r0:r1,c0:c1]
    return np.stack([cv2.resize(raw[b],(PATCH_SIZE,PATCH_SIZE),interpolation=cv2.INTER_LINEAR) for b in range(3)])

def _extract_bnd(bounds):
    minx,miny,maxx,maxy=bounds
    r0,r1,c0,c1=_geo_window(minx,miny,maxx,maxy,_bnd_tf,_bnd_array.shape[0],_bnd_array.shape[1])
    if r1<=r0 or c1<=c0: return None
    raw=_bnd_array[r0:r1,c0:c1]
    return cv2.resize(raw,(PATCH_SIZE,PATCH_SIZE),interpolation=cv2.INTER_NEAREST)

def _signal(img_patch,bnd_patch):
    gray=(0.299*img_patch[0]+0.587*img_patch[1]+0.114*img_patch[2]).astype(np.float32)/255.0
    edges=sobel(gray);emax=edges.max()
    if emax>0:edges/=emax
    if bnd_patch is None:return edges.astype(np.float32)
    bnd_f=(bnd_patch>128).astype(np.float32)
    bnd_d=dilation(bnd_f,_square(5))
    combined=0.30*edges+0.70*bnd_d;cmax=combined.max()
    if cmax>0:combined/=cmax
    return combined.astype(np.float32)

def _mask(geom,bounds):
    from rasterio.transform import from_bounds as tfrom
    from rasterio.features import rasterize
    minx,miny,maxx,maxy=bounds
    tf=tfrom(minx,miny,maxx,maxy,PATCH_SIZE,PATCH_SIZE)
    return rasterize([(geom.__geo_interface__,1)],out_shape=(PATCH_SIZE,PATCH_SIZE),
                     transform=tf,fill=0,dtype='uint8')

def _score(geom,dx,dy,bounds,signal):
    from shapely.affinity import translate
    shifted=translate(geom,xoff=dx,yoff=dy)
    mask=_mask(shifted,bounds)
    try:eroded=erosion(mask,_square(3))
    except:return 0.0
    perim=np.clip(mask.astype(np.int16)-eroded.astype(np.int16),0,1).astype(np.float32)
    psum=perim.sum()
    if psum<2:return 0.0
    return float((perim*signal).sum())/float(psum)

def _grid(cx,cy,r,steps):
    xs=np.linspace(cx-r,cx+r,steps);ys=np.linspace(cy-r,cy+r,steps)
    return np.array([(x,y) for x in xs for y in ys])

def align_parcel(geom,prior_dx=0.0,prior_dy=0.0,area_sqm=1000.0):
    result=dict(dx=0.0,dy=0.0,confidence=0.0,score=0.0,method_note='no_signal')
    if area_sqm<MIN_AREA_SQM:
        result['method_note']='too_small_flagged';return result
    bounds=geom.buffer(BUFFER_M).bounds
    img_patch=_extract_img(bounds)
    if img_patch is None:
        result['method_note']='out_of_raster';return result
    bnd_patch=_extract_bnd(bounds)
    signal=_signal(img_patch,bnd_patch)
    density=float((signal>0.2).mean())
    if density<MIN_BOUNDARY_DENSITY:
        result['method_note']='low_density';return result

    # Anchor scores
    score_zero =_score(geom,0.0,0.0,bounds,signal)
    score_prior=_score(geom,prior_dx,prior_dy,bounds,signal)
    anchor_best=max(score_zero,score_prior)

    # Search grid 1: centred on prior
    cands1=_grid(prior_dx,prior_dy,SEARCH_RADIUS_C,SEARCH_STEPS_C)
    # Search grid 2: centred on zero (catches already-aligned plots)
    cands2=_grid(0.0,0.0,SEARCH_RADIUS_C,SEARCH_STEPS_C)
    all_cands=np.vstack([cands1,cands2])
    all_scores=np.array([_score(geom,dx,dy,bounds,signal) for dx,dy in all_cands])
    bi=int(np.argmax(all_scores));bdx,bdy=all_cands[bi];best_coarse=all_scores[bi]

    # Fine search around coarse best
    cands_f=_grid(bdx,bdy,SEARCH_RADIUS_F,SEARCH_STEPS_F)
    scores_f=np.array([_score(geom,dx,dy,bounds,signal) for dx,dy in cands_f])
    fi=int(np.argmax(scores_f))
    if scores_f[fi]>=best_coarse:
        bdx,bdy=cands_f[fi];best_local=scores_f[fi];final_all=scores_f
    else:
        best_local=best_coarse;final_all=all_scores

    # Decision
    if best_local>anchor_best+LOCAL_MUST_BEAT:
        final_dx,final_dy=bdx,bdy;final_score=best_local;method='local'
        sharpness=best_local/(np.mean(final_all)+1e-9)
        f_sharp=float(np.clip((sharpness-1.0)/2.5,0,1))
        f_imp=float(np.clip((best_local-anchor_best)/LOCAL_MUST_BEAT,0,1))
        dev_m=((bdx)**2+(bdy)**2)**0.5  # deviation from zero
        f_dev=float(np.clip(1.0-dev_m/30.0,0,1))
        f_den=float(np.clip(density/0.03,0,1))
        f_area=float(np.clip(np.log10(max(area_sqm,50))/np.log10(8000),0,1))
        conf=float(np.clip(0.30*f_imp+0.25*f_sharp+0.15*f_dev+0.15*f_den+0.15*f_area,0,1))
    elif score_zero>=score_prior:
        final_dx,final_dy=0.0,0.0;final_score=score_zero;method='keep_official'
        f_margin=float(np.clip((score_zero-score_prior)/0.03,0,1))
        f_den=float(np.clip(density/0.03,0,1))
        conf=float(np.clip(0.50*f_margin+0.30*f_den+0.20*0.4,0,1))
    else:
        final_dx,final_dy=prior_dx,prior_dy;final_score=score_prior;method='prior'
        f_margin=float(np.clip((score_prior-score_zero)/0.03,0,1))
        f_den=float(np.clip(density/0.03,0,1))
        f_area=float(np.clip(np.log10(max(area_sqm,50))/np.log10(8000),0,1))
        conf=float(np.clip(0.40*f_margin+0.30*f_den+0.20*f_area+0.10*0.4,0,1))

    shift_m=((final_dx)**2+(final_dy)**2)**0.5
    result.update(dx=final_dx,dy=final_dy,confidence=round(conf,4),
                  score=round(float(final_score),4),
                  method_note=(f'{method} score={final_score:.3f} '
                               f'zero={score_zero:.3f} prior={score_prior:.3f} '
                               f'local={best_local:.3f} density={density:.3f} shift={shift_m:.1f}m'))
    return result
