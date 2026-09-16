from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import correlate
from scipy.signal import convolve2d

EPS = 1e-12


def load_gray_uint8(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _as_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating) and x.size and x.max() <= 1.0 + 1e-6:
        x = x * 255.0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def normalize1(x: np.ndarray) -> np.ndarray:
    x = _as_f64(x)
    xmax = float(np.max(x))
    xmin = float(np.min(x))
    if xmax == 0.0 and xmin == 0.0:
        return x.copy()
    if abs(xmax - xmin) < EPS:
        return np.zeros_like(x)
    return np.round((x - xmin) / (xmax - xmin) * 255.0)


def entropy_u8(img: np.ndarray) -> float:
    img = _as_u8(img)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def joint_entropy_u8(img1: np.ndarray, img2: np.ndarray) -> float:
    a = _as_u8(img1).astype(np.int64)
    b = _as_u8(img2).astype(np.int64)
    hist = np.zeros((256, 256), dtype=np.float64)
    np.add.at(hist, (a.ravel(), b.ravel()), 1.0)
    p = hist / max(hist.sum(), 1.0)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def mutual_info_reference(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float, float]:
    img1 = _as_u8(img1)
    img2 = _as_u8(img2)

    hist = np.zeros((256, 256), dtype=np.float64)
    np.add.at(hist, (img1.ravel(), img2.ravel()), 1.0)
    hist /= max(hist.sum(), 1.0)

    im1_marg = np.sum(hist, axis=0)
    im2_marg = np.sum(hist, axis=1)
    h_x = float(-np.sum(im1_marg * np.log2(im1_marg + (im1_marg == 0))))
    h_y = float(-np.sum(im2_marg * np.log2(im2_marg + (im2_marg == 0))))
    h_xy = float(-np.sum(hist * np.log2(hist + (hist == 0))))
    mi = h_x + h_y - h_xy
    return float(mi), float(h_xy), float(h_x), float(h_y)


def EN_metric(F: np.ndarray) -> float:
    return entropy_u8(F)


def MI_metric(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    h_a = entropy_u8(A)
    h_b = entropy_u8(B)
    h_f = entropy_u8(F)
    h_fa = joint_entropy_u8(F, A)
    h_fb = joint_entropy_u8(F, B)
    return float((h_a + h_f - h_fa) + (h_b + h_f - h_fb))


def NMI_metric(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    a = normalize1(A)
    b = normalize1(B)
    f = normalize1(F)
    i_fx, _, h_x, h_f1 = mutual_info_reference(a, f)
    i_fy, _, h_y, h_f2 = mutual_info_reference(b, f)
    return float(2.0 * (i_fx / (h_f1 + h_x + EPS) + i_fy / (h_f2 + h_y + EPS)))


def EI_metric(F: np.ndarray) -> float:
    img = _as_f64(F)
    w = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    gx = correlate(img, w, mode="nearest")
    gy = correlate(img, w.T, mode="nearest")
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g))


def SF_metric(F: np.ndarray) -> float:
    x = _as_f64(F)
    n0, m0 = x.shape
    rf = 0.0
    cf = 0.0
    if m0 > 1:
        rf = np.sum((x[:, 1:] - x[:, :-1]) ** 2) / (n0 * m0)
    if n0 > 1:
        cf = np.sum((x[1:, :] - x[:-1, :]) ** 2) / (n0 * m0)
    return float(np.sqrt(rf + cf))


def AG_metric(F: np.ndarray) -> float:
    img = _as_f64(F)
    r, c = img.shape
    if r < 2 or c < 2:
        return 0.0
    dzdx, dzdy = np.gradient(img, 1.0, 1.0)
    s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2.0)
    return float(np.sum(s) / ((r - 1) * (c - 1)))


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(-(size - 1) / 2.0, (size - 1) / 2.0 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    w = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    return w / np.sum(w)


def ssim_index_reference(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    img1 = _as_f64(img1)
    img2 = _as_f64(img2)
    window = _gaussian_window(11, 1.5)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = convolve2d(img1, window, mode="valid")
    mu2 = convolve2d(img2, window, mode="valid")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve2d(img1 * img1, window, mode="valid") - mu1_sq
    sigma2_sq = convolve2d(img2 * img2, window, mode="valid") - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode="valid") - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + EPS
    )
    return float(np.mean(ssim_map)), ssim_map, sigma1_sq, sigma2_sq


def SSIM_metric(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    s1, _, _, _ = ssim_index_reference(F, A)
    s2, _, _, _ = ssim_index_reference(F, B)
    return float((s1 + s2) / 2.0)


def Qabf_metric(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    tg, kg, dg = 0.9994, -15.0, 0.5
    ta, ka, da = 0.9879, -22.0, 0.8

    def grad_angle(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = _as_f64(img)
        sx = convolve2d(img, h3, mode="same")
        sy = convolve2d(img, h1, mode="same")
        g = np.sqrt(sx ** 2 + sy ** 2)
        a = np.zeros_like(g)
        zero_mask = sx == 0
        a[zero_mask] = np.pi / 2.0
        a[~zero_mask] = np.arctan(sy[~zero_mask] / sx[~zero_mask])
        return g, a

    g_a, a_a = grad_angle(A)
    g_b, a_b = grad_angle(B)
    g_f, a_f = grad_angle(F)

    def quality(g_s: np.ndarray, a_s: np.ndarray) -> np.ndarray:
        g_ratio = np.zeros_like(g_s)
        gt_mask = g_s > g_f
        eq_mask = g_s == g_f
        lt_mask = ~(gt_mask | eq_mask)
        g_ratio[gt_mask] = g_f[gt_mask] / (g_s[gt_mask] + EPS)
        g_ratio[eq_mask] = g_f[eq_mask]
        g_ratio[lt_mask] = g_s[lt_mask] / (g_f[lt_mask] + EPS)
        a_ratio = 1.0 - np.abs(a_s - a_f) / (np.pi / 2.0)
        qg = tg / (1.0 + np.exp(kg * (g_ratio - dg)))
        qa = ta / (1.0 + np.exp(ka * (a_ratio - da)))
        return qg * qa

    qaf = quality(g_a, a_a)
    qbf = quality(g_b, a_b)
    den = np.sum(g_a + g_b)
    num = np.sum(qaf * g_a + qbf * g_b)
    return float(num / (den + EPS))


def _vif_single(ref: np.ndarray, dist: np.ndarray, sigma_nsq: float = 2.0, use_valid: bool = False) -> float:
    ref = _as_f64(ref)
    dist = _as_f64(dist)
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sd = n / 5.0
        ax = np.arange(-(n // 2), n // 2 + 1, dtype=np.float64)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sd * sd))
        kernel /= np.sum(kernel)

        if scale > 1:
            mode = "valid" if use_valid else "same"
            ref = convolve2d(ref, kernel, mode=mode)[::2, ::2]
            dist = convolve2d(dist, kernel, mode=mode)[::2, ::2]

        mode = "valid" if use_valid else "same"
        mu1 = convolve2d(ref, kernel, mode=mode)
        mu2 = convolve2d(dist, kernel, mode=mode)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, kernel, mode=mode) - mu1_sq
        sigma2_sq = convolve2d(dist * dist, kernel, mode=mode) - mu2_sq
        sigma12 = convolve2d(ref * dist, kernel, mode=mode) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))

    return float(num / (den + EPS))


def VIF_metric(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    # In image-fusion evaluation, VIF is commonly reported as the average
    # of the two source-to-fused VIF scores rather than their sum.
    return float(
        (_vif_single(A, F, sigma_nsq=2.0, use_valid=False) +
         _vif_single(B, F, sigma_nsq=2.0, use_valid=False)) / 2.0
    )


def VIFF_metric(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    sigma_nsq = 0.005 * 255.0 * 255.0
    weights = np.array([1.0, 0.0, 0.15, 1.0], dtype=np.float64) / 2.15

    def scale_terms(ref: np.ndarray, dist: np.ndarray):
        ref = _as_f64(ref)
        dist = _as_f64(dist)
        out = []
        for scale in range(1, 5):
            n = 2 ** (4 - scale + 1) + 1
            sd = n / 5.0
            ax = np.arange(-(n // 2), n // 2 + 1, dtype=np.float64)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sd * sd))
            kernel /= np.sum(kernel)

            if scale > 1:
                ref = convolve2d(ref, kernel, mode="valid")[::2, ::2]
                dist = convolve2d(dist, kernel, mode="valid")[::2, ::2]

            mu1 = convolve2d(ref, kernel, mode="valid")
            mu2 = convolve2d(dist, kernel, mode="valid")
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, kernel, mode="valid") - mu1_sq
            sigma2_sq = convolve2d(dist * dist, kernel, mode="valid") - mu2_sq
            sigma12 = convolve2d(ref * dist, kernel, mode="valid") - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + 1e-10)
            sv_sq = sigma2_sq - g * sigma12
            g[sigma1_sq < 1e-10] = 0
            sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
            sigma1_sq[sigma1_sq < 1e-10] = 0
            g[sigma2_sq < 1e-10] = 0
            sv_sq[sigma2_sq < 1e-10] = 0
            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= 1e-10] = 1e-10

            out.append({
                "g": g,
                "num": np.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq)),
                "den": np.log10(1.0 + sigma1_sq / sigma_nsq),
            })
        return out

    s1 = scale_terms(A, F)
    s2 = scale_terms(B, F)
    vals = []
    for i in range(4):
        choose_first = s1[i]["g"] < s2[i]["g"]
        num = s2[i]["num"].copy()
        den = s2[i]["den"].copy()
        num[choose_first] = s1[i]["num"][choose_first]
        den[choose_first] = s1[i]["den"][choose_first]
        vals.append(np.sum(num + 1e-7) / (np.sum(den + 1e-7) + EPS))
    return float(np.sum(np.asarray(vals, dtype=np.float64) * weights))


CORE_METRIC_NAMES = ["EN", "MI", "NMI", "EI", "SF", "AG", "AVG", "Qabf", "SSIM", "VIFF", "VIF"]


def evaluate_core_metrics(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> Dict[str, float]:
    ag = AG_metric(F)
    return {
        "EN": EN_metric(F),
        "MI": MI_metric(A, B, F),
        "NMI": NMI_metric(A, B, F),
        "EI": EI_metric(F),
        "SF": SF_metric(F),
        "AG": ag,
        "AVG": ag,
        "Qabf": Qabf_metric(A, B, F),
        "SSIM": SSIM_metric(A, B, F),
        "VIFF": VIFF_metric(A, B, F),
        "VIF": VIF_metric(A, B, F),
    }
