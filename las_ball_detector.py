#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAS 点云高反小球检测工具
========================================
功能:
  1. 读取 LAS 格式点云
  2. 基于强度阈值过滤点云（支持交互式滑块选择阈值）
  3. DBSCAN 聚类提取高反小球中心（强度加权重心）
  4. 交互式剔除异常聚类（点击 3D/俯视图星形标记 或 手动输入 ID）
  5. 可视化强度过滤结果、聚类结果
  6. 导出聚类中心坐标为 CSV

依赖:
    pip install laspy[lazrs] numpy scikit-learn matplotlib scipy
"""

import sys
import csv
import argparse

import numpy as np
import laspy
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

# 中文字体尝试设置（Windows 环境）
try:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# 修复 matplotlib 3.9+ ResizeEvent 缺少 inaxes 属性导致的 AttributeError 刷屏问题
try:
    import matplotlib.cbook as _mpl_cbook
    _orig_exc_handler = getattr(_mpl_cbook.CallbackRegistry, '_default_exception_handler',
                                 _mpl_cbook._exception_handler
                                 if hasattr(_mpl_cbook, '_exception_handler') else None)

    def _safe_exc_handler(exc_info):
        """过滤 ResizeEvent/inaxes 的已知 matplotlib bug，其余照常输出。"""
        exc_type, exc_val, _ = exc_info
        if exc_type is AttributeError and 'inaxes' in str(exc_val):
            return  # 静默忽略，不影响功能
        import traceback
        traceback.print_exception(*exc_info)

    # 替换模块级默认异常处理器（影响所有新建的 CallbackRegistry）
    if hasattr(_mpl_cbook, '_exception_handler'):
        _mpl_cbook._exception_handler = _safe_exc_handler
    # 同时补丁 CallbackRegistry.__init__ 使已有实例也受益
    _orig_cr_init = _mpl_cbook.CallbackRegistry.__init__
    def _patched_cr_init(self, exception_handler=_safe_exc_handler, *args, **kwargs):
        _orig_cr_init(self, exception_handler=exception_handler, *args, **kwargs)
    _mpl_cbook.CallbackRegistry.__init__ = _patched_cr_init
except Exception:
    pass  # 如果 matplotlib 内部 API 变化，跳过补丁不影响运行


# ─────────────────────────────────────────────────────────────────────────────
# 核心数据类
# ─────────────────────────────────────────────────────────────────────────────

class BallDetector:
    """高反小球点云检测与聚类主类。"""

    def __init__(self):
        self.points: np.ndarray = None          # shape (N, 3)
        self.intensities: np.ndarray = None      # shape (N,)
        self.filtered_points: np.ndarray = None
        self.filtered_intensities: np.ndarray = None
        self.cluster_labels: np.ndarray = None  # DBSCAN 标签
        self.clusters: dict = {}                 # label -> dict
        self.removed_clusters: set = set()
        self.intensity_threshold: float = None
        self._eps: float = None

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_las(self, las_file: str):
        """读取 LAS/LAZ 文件。"""
        print(f"\n[1/4] 读取 LAS 文件: {las_file}")
        las = laspy.read(las_file)
        self.points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        self.intensities = np.array(las.intensity, dtype=np.float64)
        print(f"      共读取 {len(self.points):,} 个点")
        print(f"      强度范围: {self.intensities.min():.1f} ~ {self.intensities.max():.1f}")
        print(f"      强度均值: {self.intensities.mean():.1f}  标准差: {self.intensities.std():.1f}")

    def export_results(self, output_prefix: str = "ball_centers"):
        """导出最终聚类中心为 CSV。"""
        active = {k: v for k, v in self.clusters.items() if k not in self.removed_clusters}
        if not active:
            print("没有可导出的聚类结果。")
            return

        csv_file = f"{output_prefix}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "X", "Y", "Z", "点数", "平均强度", "最大强度"])
            for label, info in sorted(active.items()):
                c = info["center"]
                writer.writerow([
                    label,
                    f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}",
                    info["count"],
                    f"{info['mean_intensity']:.2f}",
                    f"{info['max_intensity']:.2f}",
                ])
        print(f"\n结果已导出 → {csv_file}  (共 {len(active)} 个聚类中心)")
        print(f"\n{'ID':>4}  {'X':>12}  {'Y':>12}  {'Z':>12}  {'点数':>6}  {'平均强度':>10}")
        print("─" * 62)
        for label, info in sorted(active.items()):
            c = info["center"]
            print(f"{label:>4}  {c[0]:>12.4f}  {c[1]:>12.4f}  {c[2]:>12.4f}  "
                  f"{info['count']:>6}  {info['mean_intensity']:>10.2f}")

    # ── 处理流程 ──────────────────────────────────────────────────────────────

    def filter_by_intensity(self, threshold: float):
        """按阈值过滤点云。"""
        self.intensity_threshold = threshold
        mask = self.intensities >= threshold
        self.filtered_points = self.points[mask]
        self.filtered_intensities = self.intensities[mask]
        print(f"\n[2/4] 强度过滤 (阈值={threshold:.1f}): "
              f"{mask.sum():,} 个点保留 / {len(self.points):,} 个原始点")

    def cluster(self, eps: float, min_samples: int):
        """DBSCAN 聚类。"""
        pts = self.filtered_points
        if pts is None or len(pts) == 0:
            print("过滤后无可用点，请降低强度阈值。")
            return

        self._eps = eps
        print(f"\n[3/4] DBSCAN 聚类 (eps={eps:.4f}, min_samples={min_samples}) ...")
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts)
        self.cluster_labels = db.labels_

        labels_unique = set(db.labels_) - {-1}
        n_noise = int((db.labels_ == -1).sum())
        print(f"      发现 {len(labels_unique)} 个聚类，{n_noise} 个噪声点")

        self.clusters = {}
        for label in sorted(labels_unique):
            mask = db.labels_ == label
            cpts = pts[mask]
            cint = self.filtered_intensities[mask]
            point_indices = np.where(mask)[0]  # 保存点在过滤点云中的索引
            # 强度加权重心，抑制低强度拖尾偏移
            w = cint / cint.sum()
            center = np.average(cpts, weights=w, axis=0)
            # 包围盒半径（估计球半径）
            radius = np.linalg.norm(cpts - center, axis=1).mean()
            self.clusters[label] = {
                "points": cpts,
                "intensities": cint,
                "indices": point_indices,  # 新增：点在过滤点云中的索引
                "center": center,
                "radius": radius,
                "count": len(cpts),
                "mean_intensity": float(cint.mean()),
                "max_intensity": float(cint.max()),
            }
            print(f"      聚类 {label:>3}: {len(cpts):>5} 点  "
                  f"中心=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})  "
                  f"R≈{radius:.4f}  强度均值={cint.mean():.1f}")

    # ── 可视化：强度过滤结果 ──────────────────────────────────────────────────

    def visualize_intensity_filter(self):
        """弹出窗口：强度分布 + 过滤前后 3D 对比。"""
        if self.points is None:
            return

        fig = plt.figure(figsize=(16, 5))
        fig.suptitle("强度过滤结果", fontsize=13, fontweight="bold")

        # ① 强度直方图
        ax1 = fig.add_subplot(131)
        ax1.hist(self.intensities, bins=200, color="#4C72B0", alpha=0.75, label="全部点")
        if self.intensity_threshold is not None:
            ax1.axvline(self.intensity_threshold, color="red", lw=2, ls="--",
                        label=f"阈值 {self.intensity_threshold:.0f}")
            kept = (self.intensities >= self.intensity_threshold).sum()
            ax1.set_title(f"强度分布\n阈值以上: {kept:,} 点")
        else:
            ax1.set_title("强度分布")
        ax1.set_xlabel("强度值"); ax1.set_ylabel("点数"); ax1.legend(fontsize=8)

        # ② 原始点云（按强度着色，降采样）
        ax2 = fig.add_subplot(132, projection="3d")
        step = max(1, len(self.points) // 50000)
        sc = ax2.scatter(self.points[::step, 0], self.points[::step, 1],
                         self.points[::step, 2],
                         c=self.intensities[::step], cmap="jet", s=0.3, alpha=0.4)
        plt.colorbar(sc, ax=ax2, shrink=0.5, pad=0.05, label="强度")
        ax2.set_title("原始点云（强度着色）")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        # ③ 过滤后点云
        ax3 = fig.add_subplot(133, projection="3d")
        if self.filtered_points is not None and len(self.filtered_points) > 0:
            step2 = max(1, len(self.filtered_points) // 20000)
            sc2 = ax3.scatter(self.filtered_points[::step2, 0],
                              self.filtered_points[::step2, 1],
                              self.filtered_points[::step2, 2],
                              c=self.filtered_intensities[::step2],
                              cmap="hot", s=2, alpha=0.7)
            plt.colorbar(sc2, ax=ax3, shrink=0.5, pad=0.05, label="强度")
        ax3.set_title(f"强度过滤后 (≥{self.intensity_threshold:.0f})\n"
                      f"{len(self.filtered_points):,} 个点")
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

        plt.tight_layout()
        plt.show(block=False)

    # ── 可视化：交互式聚类编辑 ────────────────────────────────────────────────

    def visualize_clusters_interactive(self):
        """交互式聚类结果视图，支持点击/输入剔除小球。"""
        if not self.clusters:
            print("无聚类结果可显示。")
            return

        self.removed_clusters = set()
        self._build_interactive_figure()

    def _build_interactive_figure(self):
        self._fig = plt.figure(figsize=(18, 9))
        self._fig.suptitle(
            "聚类结果交互编辑  |  点击星形(★)标记/取消剔除  |  右下角可手动输入 ID",
            fontsize=11)

        # ── 子图布局 ──────────────────────────────────────────────
        self._ax3d = self._fig.add_subplot(121, projection="3d")
        self._ax2d = self._fig.add_subplot(122)
        self._fig.subplots_adjust(bottom=0.18, hspace=0.35)

        # ── 按钮行 ────────────────────────────────────────────────
        ax_btn_confirm = plt.axes([0.10, 0.05, 0.12, 0.05])
        ax_btn_reset   = plt.axes([0.24, 0.05, 0.12, 0.05])
        ax_btn_export  = plt.axes([0.38, 0.05, 0.12, 0.05])
        ax_textbox     = plt.axes([0.60, 0.05, 0.20, 0.05])
        ax_btn_del_id  = plt.axes([0.82, 0.05, 0.12, 0.05])

        self._btn_confirm = Button(ax_btn_confirm, "确认剔除", color="#d9534f", hovercolor="#c9302c")
        self._btn_reset   = Button(ax_btn_reset,   "重置选择",  color="#f0ad4e", hovercolor="#ec971f")
        self._btn_export  = Button(ax_btn_export,  "导出 CSV", color="#5cb85c", hovercolor="#449d44")
        self._textbox     = TextBox(ax_textbox, "剔除 ID (逗号分隔): ", initial="")
        self._btn_del_id  = Button(ax_btn_del_id,  "按 ID 剔除",  color="#5bc0de", hovercolor="#31b0d5")

        self._btn_confirm.on_clicked(self._on_confirm)
        self._btn_reset.on_clicked(self._on_reset)
        self._btn_export.on_clicked(self._on_export)
        self._btn_del_id.on_clicked(self._on_remove_by_id)

        self._pick_map = {}   # artist -> cluster_label
        self._fig.canvas.mpl_connect("pick_event", self._on_pick)

        self._redraw()
        plt.show()

    def _get_cmap(self):
        n = max(len(self.clusters), 1)
        return [plt.cm.tab20(i % 20) for i in range(n)]

    def _redraw(self):
        self._ax3d.cla()
        self._ax2d.cla()
        self._pick_map.clear()

        colors = self._get_cmap()
        active_labels = sorted(self.clusters.keys())

        # ── 强度过滤后全量点云（浅蓝背景，供人工判断聚类中心是否居中）──────────
        if self.filtered_points is not None and len(self.filtered_points) > 0:
            _fp = self.filtered_points
            _step_bg = max(1, len(_fp) // 80000)  # 80K 采样
            self._ax3d.scatter(_fp[::_step_bg, 0], _fp[::_step_bg, 1], _fp[::_step_bg, 2],
                               c="#9ecae1", s=2, alpha=0.45,
                               label="强度过滤后点云")
            self._ax2d.scatter(_fp[::_step_bg, 0], _fp[::_step_bg, 1],
                               c="#9ecae1", s=2, alpha=0.45)

        # ── 各聚类点 + 中心星形 ───────────────────────────────────
        for idx, label in enumerate(active_labels):
            info = self.clusters[label]
            color = colors[idx]
            removed = label in self.removed_clusters
            pt_alpha = 0.08 if removed else 0.45
            star_alpha = 0.35 if removed else 1.0
            edge_col = "red" if removed else "black"

            pts = info["points"]
            step = max(1, len(pts) // 5000)

            # 聚类散点
            self._ax3d.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                               c=[color], s=4, alpha=pt_alpha)
            self._ax2d.scatter(pts[::step, 0], pts[::step, 1],
                               c=[color], s=6, alpha=pt_alpha)

            # 中心星形（可点击）
            c = info["center"]
            sc3 = self._ax3d.scatter([c[0]], [c[1]], [c[2]],
                                     c=[color], s=250, marker="*",
                                     alpha=star_alpha,
                                     edgecolors=edge_col, linewidths=1.5,
                                     picker=12,
                                     zorder=10)
            sc2 = self._ax2d.scatter([c[0]], [c[1]],
                                     c=[color], s=250, marker="*",
                                     alpha=star_alpha,
                                     edgecolors=edge_col, linewidths=1.5,
                                     picker=12,
                                     zorder=10)

            self._pick_map[sc3] = label
            self._pick_map[sc2] = label

            # ID 标注
            lbl_text = f"{label}" + ("✗" if removed else "")
            self._ax2d.annotate(lbl_text, (c[0], c[1]),
                                textcoords="offset points", xytext=(4, 4),
                                fontsize=7, color="red" if removed else "dimgray")

        # ── 标题 / 说明 ───────────────────────────────────────────
        total = len(self.clusters)
        rem_n = len(self.removed_clusters)
        self._ax3d.set_title(
            f"3D视图  共{total}个小球  已标记剔除{rem_n}个\n"
            f"背景=全量强度过滤点  ★=聚类中心  点击★标记/取消剔除", fontsize=9)
        self._ax3d.set_xlabel("X"); self._ax3d.set_ylabel("Y"); self._ax3d.set_zlabel("Z")

        self._ax2d.set_title("俯视图(XY平面)  背景=强度过滤后全部点  ★=聚类中心", fontsize=9)
        self._ax2d.set_xlabel("X"); self._ax2d.set_ylabel("Y")
        self._ax2d.set_aspect("equal", adjustable="datalim")
        self._ax2d.grid(True, alpha=0.3)

        self._fig.canvas.draw_idle()

    # ── 事件回调 ──────────────────────────────────────────────────────────────

    def _on_pick(self, event):
        artist = event.artist
        label = self._pick_map.get(artist)
        if label is None:
            return
        if label in self.removed_clusters:
            self.removed_clusters.discard(label)
            print(f"  取消剔除: 聚类 {label}")
        else:
            self.removed_clusters.add(label)
            print(f"  标记剔除: 聚类 {label}")
        self._redraw()

    def _on_confirm(self, _event):
        if not self.removed_clusters:
            print("  没有标记要剔除的聚类。")
            return
        ids = sorted(self.removed_clusters)
        print(f"\n  确认剔除聚类: {ids}")
        for lbl in ids:
            self.clusters.pop(lbl, None)
        self.removed_clusters.clear()
        self._redraw()
        print(f"  剔除完成，剩余 {len(self.clusters)} 个聚类。")

    def _on_reset(self, _event):
        self.removed_clusters.clear()
        self._redraw()
        print("  已重置所有标记。")

    def _on_export(self, _event):
        self.export_results()

    def _on_remove_by_id(self, _event):
        raw = self._textbox.text.strip()
        if not raw:
            return
        ids_to_remove = set()
        for tok in raw.replace("，", ",").split(","):
            tok = tok.strip()
            if tok.lstrip("-").isdigit():
                ids_to_remove.add(int(tok))
        invalid = ids_to_remove - set(self.clusters.keys())
        ids_to_remove -= invalid
        if invalid:
            print(f"  无效 ID (不存在): {sorted(invalid)}")
        if not ids_to_remove:
            print("  没有有效 ID 可剔除。")
            return
        for lbl in ids_to_remove:
            self.clusters.pop(lbl, None)
            self.removed_clusters.discard(lbl)
        print(f"  已剔除聚类: {sorted(ids_to_remove)}，剩余 {len(self.clusters)} 个。")
        self._redraw()



# ═══════════════════════════════════════════════════════════════════════════════
# 坐标系编辑：长方体平面拟合 + 交线 + 交点（焦点）+ 交互式变换
# ═══════════════════════════════════════════════════════════════════════════════

def _rot_mat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues 旋转公式 → 3×3 旋转矩阵（右手定则）"""
    ax = np.asarray(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-15)
    theta = np.radians(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    K = np.array([[0., -ax[2], ax[1]],
                  [ax[2], 0., -ax[0]],
                  [-ax[1], ax[0], 0.]])
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(ax, ax)


def _gs3(R: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 正交归一化 3×3 矩阵（列向量：X Y Z）"""
    x = R[:, 0].copy()
    nx = np.linalg.norm(x)
    x = x / nx if nx > 1e-12 else np.array([1., 0., 0.])
    y = R[:, 1].copy()
    y = y - np.dot(y, x) * x
    ny = np.linalg.norm(y)
    if ny < 1e-10:
        tmp = np.array([0., 0., 1.]) if abs(x[2]) < 0.9 else np.array([0., 1., 0.])
        y = tmp - np.dot(tmp, x) * x
        y /= np.linalg.norm(y)
    else:
        y /= ny
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    return np.column_stack([x, y, z])


def _fit_dominant_plane_ransac(points: np.ndarray, rng: np.random.Generator,
                               distance_threshold: float,
                               reject_parallel_to: np.ndarray = None,
                               iterations: int = 600):
    """RANSAC 拟合支撑点最多的平面，可排除与给定法向近似平行的候选。"""
    best_normal = None
    best_offset = None
    best_mask = None
    best_count = 0
    count = len(points)

    for _ in range(iterations):
        ids = rng.choice(count, 3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if (reject_parallel_to is not None and
                abs(float(np.dot(normal, reject_parallel_to))) > 0.35):
            continue

        offset = float(np.dot(normal, p0))
        mask = np.abs(points @ normal - offset) <= distance_threshold
        inlier_count = int(mask.sum())
        if inlier_count > best_count:
            best_normal = normal
            best_offset = offset
            best_mask = mask
            best_count = inlier_count

    if best_mask is None or best_count < 3:
        raise RuntimeError("RANSAC 未找到有效平面，请检查点云是否包含明显的长方体表面。")

    # 用全部内点做 SVD 精修法向，避免三个随机点带来的方向抖动。
    inliers = points[best_mask]
    inlier_center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - inlier_center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, best_normal) < 0:
        normal *= -1
    offset = float(np.median(inliers @ normal))
    residual = np.abs(inliers @ normal - offset)
    return normal, offset, best_count, float(np.median(residual))


def fit_box_planes_pca(points: np.ndarray):
    """
    稳健拟合长方体三对平面，并计算 12 条交线和 8 个角点。

    方向不再直接采用全局 PCA：先用 RANSAC 拟合两个互相垂直的主平面，
    再正交化得到第三方向。这样不受六个面点数不均、内部点或局部杂点影响。
    """
    if points is None or len(points) < 100:
        raise ValueError("点云数量不足，无法拟合长方体平面。")

    rng = np.random.default_rng(20260814)
    sample_count = min(len(points), 120000)
    if sample_count < len(points):
        sample_ids = rng.choice(len(points), sample_count, replace=False)
        sample = np.asarray(points[sample_ids], dtype=np.float64)
    else:
        sample = np.asarray(points, dtype=np.float64)

    # 用中位数中心化，降低 LAS 大坐标值导致的数值误差。
    robust_center = np.median(sample, axis=0)
    local = sample - robust_center
    spans = np.percentile(local, 99.5, axis=0) - np.percentile(local, 0.5, axis=0)
    diagonal = float(np.linalg.norm(spans))
    distance_threshold = max(diagonal * 0.0015, 1e-4)

    normal0, _, count0, residual0 = _fit_dominant_plane_ransac(
        local, rng, distance_threshold)
    normal1_raw, _, count1, residual1 = _fit_dominant_plane_ransac(
        local, rng, distance_threshold, reject_parallel_to=normal0)

    # 强制三方向严格正交。第二法向先剔除在第一法向上的分量。
    normal0 /= np.linalg.norm(normal0)
    normal1 = normal1_raw - np.dot(normal1_raw, normal0) * normal0
    if np.linalg.norm(normal1) < 1e-8:
        raise RuntimeError("检测到的两个主平面近似平行，无法建立长方体坐标系。")
    normal1 /= np.linalg.norm(normal1)
    normal2 = np.cross(normal0, normal1)
    normal2 /= np.linalg.norm(normal2)

    axes = np.column_stack([normal0, normal1, normal2])

    # 按三个方向的稳健跨度从大到小排序，继续保持 e0/e1/e2 的原有语义。
    local_proj = local @ axes
    fit_spans = (np.percentile(local_proj, 99.7, axis=0) -
                 np.percentile(local_proj, 0.3, axis=0))
    order = np.argsort(fit_spans)[::-1]
    pca_axes = axes[:, order]
    if np.linalg.det(pca_axes) < 0:
        pca_axes[:, 2] *= -1

    # 在抽样点上计算每对面的稳健边界。统一使用同一组正交法向，确保
    # 平面、交线、交点在几何上严格一致。
    projected_world = sample @ pca_axes
    d_lows = np.percentile(projected_world, 0.3, axis=0)
    d_highs = np.percentile(projected_world, 99.7, axis=0)
    extents = d_highs - d_lows
    mid_offsets = (d_lows + d_highs) * 0.5
    box_center = np.linalg.solve(pca_axes.T, mid_offsets)

    plane_pairs = []
    for i in range(3):
        other = [j for j in range(3) if j != i]
        plane_pairs.append({
            "normal": pca_axes[:, i].copy(),
            "d_lo": float(d_lows[i]),
            "d_hi": float(d_highs[i]),
            "extent": float(extents[i]),
            "ext_a": float(extents[other[0]] * 0.505),
            "ext_b": float(extents[other[1]] * 0.505),
            "label": f"e{i}",
            "box_center": box_center.copy(),
        })

    A_mat = pca_axes.T
    corners = []
    for s0 in range(2):
        for s1 in range(2):
            for s2 in range(2):
                sels = (s0, s1, s2)
                offsets = np.array([
                    plane_pairs[k]["d_lo" if sels[k] == 0 else "d_hi"]
                    for k in range(3)
                ])
                corners.append(np.linalg.solve(A_mat, offsets))
    corners = np.asarray(corners)

    lines = []
    group_names = ["e0方向", "e1方向", "e2方向"]
    for k in range(3):
        i, j = [m for m in range(3) if m != k]
        direction = pca_axes[:, k].copy()
        line_idx = 0
        for di in [plane_pairs[i]["d_lo"], plane_pairs[i]["d_hi"]]:
            for dj in [plane_pairs[j]["d_lo"], plane_pairs[j]["d_hi"]]:
                A_line = np.array([
                    plane_pairs[i]["normal"],
                    plane_pairs[j]["normal"],
                    direction,
                ])
                b_line = np.array([di, dj, float(box_center @ direction)])
                point = np.linalg.solve(A_line, b_line)
                lines.append((direction.copy(), point, k,
                              f"L{k}{line_idx}({group_names[k]})"))
                line_idx += 1

    diagnostics = {
        "method": "RANSAC + orthogonal refinement",
        "distance_threshold": distance_threshold,
        "plane_inliers": (count0, count1),
        "median_residuals": (residual0, residual1),
    }
    for pair in plane_pairs:
        pair["diagnostics"] = diagnostics

    return pca_axes, box_center, plane_pairs, corners, lines


def _add_plane_patch(ax, normal: np.ndarray, d: float,
                     u_ax: np.ndarray, v_ax: np.ndarray,
                     ext_u: float, ext_v: float, color, alpha: float = 0.09):
    """在 3D 坐标轴上绘制一个平面矩形补丁（半透明）。"""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    plane_c = n * d
    pts_poly = [
        plane_c + ext_u * u_ax + ext_v * v_ax,
        plane_c - ext_u * u_ax + ext_v * v_ax,
        plane_c - ext_u * u_ax - ext_v * v_ax,
        plane_c + ext_u * u_ax - ext_v * v_ax,
    ]
    poly = Poly3DCollection([pts_poly], alpha=alpha,
                             facecolor=color, edgecolor=color, linewidth=0.3)
    ax.add_collection3d(poly)


# ═══════════════════════════════════════════════════════════════════════════════
# 坐标系编辑器（HTML / Plotly.js 版，前置步骤）
# ═══════════════════════════════════════════════════════════════════════════════

_COORD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>坐标系编辑</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" crossorigin="anonymous"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;height:100vh;overflow:hidden;font-family:'Microsoft YaHei','Segoe UI',sans-serif;font-size:13px;background:#eee}
#pw{flex:0 0 65%;height:100vh}
#panel{flex:0 0 35%;height:100vh;overflow-y:auto;background:#f4f4f4;padding:7px;display:flex;flex-direction:column;gap:5px}
.sec{background:#fff;border-radius:5px;padding:8px;border:1px solid #e0e0e0}
.sec h4{font-size:11px;color:#666;margin-bottom:6px;padding-bottom:3px;border-bottom:1px solid #f0f0f0;text-transform:uppercase;letter-spacing:.5px}
.row{display:flex;align-items:center;gap:4px;margin-bottom:4px;flex-wrap:wrap}
label{color:#555;min-width:38px;font-size:12px}
button{padding:4px 9px;border:1px solid #bbb;border-radius:3px;cursor:pointer;font-size:12px;background:#f9f9f9;transition:background .15s}
button:hover{background:#e8e8e8}
.btn-ax{background:#ffcdd2;border-color:#e57373}
.btn-ay{background:#c8e6c9;border-color:#81c784}
.active-x{background:#e53935;color:#fff;border-color:#c62828}
.active-y{background:#43a047;color:#fff;border-color:#2e7d32}
.sel-btn{background:#f48fb1!important;border-color:#f06292!important}
input{padding:3px 6px;border:1px solid #ccc;border-radius:3px;font-size:12px}
input[type=number]{width:72px}
input[type=text]{width:130px}
#info{background:#1a1a2e;color:#a8d8ea;font-family:'Consolas','Courier New',monospace;font-size:11px;padding:8px;border-radius:4px;line-height:1.6;white-space:pre}
.sxl{color:#e53935;font-weight:bold;font-size:12px}
.syl{color:#2e7d32;font-weight:bold;font-size:12px}
.hint{color:#999;font-size:11px;font-style:italic;margin-bottom:3px}
.cbrow{display:flex;flex-wrap:wrap;gap:3px}
.cbrow button{padding:3px 7px;font-size:11px}
.acrow{display:flex;gap:6px;margin-top:3px}
.acrow button{flex:1;padding:8px;font-size:13px;font-weight:bold;border-radius:4px}
#bFinish{background:#a5d6a7;border-color:#66bb6a}
#bReset{background:#ffccbc;border-color:#ff8a65}
#bCancel{background:#cfd8dc;border-color:#90a4ae}
.title-bar{background:#1565c0;color:#fff;border-radius:4px;padding:7px 10px;font-weight:bold;font-size:13px;text-align:center}
</style>
</head>
<body>
<div id="pw"><div id="plot" style="width:100%;height:100%"></div></div>
<div id="panel">
  <div class="title-bar">坐标系编辑（前置步骤）</div>

  <div class="sec">
    <h4>坐标轴方向</h4>
    <p class="hint">先选择目标轴，再点3D交线◎；也可用方向组按钮</p>
    <div class="row">
      <button id="bSetX" class="btn-ax active-x" onclick="setTarget(0)">[当前] 设置X轴</button>
      <button id="bSetY" class="btn-ay" onclick="setTarget(1)">设置Y轴</button>
    </div>
    <div class="row">
      <label>交线:</label>
      <button style="border-color:#e74c3c;color:#c0392b" onclick="selectLineGroup(0)">红色组</button>
      <button style="border-color:#2ecc71;color:#198b48" onclick="selectLineGroup(1)">绿色组</button>
      <button style="border-color:#3498db;color:#2471a3" onclick="selectLineGroup(2)">蓝色组</button>
    </div>
    <div class="row"><span class="sxl" id="xStat">X轴: 默认(e0+)</span>
      <button onclick="setFlip(0,false)">+ 正向</button>
      <button onclick="setFlip(0,true)">- 反向</button>
    </div>
    <div class="row"><span class="syl" id="yStat">Y轴: 默认(e1+)</span>
      <button onclick="setFlip(1,false)">+ 正向</button>
      <button onclick="setFlip(1,true)">- 反向</button>
    </div>
    <div style="color:#888;font-size:11px;margin-top:2px">Z轴: 自动 = X × Y（右手系）</div>
  </div>

  <div class="sec">
    <h4>自定义方向（输入 x,y,z 覆盖交线选择）</h4>
    <div class="row"><label>X覆盖:</label>
      <input type="text" id="cxIn" placeholder="0,0,1">
      <button onclick="setCustom(0)">设置</button>
      <button onclick="clearCustom(0)">清除</button>
    </div>
    <div class="row"><label>Y覆盖:</label>
      <input type="text" id="cyIn" placeholder="0,1,0">
      <button onclick="setCustom(1)">设置</button>
      <button onclick="clearCustom(1)">清除</button>
    </div>
  </div>

  <div class="sec">
    <h4>坐标原点（◆点击3D或点按钮）</h4>
    <div class="row">
      <button id="btn-o-world" class="sel-btn" onclick="selWorldOrigin()">原始原点 (0,0,0)</button>
      <button id="btn-o--1" onclick="selOrigin(-1)">拟合盒中心</button>
    </div>
    <div class="cbrow" id="cBtns"></div>
  </div>

  <div class="sec">
    <h4>附加旋转（绕当前坐标轴，度）</h4>
    <div class="row"><label>绕X:</label><input type="number" id="rx" value="0"><button onclick="applyRot(0)">应用</button></div>
    <div class="row"><label>绕Y:</label><input type="number" id="ry" value="0"><button onclick="applyRot(1)">应用</button></div>
    <div class="row"><label>绕Z:</label><input type="number" id="rz" value="0"><button onclick="applyRot(2)">应用</button></div>
  </div>

  <div id="info">初始化中...</div>

  <div class="acrow">
    <button id="bReset" onclick="onReset()">重置</button>
    <button id="bFinish" onclick="onFinish()">完成 → 确认</button>
    <button id="bCancel" onclick="onCancel()">取消</button>
  </div>
</div>

<script>
/* ── 数据 ──────────────────────────────────────────────── */
const D = {DATA_JSON};
const pts=D.pts, planeQ=D.plane_quads, lineD=D.line_data;
const corners=D.corners, cen=D.centroid, pcaC=D.pca_cols;
const worldOrigin=D.world_origin||[0,0,0];
const axScale=D.axis_scale;
const nLines=lineD.length, nCorners=corners.length;

/* ── 交互状态 ──────────────────────────────────────────── */
let tgt=0, xLi=null, yLi=null, xFlp=false, yFlp=false;
let cusX=null, cusY=null;
let exRot=[[1,0,0],[0,1,0],[0,0,1]];
let orig=[...worldOrigin], selC=null;
let axisChanged=false, originChanged=false;

/* ── 颜色 ───────────────────────────────────────────────── */
const GC=['#e74c3c','#2ecc71','#3498db'];
const AC=['#e74c3c','#27ae60','#2980b9'];

/* ── Trace 索引常量 ──────────────────────────────────────── */
const T_CLOUD=0;
const T_PLANE=1;        // 1..6 (6条plane mesh3d)
const T_LINE=7;         // 7..9 (3 groups)
const T_LMID=10;        // line midpoints (clickable)
const T_CORNER=11;      // corners+centroid (clickable)
const T_AX=12;          // 12..14 axis lines
const T_ORIG=15;        // origin marker
const T_CLOUD_LOD=16;   // 拖动时使用的低密度点云

/* ── 数学工具 ──────────────────────────────────────────── */
const V={
  add:(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],
  sub:(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],
  sc: (v,s)=>[v[0]*s,v[1]*s,v[2]*s],
  dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
  cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],
  norm:(v)=>Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]),
  unit:(v)=>{const n=V.norm(v);return n>1e-12?V.sc(v,1/n):[1,0,0];}
};
function mm33(A,B){
  const C=[[0,0,0],[0,0,0],[0,0,0]];
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)C[i][j]+=A[i][k]*B[k][j];
  return C;
}
function gs3(xv,yv){
  xv=V.unit(xv);
  yv=V.sub(yv,V.sc(xv,V.dot(yv,xv)));
  if(V.norm(yv)<1e-10){const t=Math.abs(xv[2])<0.9?[0,0,1]:[0,1,0];yv=V.sub(t,V.sc(xv,V.dot(t,xv)));}
  yv=V.unit(yv);
  return [xv,yv,V.unit(V.cross(xv,yv))];
}
function rodrigues(axis,deg){
  const ax=V.unit(axis),t=deg*Math.PI/180,c=Math.cos(t),s=Math.sin(t);
  const[a0,a1,a2]=ax;
  return[[c+a0*a0*(1-c),a0*a1*(1-c)-a2*s,a0*a2*(1-c)+a1*s],
         [a1*a0*(1-c)+a2*s,c+a1*a1*(1-c),a1*a2*(1-c)-a0*s],
         [a2*a0*(1-c)-a1*s,a2*a1*(1-c)+a0*s,c+a2*a2*(1-c)]];
}
function computeR(){
  // 未选择时保持输入点云的实际 XYZ 方向，不自动套用拟合轴。
  let xv=cusX?[...cusX]:(xLi!==null?(xFlp?V.sc(lineD[xLi].direction,-1):[...lineD[xLi].direction]):[1,0,0]);
  let yv=cusY?[...cusY]:(yLi!==null?(yFlp?V.sc(lineD[yLi].direction,-1):[...lineD[yLi].direction]):[0,1,0]);
  const[x,y,z]=gs3(xv,yv);
  const Rb=[[x[0],y[0],z[0]],[x[1],y[1],z[1]],[x[2],y[2],z[2]]];
  return mm33(exRot,Rb);
}

/* ── 初始化 Plotly ──────────────────────────────────────── */
function buildTraces(){
  const traces=[];
  // 0: cloud
  traces.push({type:'scatter3d',x:pts.x,y:pts.y,z:pts.z,mode:'markers',
    marker:{size:1,color:'#aaaaaa',opacity:0.15},name:'点云',hoverinfo:'skip',showlegend:false});
  // 1-6: plane patches
  planeQ.forEach(q=>{
    const c=GC[q.group];
    traces.push({type:'mesh3d',x:q.x,y:q.y,z:q.z,
      i:[0,0],j:[1,2],k:[2,3],
      color:c,opacity:0.10,flatshading:true,showlegend:false,hoverinfo:'skip',
      lighting:{ambient:0.9,diffuse:0.1}});
  });
  // 7-9: line groups (scatter3d with nulls)
  for(let g=0;g<3;g++){
    const lx=[],ly=[],lz=[];
    lineD.filter(l=>l.group===g).forEach(l=>{lx.push(l.x[0],l.x[1],null);ly.push(l.y[0],l.y[1],null);lz.push(l.z[0],l.z[1],null);});
    traces.push({type:'scatter3d',x:lx,y:ly,z:lz,mode:'lines',
      line:{color:GC[g],width:3},name:'交线'+g,showlegend:false,hoverinfo:'skip'});
  }
  // 10: line midpoints (clickable circles)
  const lmx=lineD.map(l=>l.mid[0]),lmy=lineD.map(l=>l.mid[1]),lmz=lineD.map(l=>l.mid[2]);
  const lmc=lineD.map(l=>GC[l.group]);
  traces.push({type:'scatter3d',x:lmx,y:lmy,z:lmz,mode:'markers',
    marker:{size:10,color:lmc,symbol:'circle',opacity:1,
            line:{color:'#000',width:1}},
    name:'交线中点',showlegend:false,
    text:lineD.map((_,i)=>`交线${i}(组${lineD[i].group})`),hovertemplate:'%{text}<extra></extra>'});
  // 11: corners + centroid
  const cpts=[cen,...corners];
  const ccol=cpts.map((_,i)=>i===0?'#9c27b0':'#e67e22');
  const csym=cpts.map((_,i)=>i===0?'cross':'diamond');
  const ctext=cpts.map((_,i)=>i===0?'质心':`角${i-1}`);
  traces.push({type:'scatter3d',x:cpts.map(p=>p[0]),y:cpts.map(p=>p[1]),z:cpts.map(p=>p[2]),
    mode:'markers',marker:{size:8,color:ccol,symbol:csym,opacity:1,line:{color:'#000',width:1}},
    name:'角点',showlegend:false,
    text:ctext,hovertemplate:'%{text}<extra></extra>'});
  // 12-14: axis lines (placeholder, updated later)
  for(let i=0;i<3;i++){
    traces.push({type:'scatter3d',x:[0,0],y:[0,0],z:[0,0],mode:'lines+text',
      line:{color:AC[i],width:5},text:['','XYZ'[i]],
      textfont:{color:AC[i],size:14,family:'Microsoft YaHei'},
      textposition:'top center',showlegend:false,hoverinfo:'skip',name:'轴'+i});
  }
  // 15: origin marker
  traces.push({type:'scatter3d',x:[orig[0]],y:[orig[1]],z:[orig[2]],mode:'markers',
    marker:{size:10,color:'#c0392b',symbol:'cross',opacity:1,line:{color:'#fff',width:2}},
    showlegend:false,name:'原点',hovertemplate:'原点<extra></extra>'});
  // 16: 拖动专用低密度点云。预先上传 GPU，交互时只切换可见性。
  const lodStep=Math.max(1,Math.ceil(pts.x.length/20000));
  const lodX=[],lodY=[],lodZ=[];
  for(let i=0;i<pts.x.length;i+=lodStep){lodX.push(pts.x[i]);lodY.push(pts.y[i]);lodZ.push(pts.z[i]);}
  traces.push({type:'scatter3d',x:lodX,y:lodY,z:lodZ,mode:'markers',
    marker:{size:1.5,color:'#9e9e9e',opacity:0.35},name:'点云(拖动LOD)',
    hoverinfo:'skip',showlegend:false,visible:false});
  return traces;
}

const layout={
  autosize:true,
  hovermode:'closest',
  uirevision:'coord-frame-camera',
  paper_bgcolor:'#fafafa',
  plot_bgcolor:'#fafafa',
  margin:{l:0,r:0,t:28,b:0},
  title:{text:'3D视图  ◎=交线中点(点击选轴)  ◆=角点(点击选原点)',font:{size:11},x:0.5},
  scene:{
    aspectmode:'data',
    xaxis:{title:'X',showgrid:true,gridcolor:'#ddd'},
    yaxis:{title:'Y',showgrid:true,gridcolor:'#ddd'},
    zaxis:{title:'Z',showgrid:true,gridcolor:'#ddd'},
    bgcolor:'#f8f8f8',
  }
};
const config={responsive:true,displaylogo:false,plotGlPixelRatio:1,
  modeBarButtonsToRemove:['resetCameraLastSave3d']};

let fastMode=false;
let restoreTimer=null;
let dragStart=null;
function setFastMode(enabled){
  if(fastMode===enabled)return;
  fastMode=enabled;
  // 原子切换两个 trace，避免两次重绘之间出现一帧空白。
  Plotly.restyle('plot',{visible:[!enabled,enabled]},[T_CLOUD,T_CLOUD_LOD]);
}
function rememberPointerDown(event){
  if(event.button!==0)return;
  dragStart={x:event.clientX,y:event.clientY};
  if(restoreTimer){clearTimeout(restoreTimer);restoreTimer=null;}
}
function enableLodAfterDrag(event){
  if(!dragStart||fastMode)return;
  const dx=event.clientX-dragStart.x,dy=event.clientY-dragStart.y;
  // 普通点击不能 restyle，否则 Plotly 的 plotly_click 会被重绘取消。
  if(dx*dx+dy*dy>=25)setFastMode(true);
}
function scheduleFullQuality(){
  dragStart=null;
  if(restoreTimer)clearTimeout(restoreTimer);
  restoreTimer=setTimeout(()=>{setFastMode(false);restoreTimer=null;},80);
}

Plotly.newPlot('plot',buildTraces(),layout,config).then(()=>{
  updateDynamic();
  buildCornerButtons();
  updateInfo();
  const el=document.getElementById('plot');
  el.on('plotly_click',onPlotClick);
  el.addEventListener('pointerdown',rememberPointerDown,{passive:true});
  el.addEventListener('pointermove',enableLodAfterDrag,{passive:true});
  window.addEventListener('pointerup',scheduleFullQuality,{passive:true});
  window.addEventListener('pointercancel',scheduleFullQuality,{passive:true});
  window.addEventListener('blur',scheduleFullQuality,{passive:true});
  el.addEventListener('mouseleave',event=>{if(event.buttons===0)scheduleFullQuality();},{passive:true});
});

/* ── 动态更新轴 & 原点 ─────────────────────────────────── */
function updateDynamic(){
  const R=computeR();
  const upd={x:[],y:[],z:[]};
  for(let i=0;i<3;i++){
    const col=[R[0][i],R[1][i],R[2][i]];
    const end=V.add(orig,V.sc(col,axScale));
    Plotly.restyle('plot',{x:[[orig[0],end[0]]],y:[[orig[1],end[1]]],z:[[orig[2],end[2]]]},[T_AX+i]);
  }
  Plotly.restyle('plot',{x:[[orig[0]]],y:[[orig[1]]],z:[[orig[2]]]},[T_ORIG]);
  updateMidColors();
  updateCornerColors();
}

function updateMidColors(){
  const mc=lineD.map((_,i)=>i===xLi?'#c0392b':(i===yLi?'#1a9e3f':GC[lineD[i].group]));
  const ms=lineD.map((_,i)=>(i===xLi||i===yLi)?16:10);
  Plotly.restyle('plot',{'marker.color':[mc],'marker.size':[ms]},[T_LMID]);
}

function updateCornerColors(){
  const cpts=[cen,...corners];
  const cc=cpts.map((_,i)=>{
    const idx=i-1;
    if(selC===-1&&i===0)return'#c0392b';
    if(idx===selC)return'#c0392b';
    return i===0?'#9c27b0':'#e67e22';
  });
  Plotly.restyle('plot',{'marker.color':[cc]},[T_CORNER]);
}

/* ── 点击处理 ──────────────────────────────────────────── */
function onPlotClick(data){
  if(!data.points||!data.points.length)return;
  const pt=data.points[0];
  const ci=pt.curveNumber, pi=pt.pointIndex;
  if(ci===T_LMID){
    selectLine(pi);
  }else if(ci===T_CORNER){
    const idx=pi-1; // -1=centroid, 0..7=corners
    selC=idx;originChanged=true;
    orig=idx===-1?[...cen]:[...corners[idx]];
    console.log(idx===-1?`原点→质心`:`原点→角${idx}`);
    updateAll();
  }
}

/* ── UI 操作 ────────────────────────────────────────────── */
function selectLine(pi){
  if(pi<0||pi>=lineD.length)return;
  if(tgt===0){
    if(yLi!==null&&lineD[yLi].group===lineD[pi].group){
      alert('X轴和Y轴不能选择平行交线，请选择其他颜色方向。');return;
    }
    xLi=pi;cusX=null;axisChanged=true;
    console.log(`X轴→交线${pi}(组${lineD[pi].group})`);
    setTarget(1); // 设置X后自动切换到Y
  }else{
    if(xLi!==null&&lineD[xLi].group===lineD[pi].group){
      alert('Y轴和X轴不能选择平行交线，请选择其他颜色方向。');return;
    }
    yLi=pi;cusY=null;axisChanged=true;
    console.log(`Y轴→交线${pi}(组${lineD[pi].group})`);
  }
  exRot=[[1,0,0],[0,1,0],[0,0,1]];
  updateAll();
}

function selectLineGroup(group){
  const pi=lineD.findIndex(line=>line.group===group);
  selectLine(pi);
}

function setTarget(ax){
  tgt=ax;
  document.getElementById('bSetX').textContent=ax===0?'[当前] 设置X轴':'设置X轴';
  document.getElementById('bSetY').textContent=ax===1?'[当前] 设置Y轴':'设置Y轴';
  document.getElementById('bSetX').className='btn-ax'+(ax===0?' active-x':'');
  document.getElementById('bSetY').className='btn-ay'+(ax===1?' active-y':'');
}

function setFlip(ax,flip){
  if(ax===0)xFlp=flip;else yFlp=flip;
  axisChanged=true;
  exRot=[[1,0,0],[0,1,0],[0,0,1]];
  updateAll();
}

function setCustom(ax){
  const raw=document.getElementById(ax===0?'cxIn':'cyIn').value.trim();
  const parts=raw.replace(/;/g,',').split(',').map(Number);
  if(parts.length!==3||parts.some(isNaN)){alert('请输入3个数字，如: 1,0,0');return;}
  const n=V.norm(parts);if(n<1e-10){alert('零向量');return;}
  const v=V.sc(parts,1/n);
  if(ax===0){cusX=v;xLi=null;}else{cusY=v;yLi=null;}
  axisChanged=true;
  exRot=[[1,0,0],[0,1,0],[0,0,1]];
  updateAll();
}

function clearCustom(ax){
  if(ax===0)cusX=null;else cusY=null;
  axisChanged=true;
  exRot=[[1,0,0],[0,1,0],[0,0,1]];
  updateAll();
}

function selWorldOrigin(){
  selC=null;originChanged=false;orig=[...worldOrigin];
  updateAll();
}

function selOrigin(idx){
  selC=idx;originChanged=true;orig=idx===-1?[...cen]:[...corners[idx]];
  updateAll();
}

function applyRot(ax){
  const ang=parseFloat(document.getElementById(['rx','ry','rz'][ax]).value);
  if(isNaN(ang)||Math.abs(ang)<1e-9)return;
  const R=computeR();
  const col=[R[0][ax],R[1][ax],R[2][ax]];
  exRot=mm33(rodrigues(col,ang),exRot);
  axisChanged=true;
  document.getElementById(['rx','ry','rz'][ax]).value=0;
  updateAll();
}

function onReset(){
  tgt=0;xLi=null;yLi=null;xFlp=false;yFlp=false;
  cusX=null;cusY=null;exRot=[[1,0,0],[0,1,0],[0,0,1]];
  orig=[...worldOrigin];selC=null;axisChanged=false;originChanged=false;
  ['rx','ry','rz'].forEach(id=>document.getElementById(id).value=0);
  ['cxIn','cyIn'].forEach(id=>document.getElementById(id).value='');
  setTarget(0);
  updateAll();
}

function onFinish(){
  const R=computeR();
  const payload={R:R,origin:orig,axisChanged:axisChanged,originChanged:originChanged};
  fetch('/result',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)})
  .then(()=>{document.body.innerHTML='<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-size:20px;color:#2e7d32;font-family:Microsoft YaHei">✔ 坐标系已确认，请切回终端继续...</div>';})
  .catch(e=>alert('发送失败: '+e));
}

function onCancel(){
  fetch('/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
  .then(()=>{document.body.innerHTML='<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-size:18px;color:#888;font-family:Microsoft YaHei">已取消，使用原始坐标系</div>';})
  .catch(()=>{});
}

/* ── 统一更新 ──────────────────────────────────────────── */
function updateAll(){updateDynamic();updateStateLabels();updateInfo();updateCornerBtnColors();}

function updateStateLabels(){
  const xl=cusX?'X轴: 自定义':(xLi!==null?`X轴: 交线${xLi}(组${lineD[xLi].group}) ${xFlp?'- 反向':'+ 正向'}`:'X轴: 原始X方向');
  const yl=cusY?'Y轴: 自定义':(yLi!==null?`Y轴: 交线${yLi}(组${lineD[yLi].group}) ${yFlp?'- 反向':'+ 正向'}`:'Y轴: 原始Y方向');
  document.getElementById('xStat').textContent=xl;
  document.getElementById('yStat').textContent=yl;
}

function updateInfo(){
  const R=computeR();
  const fmt=(v)=>v.map(x=>(x>=0?'+':'')+x.toFixed(4)).join(', ');
  const on=selC===-1?'质心':(selC!==null?`角${selC}`:'原始坐标原点');
  let s=`坐标系:\n`;
  s+=` X: [${fmt([R[0][0],R[1][0],R[2][0]])}]\n`;
  s+=` Y: [${fmt([R[0][1],R[1][1],R[2][1]])}]\n`;
  s+=` Z: [${fmt([R[0][2],R[1][2],R[2][2]])}]\n`;
  s+=`原点(${on}):\n [${orig.map(v=>v.toFixed(4)).join(', ')}]`;
  document.getElementById('info').textContent=s;
}

function buildCornerButtons(){
  const div=document.getElementById('cBtns');
  corners.forEach((_,i)=>{
    const b=document.createElement('button');
    b.id=`btn-o-${i}`;b.textContent=`角${i}`;b.onclick=()=>selOrigin(i);div.appendChild(b);
  });
}

function updateCornerBtnColors(){
  const worldBtn=document.getElementById('btn-o-world');
  if(worldBtn)worldBtn.className=(!originChanged&&selC===null)?'sel-btn':'';
  const ids=[-1,...Array(corners.length).keys()];
  ids.forEach(i=>{
    const b=document.getElementById(`btn-o-${i}`);if(!b)return;
    b.className=selC===i?'sel-btn':'';
  });
}
</script>
</body>
</html>"""


def interactive_coord_frame_window(detector: 'BallDetector'):
    """
    HTML/Plotly.js 版坐标系编辑器（前置步骤）。
    启动本地 HTTP 服务器，打开浏览器，用户完成设置后返回 (R, origin)。
    """
    import threading as _threading
    import webbrowser as _wb
    import json as _json
    import time as _time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    if detector.points is None or len(detector.points) == 0:
        print('  无点云数据，跳过坐标系编辑。')
        return None, None, False, False

    print('\n[坐标系编辑] RANSAC 拟合长方体主平面 ...')
    pca_axes, centroid, plane_pairs, corners, lines = fit_box_planes_pca(detector.points)

    for i, lbl in enumerate(['e0(长边方向)', 'e1(中边方向)', 'e2(短边方向)']):
        v = pca_axes[:, i]
        print(f'  PCA {lbl}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]')
    for i, pp in enumerate(plane_pairs):
        print(f'  面对{i}(法向=e{i}): 厚度={pp["extent"]:.4f}')
    diagnostics = plane_pairs[0].get('diagnostics', {})
    if diagnostics:
        inliers = diagnostics.get('plane_inliers', ())
        residuals = diagnostics.get('median_residuals', ())
        threshold = diagnostics.get('distance_threshold', float('nan'))
        print(f'  RANSAC阈值={threshold:.6f}  主平面内点={inliers}')
        print(f'  主平面中位残差={tuple(round(v, 6) for v in residuals)}')
    print(f'  角点={len(corners)}个  交线={len(lines)}条')

    # ── 降采样点云 ──────────────────────────────────────────────────────────
    pts = detector.points
    step = max(1, len(pts) // 80000)  # 80K 采样，清晰显示点云形态
    pts_ds = pts[::step]

    # ── 预计算面片顶点 ──────────────────────────────────────────────────────
    plane_quads = []
    for pp_i, pp in enumerate(plane_pairs):
        other = [j for j in range(3) if j != pp_i]
        u_ax = pca_axes[:, other[0]]
        v_ax = pca_axes[:, other[1]]
        for d in [pp['d_lo'], pp['d_hi']]:
            _n = pp['normal']
            # 所有面片共用稳健包围盒中心，避免采样不均导致质心偏移。
            _box_center = pp.get('box_center', centroid)
            c = _box_center - (float(np.dot(_n, _box_center)) - d) * _n
            ea, ev = pp['ext_a'], pp['ext_b']
            v0 = c + ea * u_ax + ev * v_ax
            v1 = c - ea * u_ax + ev * v_ax
            v2 = c - ea * u_ax - ev * v_ax
            v3 = c + ea * u_ax - ev * v_ax
            plane_quads.append({
                'x': [round(float(x), 4) for x in [v0[0], v1[0], v2[0], v3[0]]],
                'y': [round(float(x), 4) for x in [v0[1], v1[1], v2[1], v3[1]]],
                'z': [round(float(x), 4) for x in [v0[2], v1[2], v2[2], v3[2]]],
                'group': pp_i,
            })

    # ── 预计算交线端点 ──────────────────────────────────────────────────────
    ext_half = [pp['extent'] * 0.55 for pp in plane_pairs]
    line_data = []
    for li, (direction, pt, group_k, _) in enumerate(lines):
        t = ext_half[group_k]
        p1 = pt - direction * t
        p2 = pt + direction * t
        line_data.append({
            'x': [round(float(p1[0]), 4), round(float(p2[0]), 4)],
            'y': [round(float(p1[1]), 4), round(float(p2[1]), 4)],
            'z': [round(float(p1[2]), 4), round(float(p2[2]), 4)],
            'mid': [round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)],
            'direction': [round(float(direction[0]), 6),
                          round(float(direction[1]), 6),
                          round(float(direction[2]), 6)],
            'group': int(group_k),
            'idx': li,
        })

    data = {
        'pts': {
            'x': [round(float(v), 4) for v in pts_ds[:, 0]],
            'y': [round(float(v), 4) for v in pts_ds[:, 1]],
            'z': [round(float(v), 4) for v in pts_ds[:, 2]],
        },
        'plane_quads': plane_quads,
        'line_data':   line_data,
        'corners':     [[round(float(v), 4) for v in c] for c in corners],
        'centroid':    [round(float(v), 6) for v in centroid],
        'world_origin': [0.0, 0.0, 0.0],
        'pca_cols':    [[round(float(pca_axes[row, col]), 6) for row in range(3)]
                        for col in range(3)],
        'axis_scale':  float(min(pp['extent'] for pp in plane_pairs) * 0.35),
    }

    # 防止 </script> 注入
    data_json = _json.dumps(data).replace('</', '<\\/')
    html = _COORD_HTML.replace('{DATA_JSON}', data_json)

    # ── HTTP 服务器 ─────────────────────────────────────────────────────────
    result = {'R': None, 'origin': None, 'axis_changed': False,
              'origin_changed': False, 'done': False}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                body = html.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            body = _json.loads(self.rfile.read(n))
            if self.path == '/result':
                result['R']      = body.get('R')
                result['origin'] = body.get('origin')
                result['axis_changed'] = bool(body.get('axisChanged', False))
                result['origin_changed'] = bool(body.get('originChanged', False))
                result['done']   = True
            elif self.path == '/cancel':
                result['done'] = True
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_):
            pass  # 静默

    server = HTTPServer(('127.0.0.1', 0), _Handler)
    port = server.server_address[1]
    _threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f'http://127.0.0.1:{port}/'
    print(f'\n[坐标系编辑] 浏览器地址: {url}')
    print('  · 点击3D中 ◎ 圆点 → 分配给X或Y轴')
    print('  · 点击 ◆ 角点   → 选为坐标原点')
    print('  · 点击 [完成→确认] → 关闭浏览器并继续')
    _wb.open(url)

    try:
        while not result['done']:
            _time.sleep(0.1)
    except KeyboardInterrupt:
        print('\n  已中断，使用原始坐标系继续。')
        server.shutdown()
        return None, None, False, False

    server.shutdown()

    if result['R'] is None:
        print('  坐标系编辑已取消，使用原始坐标系。')
        return None, None, False, False

    R      = np.array(result['R'],      dtype=np.float64)
    origin = np.array(result['origin'], dtype=np.float64)
    if R.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("浏览器返回的坐标变换尺寸无效。")
    if not np.isfinite(R).all() or not np.isfinite(origin).all():
        raise ValueError("浏览器返回的坐标变换包含非有限数值。")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0.999:
        raise ValueError("浏览器返回的坐标轴不是有效右手正交坐标系。")

    # 不信任前端状态标志：根据实际矩阵和原点再次判定是否发生变换。
    axis_changed = not np.allclose(R, np.eye(3), atol=1e-8)
    origin_changed = not np.allclose(origin, np.zeros(3), atol=1e-8)

    o = origin
    print(f'\n  坐标系已确认:')
    print(f'  原点: [{o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}]')
    for ai in range(3):
        v = R[:, ai]
        print(f'  {"XYZ"[ai]}轴: [{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]')

    return R, origin, axis_changed, origin_changed


def _export_transformed_las(detector: 'BallDetector', output_file: str):
    """将 detector.points（已变换到新坐标系）和强度导出为 LAS 文件。"""
    pts  = detector.points
    ints = detector.intensities
    print(f'\n[导出] 变换后完整点云 → {output_file}  ({len(pts):,} 个点) ...')
    hdr = laspy.LasHeader(point_format=0, version='1.2')
    las_out = laspy.LasData(header=hdr)
    las_out.x = pts[:, 0].astype(np.float64)
    las_out.y = pts[:, 1].astype(np.float64)
    las_out.z = pts[:, 2].astype(np.float64)
    las_out.intensity = ints.astype(np.uint16)
    las_out.write(output_file)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    print(f'  X=[{x.min():.4f},{x.max():.4f}]  '
          f'Y=[{y.min():.4f},{y.max():.4f}]  Z=[{z.min():.4f},{z.max():.4f}]')
    print(f'  已保存 → {output_file}')


def _export_transformed_pcd(detector: 'BallDetector', output_file: str,
                            chunk_size: int = 500_000):
    """将当前坐标系下的完整点云按块写为 binary PCD（XYZ + intensity）。"""
    pts = np.asarray(detector.points, dtype=np.float64)
    intensities = np.asarray(detector.intensities, dtype=np.float64)
    if len(pts) != len(intensities):
        raise ValueError("点云坐标与强度数量不一致，无法导出 PCD。")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("点云坐标必须是 N×3 数组。")

    count = len(pts)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {count}\n"
        "DATA binary\n"
    ).encode("ascii")

    print(f"\n[导出] 变换后完整点云 PCD → {output_file}  ({count:,} 个点) ...")
    with open(output_file, "wb") as stream:
        stream.write(header)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            block = np.empty((stop - start, 4), dtype="<f4")
            block[:, :3] = pts[start:stop]
            block[:, 3] = intensities[start:stop]
            stream.write(block.tobytes(order="C"))
    print(f"  已保存 → {output_file}")


def _export_cluster_las(detector: 'BallDetector', output_file: str):
    """将每个有效聚类的小球球心导出为 LAS 文件。"""
    active = {k: v for k, v in detector.clusters.items()
              if k not in detector.removed_clusters}
    if not active:
        print('  无可用聚类，跳过聚类球心 LAS 导出。')
        return
    labels = sorted(active)
    centers = np.vstack([active[label]['center'] for label in labels])
    intensities = np.array([
        float(np.max(active[label].get('intensities', np.array([0]))))
        for label in labels
    ], dtype=np.float64)
    print(f'\n[导出] 聚类球心 LAS → {output_file}  ({len(centers):,} 个球心) ...')
    hdr = laspy.LasHeader(point_format=0, version='1.2')
    las_out = laspy.LasData(header=hdr)
    las_out.x = centers[:, 0].astype(np.float64)
    las_out.y = centers[:, 1].astype(np.float64)
    las_out.z = centers[:, 2].astype(np.float64)
    las_out.intensity = intensities.astype(np.uint16)
    las_out.write(output_file)
    print(f'  已保存 → {output_file}')


# ─────────────────────────────────────────────────────────────────────────────
# 交互式强度阈值选择窗口
# ─────────────────────────────────────────────────────────────────────────────

def interactive_threshold_window(detector: BallDetector) -> float:
    """弹出强度阈值交互选择窗口，返回用户确认的阈值。"""

    init_thresh = float(np.percentile(detector.intensities, 90))
    selected = [init_thresh]
    confirmed = [False]

    fig, (ax_hist, ax_3d) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("交互式强度阈值选择", fontsize=13, fontweight="bold")
    plt.subplots_adjust(bottom=0.22, wspace=0.35)

    # 直方图
    ax_hist.hist(detector.intensities, bins=300, color="#4C72B0", alpha=0.7)
    ax_hist.set_xlabel("强度值"); ax_hist.set_ylabel("点数")
    ax_hist.set_title("强度分布直方图")
    vline = ax_hist.axvline(init_thresh, color="red", lw=2, ls="--")
    info_text = ax_hist.text(0.98, 0.97, "", transform=ax_hist.transAxes,
                              ha="right", va="top", fontsize=9,
                              bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    # 3D 预览（将 ax_3d 替换为 3D axes）
    ax_hist_pos = ax_hist.get_position()
    ax_3d.remove()
    ax3d = fig.add_subplot(122, projection="3d")

    def _update_info(thresh):
        cnt = int((detector.intensities >= thresh).sum())
        info_text.set_text(f"阈值: {thresh:.0f}\n保留: {cnt:,} 点")

    def _preview(thresh):
        ax3d.cla()
        mask = detector.intensities >= thresh
        pts = detector.points[mask]
        ints = detector.intensities[mask]
        if len(pts) > 0:
            step = max(1, len(pts) // 15000)
            sc = ax3d.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                              c=ints[::step], cmap="hot", s=2, alpha=0.8)
        ax3d.set_title(f"过滤后点云预览\n({len(pts):,} 点)", fontsize=9)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        fig.canvas.draw_idle()

    # 滑块
    ax_slider = plt.axes([0.12, 0.10, 0.55, 0.03])
    slider = Slider(ax_slider, "强度阈值",
                    float(detector.intensities.min()),
                    float(detector.intensities.max()),
                    valinit=init_thresh, valstep=1.0)

    def _on_slider(val):
        vline.set_xdata([val, val])
        _update_info(val)
        _preview(val)

    slider.on_changed(_on_slider)

    # 确认按钮
    ax_ok = plt.axes([0.70, 0.08, 0.12, 0.06])
    btn_ok = Button(ax_ok, "确认")

    def _on_ok(_):
        selected[0] = slider.val
        confirmed[0] = True
        plt.close(fig)

    btn_ok.on_clicked(_on_ok)

    _update_info(init_thresh)
    _preview(init_thresh)
    plt.show()

    if not confirmed[0]:
        print("  窗口关闭，使用默认阈值。")
    return selected[0]


# ─────────────────────────────────────────────────────────────────────────────
# 自动估算 DBSCAN eps
# ─────────────────────────────────────────────────────────────────────────────

def estimate_eps(points: np.ndarray, k: int = 4) -> float:
    """
    使用 k-NN 距离曲线拐点法估算 DBSCAN eps。
    取第 k 近邻距离的 95 百分位值作为起始建议值。
    """
    from sklearn.neighbors import NearestNeighbors
    n = min(len(points), 5000)
    sample = points[np.random.choice(len(points), n, replace=False)]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(sample)
    dists, _ = nbrs.kneighbors(sample)
    knn_dists = np.sort(dists[:, k])
    # 95 百分位，避免被少量极端拖尾值拉偏
    eps = float(np.percentile(knn_dists, 95))
    return max(eps, 1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LAS 高反小球点云检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 打开桌面工作台，再在各功能页选择点云
  python las_ball_detector.py

  # 打开桌面工作台，并预先加载一份点云
  python las_ball_detector.py scan.las

  # 指定强度阈值 + 自动聚类
  python las_ball_detector.py scan.las -t 3000 --no-viz

  # 指定强度阈值 + 自定义聚类参数
  python las_ball_detector.py scan.las -t 3000 -e 0.05 -m 10 --no-viz

  # 仅导出，不弹出交互窗口
  python las_ball_detector.py scan.las -t 3000 --no-viz -o result
""")
    p.add_argument("las_file", nargs="?", default=None,
                   help="LAS/LAZ 文件路径；桌面工作台可省略，批处理 --no-viz 必填")
    p.add_argument("-t", "--threshold", type=float, default=None,
                   help="强度阈值（不指定时使用百分位数自动计算）")
    p.add_argument("-p", "--percentile", type=float, default=90,
                   help="自动阈值百分位数，默认 90")
    p.add_argument("-e", "--eps", type=float, default=None,
                   help="DBSCAN 搜索半径（默认：自动估算）")
    p.add_argument("-m", "--min-samples", type=int, default=5,
                   help="DBSCAN 最小点数，默认 5")
    p.add_argument("-i", "--interactive-threshold", action="store_true",
                   help="弹出滑块窗口交互式选择强度阈值")
    p.add_argument("--no-viz", action="store_true",
                   help="跳过所有可视化，直接导出结果")
    p.add_argument("-o", "--output", default="ball_centers",
                   help="输出 CSV 文件前缀，默认 ball_centers")
    p.add_argument("--coord-frame", action="store_true",
                   help="前置：先编辑坐标系(交线选轴)，再过滤/聚类，最后导出变换后LAS+聚类LAS")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    detector = BallDetector()

    # ── 1. 加载 ───────────────────────────────────────────────────────────────
    if args.las_file:
        detector.load_las(args.las_file)
    elif args.no_viz:
        parser.error("--no-viz 批处理模式必须提供 las_file；桌面工作台可省略。")

    # 所有交互步骤统一进入本地 Qt/OpenGL 工作台；批处理模式保持原流程。
    if not args.no_viz:
        try:
            from desktop_app import run_unified_desktop
        except ImportError as exc:
            raise RuntimeError(
                "桌面界面依赖未安装，请执行: pip install PySide6 pyqtgraph PyOpenGL"
            ) from exc
        return run_unified_desktop(detector, args, sys.modules[__name__])

    _R_new, _origin_new = None, None
    _axis_changed, _origin_changed = False, False
    if not args.no_viz and args.coord_frame:
        print("\n[前置] 启动坐标系编辑（强度过滤/聚类之前）...")
        (_R_new, _origin_new,
         _axis_changed, _origin_changed) = interactive_coord_frame_window(detector)
        if _R_new is not None and (_axis_changed or _origin_changed):
            print("\n[变换] 将全量点云变换到新坐标系 ...")
            pts_t = ((_R_new.T) @ (detector.points - _origin_new).T).T
            detector.points = pts_t.astype(np.float64)
            x_, y_, z_ = (detector.points[:, 0],
                           detector.points[:, 1],
                           detector.points[:, 2])
            print(f"  完成，{len(detector.points):,} 个点已变换至新坐标系")
            print(f"  X=[{x_.min():.4f},{x_.max():.4f}]  "
                  f"Y=[{y_.min():.4f},{y_.max():.4f}]  "
                  f"Z=[{z_.min():.4f},{z_.max():.4f}]")
        else:
            print("  使用原始坐标系继续。")

    # ── 2. 确定强度阈值 ───────────────────────────────────────────────────────
    if args.interactive_threshold and not args.no_viz:
        print("\n[2/4] 启动交互式阈值选择窗口 ...")
        threshold = interactive_threshold_window(detector)
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = float(np.percentile(detector.intensities, args.percentile))
        print(f"\n[2/4] 使用第 {args.percentile} 百分位数作为阈值: {threshold:.1f}")

    # ── 3. 强度过滤 ───────────────────────────────────────────────────────────
    detector.filter_by_intensity(threshold)

    if not args.no_viz:
        detector.visualize_intensity_filter()

    # ── 4. 确定 eps ───────────────────────────────────────────────────────────
    eps = args.eps
    if eps is None:
        if detector.filtered_points is not None and len(detector.filtered_points) >= 5:
            print("\n      自动估算 eps (k-NN 距离法) ...")
            eps = estimate_eps(detector.filtered_points)
            print(f"      建议 eps = {eps:.6f}  (可用 -e 参数手动指定覆盖)")
        else:
            eps = 0.5

    # ── 5. 聚类 ───────────────────────────────────────────────────────────────
    detector.cluster(eps=eps, min_samples=args.min_samples)

    if not detector.clusters:
        print("\n未找到任何聚类，建议:\n"
              "  · 降低强度阈值 (-t)\n"
              "  · 增大 eps (-e)\n"
              "  · 减小 min-samples (-m)")
        sys.exit(1)

    # ── 6. 交互式编辑 + 导出 ──────────────────────────────────────────────────
    if not args.no_viz:
        print("\n[4/4] 启动交互式聚类编辑窗口 ...")
        print("      操作说明:")
        print("        · 点击 3D 视图或俯视图中的 ★ → 标记/取消剔除")
        print("        · [确认剔除] → 永久删除已标记聚类")
        print("        · [重置选择] → 清除所有标记（不删除）")
        print("        · [导出 CSV] → 保存当前结果")
        print("        · 文本框输入 ID（如 2,5,7）再点 [按 ID 剔除] → 直接删除")
        detector.visualize_clusters_interactive()
    else:
        print("\n[4/4] 跳过可视化。")

    detector.export_results(args.output)

    # ── 7. 导出变换后 LAS（--coord-frame 且用户已确认变换时）
    if _R_new is not None and (_axis_changed or _origin_changed):
        _export_transformed_las(detector, args.output + "_cloud.las")
        _export_cluster_las(detector,     args.output + "_clusters.las")


if __name__ == "__main__":
    main()
