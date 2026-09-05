#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一的 Qt/OpenGL 点云处理界面。"""

from pathlib import Path
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


MAX_VIEW_POINTS = 120_000
MAX_CLUSTER_VIEW_POINTS = 180_000
GROUP_COLORS = ((0.91, 0.30, 0.24, 1.0), (0.18, 0.80, 0.44, 1.0),
                (0.20, 0.60, 0.86, 1.0))


def _sample_indices(count, limit):
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _colors_from_intensity(values, alpha=0.85):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.empty((0, 4), dtype=np.float32)
    lo, hi = np.percentile(values, (2, 98))
    t = np.clip((values - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    colors = np.empty((len(values), 4), dtype=np.float32)
    colors[:, 0] = np.clip(1.7 * t, 0, 1)
    colors[:, 1] = np.clip(1.7 - np.abs(t - 0.55) * 3.0, 0, 1)
    colors[:, 2] = np.clip(1.4 * (1.0 - t), 0, 1)
    colors[:, 3] = alpha
    return colors


def _read_las_points(filepath, scale=1.0):
    pts, _intensities = _read_las_cloud(filepath, scale=scale)
    return pts


def _read_las_cloud(filepath, scale=1.0):
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"点云文件不存在：{filepath}")

    with path.open("rb") as file:
        signature = file.read(4)
    if signature == b"CCB2":
        raise ValueError(
            "选择的文件是 CloudCompare BIN2 格式（文件签名 CCB2），不是 LAS/LAZ 点云。\n\n"
            "请先在 CloudCompare 中转换为 LAS 后再加载：\n"
            'CloudCompare.exe -SILENT -O "input.bin" -C_EXPORT_FMT LAS -SAVE_CLOUDS'
        )
    if signature[:3].lower() == b"pcd":
        raise ValueError("选择的文件是 PCD 格式；当前此入口只支持 LAS/LAZ，请先转换为 LAS。")

    import laspy as _lp
    try:
        las = _lp.read(str(path))
    except Exception as exc:
        raise ValueError(f"无法读取 LAS/LAZ 文件：{path.name}\n{exc}") from exc
    count = len(las.points)
    if count <= 0:
        raise ValueError(f"点云文件没有有效点：{filepath}")

    pts = np.empty((count, 3), dtype=np.float64)
    pts[:, 0] = las.x
    pts[:, 1] = las.y
    pts[:, 2] = las.z
    if scale != 1.0:
        pts *= scale
    intensities = np.asarray(las.intensity, dtype=np.float64)
    return pts, intensities


def _read_pcd_cloud(filepath, scale=1.0):
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"点云文件不存在：{filepath}")

    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD 文件缺少 DATA 头：{path.name}")
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                data_offset = stream.tell()
                break

    fields = [name.lower() for name in (header.get("FIELDS") or header.get("FIELD") or [])]
    if not all(name in fields for name in ("x", "y", "z")):
        raise ValueError(f"PCD 文件必须包含 x、y、z 字段：{path.name}")
    try:
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        point_count = int((header.get("POINTS") or [str(
            int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0])
        )])[0])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"PCD 文件头无效：{path.name}") from exc
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"PCD 字段定义数量不一致：{path.name}")
    if any(count <= 0 for count in counts):
        raise ValueError(f"PCD 字段 COUNT 必须为正整数：{path.name}")

    type_map = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
        ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    }
    try:
        dtype_fields = []
        for name, kind, size, count in zip(fields, types, sizes, counts):
            scalar_dtype = type_map[(kind.upper(), size)]
            dtype_fields.append((name, scalar_dtype) if count == 1
                                else (name, scalar_dtype, (count,)))
        dtype = np.dtype(dtype_fields)
    except KeyError as exc:
        raise ValueError(f"PCD 包含不支持的字段类型：{exc.args[0]}") from exc

    with path.open("rb") as stream:
        stream.seek(data_offset)
        data_kind = header["DATA"][0].lower()
        if data_kind == "binary":
            records = np.fromfile(stream, dtype=dtype, count=point_count)
        elif data_kind == "ascii":
            matrix = np.loadtxt(stream, dtype=np.float64, ndmin=2, max_rows=point_count)
            if matrix.shape[1] != sum(counts):
                raise ValueError(f"PCD 数据列数与 SIZE/COUNT 定义不一致：{path.name}")
            records = np.empty(len(matrix), dtype=dtype)
            column = 0
            for name, count in zip(fields, counts):
                records[name] = (matrix[:, column] if count == 1
                                 else matrix[:, column:column + count])
                column += count
        else:
            raise ValueError("暂不支持 binary_compressed PCD，请转换为 binary 或 ascii PCD。")
    if len(records) != point_count:
        raise ValueError(f"PCD 点数不完整：声明 {point_count:,}，实际 {len(records):,}。")

    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
    if scale != 1.0:
        points *= scale
    intensities = (np.asarray(records["intensity"], dtype=np.float64)
                   if "intensity" in fields else np.zeros(len(points), dtype=np.float64))
    return points, intensities


def _read_point_cloud(filepath, scale=1.0):
    if Path(filepath).suffix.lower() == ".pcd":
        return _read_pcd_cloud(filepath, scale=scale)
    return _read_las_cloud(filepath, scale=scale)

def _split_trc_fields(line):
    if "\t" in line:
        return [part.strip() for part in line.strip().split("\t") if part.strip() != ""]
    return [part.strip() for part in line.strip().split() if part.strip() != ""]


def _read_trc_first_frame_points(filepath, scale=1.0):
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"TRC 文件不存在：{filepath}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise ValueError("TRC 文件不完整，至少需要表头和一行帧数据。")

    data_start = 4
    while data_start < len(lines) and not lines[data_start].strip():
        data_start += 1
    if data_start >= len(lines):
        raise ValueError("TRC 文件没有有效帧数据。")

    row = _split_trc_fields(lines[data_start])
    if len(row) <= 3:
        raise ValueError("TRC 第一帧列数不足，无法提取 XYZ 点。")

    values = row[3:]
    usable = (len(values) // 3) * 3
    if usable <= 0:
        raise ValueError("TRC 第一帧没有可用 XYZ 三元组。")

    coords = np.array([float(v) for v in values[:usable]], dtype=np.float64).reshape(-1, 3)
    if scale != 1.0:
        coords *= scale
    return coords


def _read_source_points(filepath, scale=1.0):
    if Path(filepath).suffix.lower() == ".trc":
        return _read_trc_first_frame_points(filepath, scale=scale)
    return _read_point_cloud(filepath, scale=scale)[0]


class TaskWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.function())
        except Exception:
            self.failed.emit(traceback.format_exc())


class PointCloudView(gl.GLViewWidget):
    gpu_ready = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((20, 24, 28))
        self.opts["distance"] = 30
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2,
                                            color=(0.4, 0.7, 1.0, 0.8), pxMode=True)
        self.center_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=13,
                                                   color=(1, 1, 1, 1), pxMode=True)
        self.addItem(self.scatter)
        self.addItem(self.center_scatter)
        self.guides = []
        self.frame_guides = []
        self.origin_marker = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=18,
                                                  color=(1.0, 0.85, 0.05, 1.0), pxMode=True)
        self.addItem(self.origin_marker)
        self.display_origin = np.zeros(3)
        # 对比 / 配准用额外散点（默认空）
        self.compare_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=2,
                                                     color=(1.0, 0.55, 0.0, 0.75), pxMode=True)
        self.reg_source_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=3,
                                                        color=(1.0, 0.2, 0.2, 0.85), pxMode=True)
        self.reg_target_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=6,
                                                        color=(0.2, 0.9, 0.3, 1.0), pxMode=True)
        self.addItem(self.compare_scatter)
        self.addItem(self.reg_source_scatter)
        self.addItem(self.reg_target_scatter)
        self._transform_mode   = False
        self._tm_last_pos      = None
        self._tm_rotate_cb     = None
        self._tm_translate_cb  = None
        self._tm_release_cb    = None
        self._cluster_hover = []
        self.setMouseTracking(True)

    def initializeGL(self):
        super().initializeGL()
        try:
            from OpenGL.GL import GL_RENDERER, glGetString
            renderer = glGetString(GL_RENDERER)
            self.gpu_ready.emit(renderer.decode("utf-8", errors="replace") if renderer else "OpenGL")
        except Exception:
            self.gpu_ready.emit("OpenGL（渲染器信息不可用）")

    def set_cloud(self, points, intensities=None, colors=None, limit=MAX_VIEW_POINTS, size=2):
        if points is None or not len(points):
            self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return
        ids = _sample_indices(len(points), limit)
        sampled = np.asarray(points[ids], dtype=np.float64)
        self.display_origin = np.median(sampled, axis=0)
        positions = (sampled - self.display_origin).astype(np.float32)
        if colors is None:
            if intensities is None:
                draw_colors = np.tile((0.35, 0.70, 0.95, 0.75), (len(ids), 1))
            else:
                draw_colors = _colors_from_intensity(np.asarray(intensities)[ids])
        else:
            draw_colors = np.asarray(colors)[ids]
        self.scatter.setData(pos=positions, color=draw_colors, size=size, pxMode=True)
        span = np.ptp(positions, axis=0)
        self.opts["distance"] = max(float(np.linalg.norm(span)) * 1.3, 1.0)
        self.opts["center"] = pg.Vector(0, 0, 0)
        self.update()

    def clear_frame(self):
        for item in self.frame_guides:
            self.removeItem(item)
        self.frame_guides.clear()
        self.origin_marker.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def clear_guides(self):
        for item in self.guides:
            self.removeItem(item)
        self.guides.clear()
        self.clear_frame()
        self.center_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._cluster_hover = []

    def show_coordinate_frame(self, origin, rotation, scale):
        self.clear_frame()
        origin = np.asarray(origin, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        colors = ((1.0, 0.12, 0.10, 1.0), (0.10, 1.0, 0.30, 1.0),
                  (0.15, 0.50, 1.0, 1.0))
        for axis_index, color in enumerate(colors):
            endpoint = origin + rotation[:, axis_index] * scale
            positions = np.vstack((origin, endpoint)) - self.display_origin
            item = gl.GLLinePlotItem(pos=positions.astype(np.float32), color=color,
                                     width=6, antialias=True)
            self.addItem(item)
            self.frame_guides.append(item)
        marker_position = (origin - self.display_origin).reshape(1, 3).astype(np.float32)
        self.origin_marker.setData(pos=marker_position, color=(1.0, 0.85, 0.05, 1.0), size=22)
        self.update()

    def show_box(self, lines, corners, highlight_groups=()):
        self.clear_guides()
        for direction, point, group, _ in lines:
            corner_projection = np.asarray(corners) @ direction
            half = max(float(np.ptp(corner_projection)) * 0.55, 1e-3)
            endpoints = np.vstack((point - direction * half, point + direction * half))
            selected = group in highlight_groups
            base_color = GROUP_COLORS[group]
            # 高亮时保持原色、完全不透明；非高亮时降低透明度到 25%
            color = base_color if selected else (base_color[0], base_color[1], base_color[2], 0.25)
            item = gl.GLLinePlotItem(pos=(endpoints - self.display_origin).astype(np.float32),
                                     color=color, width=2, antialias=True)
            self.addItem(item)
            self.guides.append(item)
        corner_pos = (np.asarray(corners) - self.display_origin).astype(np.float32)
        self.center_scatter.setData(pos=corner_pos, color=(1.0, 0.82, 0.25, 1.0), size=11)

    def show_cluster_centers(self, centers, removed):
        if not len(centers):
            self.center_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._cluster_hover = []
            return
        colors = np.array([(0.95, 0.25, 0.20, 1.0) if label in removed
                           else (1.0, 0.88, 0.18, 1.0) for label, _ in centers], dtype=np.float32)
        world_centers = np.array([center for _, center in centers], dtype=np.float64)
        positions = world_centers - self.display_origin
        self.center_scatter.setData(pos=positions.astype(np.float32), color=colors, size=14)
        self._cluster_hover = [
            (int(label), np.asarray(center, dtype=np.float64), positions[i].astype(np.float64))
            for i, (label, center) in enumerate(centers)
        ]

    def _project_to_screen(self, view_pos):
        try:
            vec = QtGui.QVector4D(float(view_pos[0]), float(view_pos[1]), float(view_pos[2]), 1.0)
            clip = self.projectionMatrix() * self.viewMatrix() * vec
            if abs(clip.w()) < 1e-9:
                return None
            ndc_x = clip.x() / clip.w()
            ndc_y = clip.y() / clip.w()
            if not (-1.2 <= ndc_x <= 1.2 and -1.2 <= ndc_y <= 1.2):
                return None
            return QtCore.QPointF(
                (ndc_x + 1.0) * 0.5 * self.width(),
                (1.0 - ndc_y) * 0.5 * self.height(),
            )
        except Exception:
            return None

    def _update_cluster_hover(self, ev):
        if not self._cluster_hover:
            QtWidgets.QToolTip.hideText()
            return
        mouse_pos = ev.position() if hasattr(ev, "position") else ev.pos()
        best = None
        best_d2 = 18.0 * 18.0
        for label, world, view_pos in self._cluster_hover:
            screen = self._project_to_screen(view_pos)
            if screen is None:
                continue
            dx = screen.x() - mouse_pos.x()
            dy = screen.y() - mouse_pos.y()
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best = (label, world)
                best_d2 = d2
        if best is None:
            QtWidgets.QToolTip.hideText()
            return
        label, world = best
        global_pos = ev.globalPosition().toPoint() if hasattr(ev, "globalPosition") else ev.globalPos()
        QtWidgets.QToolTip.showText(
            global_pos,
            f"聚类 ID: {label}\nX={world[0]:.4f}\nY={world[1]:.4f}\nZ={world[2]:.4f}",
            self,
        )


    # ── 对比视图 ──────────────────────────────────────────────────────────────

    def set_compare_cloud(self, points, error_colors=None):
        """显示第二份点云，可按误差着色。"""
        if points is None or not len(points):
            self.compare_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return
        ids = _sample_indices(len(points), MAX_VIEW_POINTS)
        sampled = (np.asarray(points[ids], dtype=np.float64) - self.display_origin).astype(np.float32)
        if error_colors is not None:
            c = np.asarray(error_colors)[ids].astype(np.float32)
        else:
            c = np.tile((1.0, 0.55, 0.0, 0.75), (len(ids), 1)).astype(np.float32)
        self.compare_scatter.setData(pos=sampled, color=c, size=2, pxMode=True)
        self.update()

    def clear_compare(self):
        self.compare_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.update()

    def set_display_origin_from_points(self, points):
        if points is None or not len(points):
            self.display_origin = np.zeros(3)
            return
        ids = _sample_indices(len(points), MAX_VIEW_POINTS)
        sampled = np.asarray(points[ids], dtype=np.float64)
        self.display_origin = np.median(sampled, axis=0)
        span = np.ptp((sampled - self.display_origin).astype(np.float32), axis=0)
        self.opts["distance"] = max(float(np.linalg.norm(span)) * 1.3, 1.0)
        self.opts["center"] = pg.Vector(0, 0, 0)

    def clear_primary_cloud(self):
        self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.center_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._cluster_hover = []
        self.update()

    # ── 配准视图 ──────────────────────────────────────────────────────────────

    def set_reg_source(self, points, color=(1.0, 0.2, 0.2, 0.85), size=3):
        """显示源点云（红色）。"""
        if points is None or not len(points):
            self.reg_source_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.update()
            return
        ids = _sample_indices(len(points), MAX_VIEW_POINTS)
        sampled = (np.asarray(points[ids], dtype=np.float64) - self.display_origin).astype(np.float32)
        self.reg_source_scatter.setData(pos=sampled, color=color, size=size, pxMode=True)
        self.update()

    def set_reg_target(self, points, size=6):
        """显示目标点云（绿色聚类点）。"""
        if points is None or not len(points):
            self.reg_target_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.update()
            return
        ids = _sample_indices(len(points), MAX_CLUSTER_VIEW_POINTS)
        sampled = (np.asarray(points[ids], dtype=np.float64) - self.display_origin).astype(np.float32)
        self.reg_target_scatter.setData(pos=sampled, color=(0.2, 0.9, 0.3, 1.0), size=size, pxMode=True)
        self.update()

    def clear_reg(self):
        self.reg_source_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.reg_target_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.update()
    def set_transform_mode(self, enabled, rotate_cb=None,
                           translate_cb=None, release_cb=None):
        self._transform_mode  = enabled
        self._tm_rotate_cb    = rotate_cb
        self._tm_translate_cb = translate_cb
        self._tm_release_cb   = release_cb
        self._tm_last_pos     = None
        c = (QtCore.Qt.CursorShape.SizeAllCursor
             if enabled else QtCore.Qt.CursorShape.ArrowCursor)
        self.setCursor(c)

    def mousePressEvent(self, ev):
        if self._transform_mode:
            self._tm_last_pos = ev.pos()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._transform_mode and self._tm_last_pos is not None:
            dx = ev.pos().x() - self._tm_last_pos.x()
            dy = ev.pos().y() - self._tm_last_pos.y()
            self._tm_last_pos = ev.pos()
            shift = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            right = bool(ev.buttons() & QtCore.Qt.MouseButton.RightButton)
            if shift or right:
                if self._tm_translate_cb:
                    self._tm_translate_cb(dx, dy)
            else:
                if self._tm_rotate_cb:
                    self._tm_rotate_cb(dx, dy)
            ev.accept()
        else:
            self._update_cluster_hover(ev)
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._transform_mode:
            self._tm_last_pos = None
            if self._tm_release_cb:
                self._tm_release_cb()
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)



class UnifiedWindow(QtWidgets.QMainWindow):
    _task_success_requested = QtCore.Signal(object, object)
    _task_failure_requested = QtCore.Signal(str, object)

    def __init__(self, detector, args, api):
        super().__init__()
        self.detector = detector
        if self.detector.points is None:
            self.detector.points = np.empty((0, 3), dtype=np.float64)
        if self.detector.intensities is None:
            self.detector.intensities = np.empty((0,), dtype=np.float64)
        self.args = args
        self.api = api
        self.axes = None
        self.centroid = None
        self.corners = np.empty((0, 3))
        self.lines = []
        self.transform_applied = False
        self.frame_scale = 1.0
        # 已应用的坐标系变换（供对比Tab使用相同变换）
        self._applied_rotation = np.eye(3)
        self._applied_origin = np.zeros(3)
        # 角点表面贴合补偿结果缓存
        self._last_snap_devs = (0.0, 0.0, 0.0)  # (x_dev, y_dev, z_dev)
        # 点云对比 Tab 状态
        self.compare_cloud_pts = None
        self.compare1_pts      = None
        self.compare1_intensities = None
        self.compare2_raw_pts  = None        # 原始加载的第二份点云
        self.compare2_pts      = None        # 坐标系变换后
        self.compare2_axes     = None
        self.compare2_centroid = None
        self.compare2_corners  = np.empty((0, 3))
        self.compare2_lines    = []
        # 点云配准 Tab 状态
        self.reg_source_pts = None          # 原始加载的红色点云（已乘缩放）
        self.reg_target_pts = None          # 单独加载的绿色目标点云
        self.reg_show_source = True
        self.reg_show_target = True
        self.reg_manual_rot = np.eye(3)     # 累积的手动旋转矩阵
        self.reg_translation     = np.zeros(3)   # 累积的手动平移
        self.reg_interactive_R   = np.eye(3)     # 交互中的旋转增量
        self.reg_interactive_t   = np.zeros(3)   # 交互中的平移增量
        self.reg_view_mode = "free"
        self.current_rotation = np.eye(3)
        self.current_origin = np.zeros(3)
        self.threads = []
        self.workers = []
        self._task_success_requested.connect(
            self._task_succeeded,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self._task_failure_requested.connect(
            self._task_failed_with_callback,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self.setWindowTitle("LAS 高反球检测 - GPU 工作台")
        self.resize(1500, 900)
        self._build_ui()
        self._set_cloud(self.detector.points, self.detector.intensities)
        if self._has_cloud():
            self._fit_coordinate_geometry()
        else:
            self.coord_info.setText("未加载点云。可在功能一/二/三/四对应页面选择 LAS/LAZ 文件。")
            self.status_label.setText("未加载点云")

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("LAS 高反球检测")
        title.setObjectName("title")
        status = f"已加载 {len(self.detector.points):,} 个点" if len(self.detector.points) else "未加载点云"
        self.status_label = QtWidgets.QLabel(status)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_label)
        layout.addLayout(header)

        splitter = QtWidgets.QSplitter()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(390)
        self.tabs.setMaximumWidth(480)
        self.view = PointCloudView()
        self.view.gpu_ready.connect(lambda renderer: self.status_label.setText(f"GPU: {renderer}"))
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._build_coordinate_tab()
        self._build_threshold_tab()
        self._build_cluster_tab()
        self._build_export_tab()
        self._build_compare_tab()
        self._build_registration_tab()
        self.tabs.currentChanged.connect(self._tab_changed)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.setStyleSheet("""
            QMainWindow, QWidget { background:#f4f5f2; color:#22282c; font-family:'Microsoft YaHei UI'; font-size:13px; }
            QLabel#title { font-family:'Microsoft YaHei UI'; font-size:22px; font-weight:700; color:#162126; }
            QTabWidget::pane { border:1px solid #c8cdca; background:#fff; }
            QTabBar::tab { padding:10px 14px; background:#e2e5e2; }
            QTabBar::tab:selected { background:#fff; color:#006b5d; font-weight:700; }
            QGroupBox { border:1px solid #d3d7d4; margin-top:12px; padding-top:12px; font-weight:700; }
            QPushButton { background:#fff; border:1px solid #aeb6b2; padding:7px 11px; }
            QPushButton:hover { border-color:#00796b; color:#006b5d; }
            QPushButton#primary { background:#00796b; color:#fff; border-color:#00796b; font-weight:700; }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background:#fff; border:1px solid #b9c0bc; padding:5px; }
            QTableWidget { background:#fff; alternate-background-color:#f1f4f2; gridline-color:#d9ddda; }
        """)

    def _page(self):
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        box.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        return page, box

    def _build_coordinate_tab(self):
        page, box = self._page()
        intro = QtWidgets.QLabel("RANSAC 拟合盒体后，从三个非平行方向中定义新坐标轴。右侧 OpenGL 视图可直接旋转缩放。")
        intro.setWordWrap(True)
        box.addWidget(intro)

        load_group = QtWidgets.QGroupBox("打开点云")
        load_form = QtWidgets.QFormLayout(load_group)
        coord_file_row = QtWidgets.QHBoxLayout()
        self.coord_file_edit = QtWidgets.QLineEdit()
        coord_file_browse = QtWidgets.QPushButton("浏览...")
        coord_file_browse.clicked.connect(self._browse_coordinate_file)
        coord_file_row.addWidget(self.coord_file_edit)
        coord_file_row.addWidget(coord_file_browse)
        load_form.addRow("LAS 文件", coord_file_row)
        coord_load = QtWidgets.QPushButton("打开点云并拟合坐标系")
        coord_load.setObjectName("primary")
        coord_load.clicked.connect(self._load_coordinate_cloud)
        load_form.addRow("", coord_load)
        box.addWidget(load_group)

        axis_group = QtWidgets.QGroupBox("坐标轴")
        form = QtWidgets.QFormLayout(axis_group)
        def _group_icon(r, g, b):
            pix = QtGui.QPixmap(14, 14)
            pix.fill(QtGui.QColor(int(r*255), int(g*255), int(b*255)))
            return QtGui.QIcon(pix)
        _icons = [
            _group_icon(0.91, 0.30, 0.24),
            _group_icon(0.18, 0.80, 0.44),
            _group_icon(0.20, 0.60, 0.86),
        ]
        self.x_axis = QtWidgets.QComboBox()
        self.y_axis = QtWidgets.QComboBox()
        for combo in (self.x_axis, self.y_axis):
            combo.addItem(_icons[0], "红色方向（组 0）")
            combo.addItem(_icons[1], "绿色方向（组 1）")
            combo.addItem(_icons[2], "蓝色方向（组 2）")
            combo.addItem("自定义")
        # 默认保持实际坐标方向；用户可明确切换到拟合交线方向。
        self.x_axis.setCurrentIndex(3)
        self.y_axis.setCurrentIndex(3)
        self.x_sign = QtWidgets.QComboBox(); self.x_sign.addItems(["正向", "反向"])
        self.y_sign = QtWidgets.QComboBox(); self.y_sign.addItems(["正向", "反向"])
        self.x_custom = QtWidgets.QLineEdit("1,0,0")
        self.y_custom = QtWidgets.QLineEdit("0,1,0")
        self.x_dir_label = QtWidgets.QLabel("—")
        self.y_dir_label = QtWidgets.QLabel("—")
        self.x_dir_label.setStyleSheet("color:#555; font-size:11px;")
        self.y_dir_label.setStyleSheet("color:#555; font-size:11px;")
        form.addRow("X 方向", self.x_axis); form.addRow("X 正反向", self.x_sign)
        form.addRow("X 自定义", self.x_custom); form.addRow("X 实际方向", self.x_dir_label)
        form.addRow("Y 方向", self.y_axis); form.addRow("Y 正反向", self.y_sign)
        form.addRow("Y 自定义", self.y_custom); form.addRow("Y 实际方向", self.y_dir_label)
        box.addWidget(axis_group)
        self.axis_warning = QtWidgets.QLabel()
        self.axis_warning.setWordWrap(True)
        self.axis_warning.setStyleSheet("color:#c0392b; font-weight:bold; padding:4px;")
        box.addWidget(self.axis_warning)

        origin_group = QtWidgets.QGroupBox("原点与附加旋转")
        origin_form = QtWidgets.QFormLayout(origin_group)
        self.origin_combo = QtWidgets.QComboBox()
        self.origin_combo.addItems(["世界原点 (0,0,0)", "拟合盒中心"] + [f"角点 {i}" for i in range(8)])
        self.rot_x = QtWidgets.QDoubleSpinBox(); self.rot_y = QtWidgets.QDoubleSpinBox(); self.rot_z = QtWidgets.QDoubleSpinBox()
        for spin in (self.rot_x, self.rot_y, self.rot_z):
            spin.setRange(-360, 360); spin.setDecimals(2); spin.setSuffix("°")
        origin_form.addRow("坐标原点", self.origin_combo)
        origin_form.addRow("绕 X 旋转", self.rot_x); origin_form.addRow("绕 Y 旋转", self.rot_y)
        origin_form.addRow("绕 Z 旋转", self.rot_z)
        # 角点表面贴合：自动补偿地面/墙面不平整导致的原点偏差
        self.snap_check = QtWidgets.QCheckBox("启用（补偿地面/墙面不平整）")
        self.snap_check.setChecked(True)
        self.snap_check.setToolTip(
            "选择角点为原点时，自动搜索角点附近实际点云表面，\n"
            "计算各轴偏差并补偿原点位置，使实际表面点位于坐标零点。\n"
            "解决水泥地面凹凸不平导致的 Z=0.02 等偏差问题。")
        self.snap_radius = QtWidgets.QDoubleSpinBox()
        self.snap_radius.setRange(0.02, 2.0)
        self.snap_radius.setValue(0.15)
        self.snap_radius.setSuffix(" m")
        self.snap_radius.setSingleStep(0.01)
        self.snap_radius.setDecimals(3)
        origin_form.addRow("角点表面贴合", self.snap_check)
        origin_form.addRow("贴合搜索半径", self.snap_radius)
        self.snap_info = QtWidgets.QLabel("")
        self.snap_info.setWordWrap(True)
        self.snap_info.setStyleSheet("color:#0F6E56; font-size:11px;")
        origin_form.addRow("", self.snap_info)
        box.addWidget(origin_group)
        for widget in (self.x_axis, self.y_axis, self.x_sign, self.y_sign, self.origin_combo):
            widget.currentIndexChanged.connect(self._preview_coordinate_frame)
        for widget in (self.x_custom, self.y_custom):
            widget.editingFinished.connect(self._preview_coordinate_frame)
        for widget in (self.rot_x, self.rot_y, self.rot_z, self.snap_radius):
            widget.valueChanged.connect(lambda _v: self._preview_coordinate_frame())
        self.snap_check.stateChanged.connect(lambda _v: self._preview_coordinate_frame())

        self.apply_coord = QtWidgets.QPushButton("应用坐标系")
        self.apply_coord.setObjectName("primary")
        self.apply_coord.clicked.connect(self._apply_coordinate_frame)
        box.addWidget(self.apply_coord)
        export_group = QtWidgets.QGroupBox("坐标系转换结果")
        export_form = QtWidgets.QFormLayout(export_group)
        export_row = QtWidgets.QHBoxLayout()
        self.coord_output_edit = QtWidgets.QLineEdit(f"{self.args.output}_cloud.las")
        coord_browse = QtWidgets.QPushButton("另存为...")
        coord_browse.clicked.connect(self._browse_coord_output)
        export_row.addWidget(self.coord_output_edit)
        export_row.addWidget(coord_browse)
        export_form.addRow("点云输出", export_row)
        coord_export = QtWidgets.QPushButton("导出当前坐标系点云")
        coord_export.setObjectName("primary")
        coord_export.clicked.connect(self._export_coordinate_las)
        export_form.addRow("", coord_export)
        box.addWidget(export_group)
        self.coord_info = QtWidgets.QLabel("正在拟合盒体平面...")
        self.coord_info.setWordWrap(True)
        box.addWidget(self.coord_info)
        self.tabs.addTab(page, "1 坐标系")
        # axis_warning 在 _build_coordinate_tab 里已创建，此处无需重复添加

    def _build_threshold_tab(self):
        page, box = self._page()
        load_group = QtWidgets.QGroupBox("打开点云")
        load_form = QtWidgets.QFormLayout(load_group)
        file_row = QtWidgets.QHBoxLayout()
        self.analysis_file_edit = QtWidgets.QLineEdit()
        analysis_browse = QtWidgets.QPushButton("浏览...")
        analysis_browse.clicked.connect(self._browse_analysis_file)
        file_row.addWidget(self.analysis_file_edit)
        file_row.addWidget(analysis_browse)
        load_form.addRow("LAS 文件", file_row)
        analysis_load = QtWidgets.QPushButton("打开点云用于强度过滤和聚类")
        analysis_load.clicked.connect(self._load_analysis_cloud)
        load_form.addRow("", analysis_load)
        box.addWidget(load_group)
        self.histogram = pg.PlotWidget(background="w")
        self.histogram.setMinimumHeight(220)
        self.histogram.setLabel("bottom", "强度")
        self.histogram.setLabel("left", "点数")
        box.addWidget(self.histogram)
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        if self._has_intensities():
            self.threshold_slider.setRange(int(self.detector.intensities.min()), int(self.detector.intensities.max()))
            initial = int(np.percentile(self.detector.intensities, self.args.percentile))
        else:
            self.threshold_slider.setRange(0, 65535)
            initial = 0
        self.threshold_slider.setValue(initial)
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(self.threshold_slider.minimum(), self.threshold_slider.maximum())
        self.threshold_spin.setValue(initial)
        self.threshold_slider.valueChanged.connect(self.threshold_spin.setValue)
        self.threshold_spin.valueChanged.connect(self.threshold_slider.setValue)
        self.threshold_slider.valueChanged.connect(self._preview_threshold)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("强度阈值")); row.addWidget(self.threshold_spin)
        box.addLayout(row); box.addWidget(self.threshold_slider)
        self.keep_label = QtWidgets.QLabel()
        box.addWidget(self.keep_label)
        button = QtWidgets.QPushButton("应用过滤并进入聚类")
        button.setObjectName("primary"); button.clicked.connect(self._apply_threshold)
        box.addWidget(button)
        self.threshold_line = None
        if self._has_intensities():
            self._reset_intensity_widgets(initial)
        else:
            self.keep_label.setText("未加载点云")
        self._preview_threshold(initial)
        self.tabs.addTab(page, "2 强度")

    def _build_cluster_tab(self):
        page, box = self._page()
        params = QtWidgets.QGroupBox("DBSCAN 参数")
        form = QtWidgets.QFormLayout(params)
        self.eps_spin = QtWidgets.QDoubleSpinBox(); self.eps_spin.setDecimals(6); self.eps_spin.setRange(0.000001, 1e6)
        self.eps_spin.setValue(self.args.eps or 0.05)
        self.auto_eps = QtWidgets.QCheckBox("自动估算"); self.auto_eps.setChecked(self.args.eps is None)
        self.min_samples = QtWidgets.QSpinBox(); self.min_samples.setRange(2, 9999); self.min_samples.setValue(self.args.min_samples)
        form.addRow("eps", self.eps_spin); form.addRow("", self.auto_eps); form.addRow("最小点数", self.min_samples)
        box.addWidget(params)
        run = QtWidgets.QPushButton("运行聚类")
        run.setObjectName("primary"); run.clicked.connect(self._run_clustering)
        box.addWidget(run)
        self.cluster_summary = QtWidgets.QLabel("尚未运行聚类")
        box.addWidget(self.cluster_summary)
        self.cluster_table = QtWidgets.QTableWidget(0, 9)
        self.cluster_table.setHorizontalHeaderLabels(["保留", "ID", "点数", "强度范围", "X", "Y", "Z", "样本点号", "备注"])
        self.cluster_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.cluster_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.cluster_table.itemChanged.connect(self._cluster_check_changed)
        box.addWidget(self.cluster_table, 1)
        self.tabs.addTab(page, "3 聚类审核")

    def _build_export_tab(self):
        page, box = self._page()
        self.output_edit = QtWidgets.QLineEdit(self.args.output)
        box.addWidget(QtWidgets.QLabel("输出文件前缀")); box.addWidget(self.output_edit)
        self.export_csv = QtWidgets.QCheckBox("聚类中心 CSV"); self.export_csv.setChecked(True)
        self.export_cloud = QtWidgets.QCheckBox("新坐标系完整点云 LAS"); self.export_cloud.setChecked(True)
        self.export_pcd = QtWidgets.QCheckBox("新坐标系完整点云 PCD（二进制）"); self.export_pcd.setChecked(True)
        self.export_clusters = QtWidgets.QCheckBox("审核后聚类点 LAS"); self.export_clusters.setChecked(True)
        self.close_after_export = QtWidgets.QCheckBox("导出完成后关闭程序")
        for widget in (self.export_csv, self.export_cloud, self.export_pcd,
                       self.export_clusters, self.close_after_export):
            box.addWidget(widget)
        button_row = QtWidgets.QHBoxLayout()
        export = QtWidgets.QPushButton("导出结果")
        export.setObjectName("primary"); export.clicked.connect(self._export)
        self.close_button = QtWidgets.QPushButton("关闭程序")
        self.close_button.setEnabled(False); self.close_button.clicked.connect(self.close)
        button_row.addWidget(export); button_row.addWidget(self.close_button)
        box.addLayout(button_row)
        self.export_info = QtWidgets.QLabel()
        self.export_info.setWordWrap(True); box.addWidget(self.export_info)
        self.tabs.addTab(page, "4 导出")

    def _set_busy(self, message):
        self.status_label.setText(message)
        self.progress.setRange(0, 0)
        self.tabs.setEnabled(False)

    def _set_ready(self, message):
        self.status_label.setText(message)
        self.progress.setRange(0, 1); self.progress.setValue(1)
        self.tabs.setEnabled(True)

    def _run_task(self, function, on_success, on_failure=None):
        thread = QtCore.QThread(self)
        worker = TaskWorker(function)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda result, callback=on_success: self._task_success_requested.emit(result, callback)
        )
        worker.finished.connect(thread.quit)
        failure_callback = on_failure or self._task_failed
        worker.failed.connect(
            lambda details, callback=failure_callback: self._task_failure_requested.emit(details, callback)
        )
        worker.failed.connect(thread.quit)
        def release():
            if thread in self.threads:
                self.threads.remove(thread)
            if worker in self.workers:
                self.workers.remove(worker)
        thread.finished.connect(release)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.threads.append(thread)
        self.workers.append(worker)
        thread.start()

    def _task_succeeded(self, result, on_success):
        try:
            on_success(result)
        except Exception:
            self._task_failed(traceback.format_exc())

    def _task_failed_with_callback(self, details, on_failure):
        on_failure(details)

    def _task_failed(self, details):
        self._set_ready("操作失败")
        QtWidgets.QMessageBox.critical(self, "操作失败", details)

    def _set_cloud(self, points, intensities=None, colors=None, limit=MAX_VIEW_POINTS, size=2):
        self.view.set_cloud(points, intensities, colors, limit, size)

    def _has_cloud(self):
        return self.detector.points is not None and len(self.detector.points) > 0

    def _has_intensities(self):
        return self.detector.intensities is not None and len(self.detector.intensities) > 0

    def _browse_coord_output(self):
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存坐标系转换点云", self.coord_output_edit.text().strip(),
            "LAS 点云 (*.las);;PCD 点云 (*.pcd)")
        if path:
            if Path(path).suffix.lower() not in (".las", ".pcd"):
                path += ".pcd" if selected_filter.startswith("PCD") else ".las"
            self.coord_output_edit.setText(path)

    def _browse_coordinate_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开用于坐标系转换的点云", "", "LAS/LAZ 点云 (*.las *.laz)")
        if path:
            self.coord_file_edit.setText(path)

    def _load_coordinate_cloud(self):
        filepath = self.coord_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择点云文件。")
            return
        self._set_busy("正在打开点云...")

        def work():
            return _read_las_cloud(filepath)

        self._run_task(work, self._coordinate_cloud_loaded)

    def _coordinate_cloud_loaded(self, result):
        points, intensities = result
        self._replace_current_cloud(points, intensities)
        if hasattr(self, "analysis_file_edit"):
            self.analysis_file_edit.setText(self.coord_file_edit.text().strip())
        self.coord_info.setText(f"已打开点云：{len(points):,} 个点，正在自动拟合盒体...")
        self._fit_coordinate_geometry()

    def _fit_coordinate_geometry(self):
        if not self._has_cloud():
            self.coord_info.setText("未加载点云，无法自动拟合盒体。")
            return
        self._set_busy("正在 RANSAC 拟合盒体...")
        self._run_task(
            lambda: self.api.fit_box_planes_pca(self.detector.points),
            self._fit_done,
            self._fit_failed,
        )

    def _fit_done(self, result):
        self.axes, self.centroid, plane_pairs, self.corners, self.lines = result
        self.frame_scale = max(float(np.max(np.ptp(self.corners, axis=0))) * 0.22, 0.1)
        # 先刷新显示原点（点云中位数），确保盒体和坐标轴基于当前点云偏移绘制。
        self._set_cloud(self.detector.points, self.detector.intensities)
        self._preview_coordinate_frame()  # 内部调用 show_box 并高亮已选组
        diag = plane_pairs[0].get("diagnostics", {}) if plane_pairs else {}
        self.coord_info.setText(f"拟合完成：8 个角点，12 条交线；RANSAC 阈值 {diag.get('distance_threshold', 0):.6f}")
        self._set_ready("坐标系拟合完成")

    def _fit_failed(self, details):
        self.axes = None
        self.centroid = None
        self.corners = np.empty((0, 3))
        self.lines = []
        self._set_cloud(self.detector.points, self.detector.intensities)
        self.coord_info.setText(
            "自动拟合盒体失败。可继续使用默认原始坐标系，或在自定义 X/Y 方向和世界原点下直接进入强度过滤。"
        )
        self.axis_warning.setText("自动拟合未完成：当前不可选择红/绿/蓝拟合方向。")
        self._set_ready("自动拟合失败，已保持原始坐标系")

    def _direction(self, combo, custom, sign):
        index = combo.currentIndex()
        if index < 3:
            vector = np.asarray(self.lines[index * 4][0], dtype=float)
        else:
            parts = [float(part.strip()) for part in custom.text().replace("，", ",").split(",")]
            if len(parts) != 3:
                raise ValueError("自定义方向必须是三个逗号分隔的数字。")
            vector = np.asarray(parts, dtype=float)
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("坐标轴方向不能为零向量。")
        return vector / norm * (-1 if sign.currentIndex() else 1)

    def _compute_surface_snap(self, origin, rotation, radius=0.15, points=None):
        """计算角点附近实际点云表面与拟合平面的偏差，返回补偿后的原点。

        在新坐标系中，角点附近点来自三个表面：
          - 地面：Z ≈ floor_dev（新Z最小）
          - 墙面1：X ≈ wall1_dev（新|X|最小）
          - 墙面2：Y ≈ wall2_dev（新|Y|最小）
        每个点归属 |坐标| 最小的那个轴所在表面。
        取各表面点的坐标中位数作为偏差，修正原点使实际表面位于坐标零点。

        返回 (corrected_origin, (x_dev, y_dev, z_dev))。
        """
        if points is None:
            points = self.detector.points
        if points is None or len(points) == 0:
            return origin, (0.0, 0.0, 0.0)

        # 分块扫描全量点云，只保留角点邻域。全局等步长降采样会让大型点云
        # 的局部邻域只剩十几个点，进而静默跳过表面补偿。
        pts = np.asarray(points)
        radius_sq = radius * radius
        nearby_blocks = []
        nearby_count = 0
        for start in range(0, len(pts), 500_000):
            block = np.asarray(pts[start:start + 500_000], dtype=np.float64)
            diff = block - origin
            mask = np.einsum("ij,ij->i", diff, diff) < radius_sq
            if np.any(mask):
                selected = block[mask]
                nearby_blocks.append(selected)
                nearby_count += len(selected)
        if nearby_count < 20:
            return origin, (0.0, 0.0, 0.0)

        nearby = np.concatenate(nearby_blocks, axis=0)
        # 变换到新坐标系
        nearby_new = (rotation.T @ (nearby - origin).T).T
        abs_coords = np.abs(nearby_new)

        # 每个点归属 |坐标| 最小的轴（0=X墙面, 1=Y墙面, 2=Z地面）
        min_axis = np.argmin(abs_coords, axis=1)

        x_dev = y_dev = z_dev = 0.0
        # Z 表面（地面）
        z_pts = nearby_new[min_axis == 2]
        if len(z_pts) > 5:
            z_dev = float(np.median(z_pts[:, 2]))
        # X 表面（墙面1）
        x_pts = nearby_new[min_axis == 0]
        if len(x_pts) > 5:
            x_dev = float(np.median(x_pts[:, 0]))
        # Y 表面（墙面2）
        y_pts = nearby_new[min_axis == 1]
        if len(y_pts) > 5:
            y_dev = float(np.median(y_pts[:, 1]))

        # 在原始坐标系中修正原点：origin + R @ [x_dev, y_dev, z_dev]
        devs = np.array([x_dev, y_dev, z_dev])
        if np.linalg.norm(devs) < 1e-7:
            return origin, (0.0, 0.0, 0.0)
        corrected = origin + rotation @ devs
        return corrected, (x_dev, y_dev, z_dev)

    def _selected_frame(self):
        x = self._direction(self.x_axis, self.x_custom, self.x_sign)
        y = self._direction(self.y_axis, self.y_custom, self.y_sign)
        if abs(float(np.dot(x, y))) > 0.95:
            raise ValueError("X 和 Y 轴不能平行，请选择不同方向。")
        rotation = self.api._gs3(np.column_stack((x, y, np.cross(x, y))))
        for axis_index, angle in enumerate((self.rot_x.value(), self.rot_y.value(), self.rot_z.value())):
            rotation = self.api._rot_mat(rotation[:, axis_index], angle) @ rotation
            rotation = self.api._gs3(rotation)
        origin_index = self.origin_combo.currentIndex()
        origin = np.zeros(3) if origin_index == 0 else (self.centroid if origin_index == 1 else self.corners[origin_index - 2])
        origin = np.asarray(origin, dtype=np.float64)
        # 角点表面贴合补偿：选择角点为原点且启用时，自动补偿表面偏差
        if self.snap_check.isChecked() and origin_index >= 2:
            radius = self.snap_radius.value()
            origin, devs = self._compute_surface_snap(origin, rotation, radius)
            self._last_snap_devs = devs
        else:
            self._last_snap_devs = (0.0, 0.0, 0.0)
        return rotation, origin

    def _direction_label(self, combo, custom, sign):
        """返回 (vector, error_str)；不抛异常。"""
        try:
            v = self._direction(combo, custom, sign)
            return v, None
        except Exception as exc:
            return None, str(exc)

    def _preview_coordinate_frame(self, *_):
        if not self.lines:
            return
        # 解析 X/Y 方向向量并更新标签
        xv, xe = self._direction_label(self.x_axis, self.x_custom, self.x_sign)
        yv, ye = self._direction_label(self.y_axis, self.y_custom, self.y_sign)
        self.x_dir_label.setText(
            f"[{xv[0]:+.3f}, {xv[1]:+.3f}, {xv[2]:+.3f}]" if xv is not None else f"错误：{xe}")
        self.y_dir_label.setText(
            f"[{yv[0]:+.3f}, {yv[1]:+.3f}, {yv[2]:+.3f}]" if yv is not None else f"错误：{ye}")
        # 计算选定的交线组并高亮
        x_group = self.x_axis.currentIndex() if self.x_axis.currentIndex() < 3 else -1
        y_group = self.y_axis.currentIndex() if self.y_axis.currentIndex() < 3 else -1
        highlight = set(g for g in (x_group, y_group) if g >= 0)
        self.view.show_box(self.lines, self.corners, highlight_groups=highlight)
        # 尝试计算完整坐标系
        if xv is None or yv is None:
            self.axis_warning.setText(xe or ye)
            return
        if abs(float(np.dot(xv, yv))) > 0.95:
            self.axis_warning.setText("⚠ X 轴与 Y 轴方向平行，请选择不同颜色的交线或修改自定义方向。")
            return
        self.axis_warning.setText("")
        try:
            rotation, origin = self._selected_frame()
            self._applied_rotation = rotation.copy()
            self._applied_origin = origin.copy()
            self.current_rotation = rotation
            self.current_origin = origin
            self.view.show_coordinate_frame(origin, rotation, self.frame_scale)
            # 显示表面贴合补偿信息
            x_d, y_d, z_d = self._last_snap_devs
            if abs(x_d) + abs(y_d) + abs(z_d) > 1e-7:
                def _fmt(v):
                    return f"{v:+.4f}"
                self.snap_info.setText(
                    f"表面贴合补偿: X={_fmt(x_d)}  Y={_fmt(y_d)}  Z={_fmt(z_d)} m"
                    + (f"  (主偏差 Z={_fmt(z_d)})" if abs(z_d) > max(abs(x_d), abs(y_d)) else ""))
            else:
                self.snap_info.setText("")
        except Exception as exc:
            self.axis_warning.setText(f"⚠ {exc}")

    def _apply_coordinate_frame(self):
        if not self._has_cloud():
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载点云。")
            return
        try:
            rotation, origin = self._selected_frame()
            self._applied_rotation = rotation.copy()
            self._applied_origin = origin.copy()
            changed = not np.allclose(rotation, np.eye(3), atol=1e-8) or not np.allclose(origin, 0, atol=1e-8)
            if not changed:
                self._coordinate_applied(False)
                return
            self._set_busy("正在后台变换全量点云...")
            points = self.detector.points
            self._run_task(
                lambda: ((rotation.T @ (points - origin).T).T).astype(np.float64),
                lambda transformed: self._coordinate_transform_done(transformed))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "坐标系无效", str(exc))

    def _coordinate_transform_done(self, transformed):
        rotation = self._applied_rotation   # 在 _apply_coordinate_frame 中已保存
        origin   = self._applied_origin
        self.detector.points = transformed
        # 同步已有 filtered_points，避免旧坐标残留导致聚类坐标错误
        if self.detector.filtered_points is not None:
            fp = self.detector.filtered_points
            self.detector.filtered_points = (
                rotation.T @ (fp - origin).T).T.astype(np.float64)
        # 同步已有聚类中心和点云
        for info in self.detector.clusters.values():
            cpts = info['points']
            info['points'] = (rotation.T @ (cpts - origin).T).T.astype(np.float64)
            c = info['center']
            info['center'] = (rotation.T @ (c - origin)).astype(np.float64)
        self._coordinate_applied(True)

    def _coordinate_applied(self, changed):
        self.transform_applied = changed
        self.view.clear_guides()
        self._set_cloud(self.detector.points, self.detector.intensities)
        self.current_origin = np.zeros(3)
        self.current_rotation = np.eye(3)
        self.view.show_coordinate_frame(self.current_origin, self.current_rotation, self.frame_scale)
        self._last_snap_devs = (0.0, 0.0, 0.0)
        self.snap_info.setText("")
        self._set_ready("坐标系已应用" if changed else "保持原始坐标系")

    def _export_coordinate_las(self):
        if not self._has_cloud():
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载并转换点云。")
            return
        output_file = self.coord_output_edit.text().strip() or f"{self.args.output}_cloud.las"
        suffix = Path(output_file).suffix.lower()
        if suffix not in (".las", ".pcd"):
            output_file += ".las"
            self.coord_output_edit.setText(output_file)
            suffix = ".las"
        format_name = suffix[1:].upper()
        self._set_busy(f"正在导出坐标系转换 {format_name}...")

        def work():
            if suffix == ".pcd":
                self.api._export_transformed_pcd(self.detector, output_file)
            else:
                self.api._export_transformed_las(self.detector, output_file)
            return str(Path(output_file).resolve())

        def done(path):
            self.coord_info.setText(f"坐标系转换 {format_name} 已导出：{path}")
            self._set_ready(f"坐标系转换 {format_name} 导出完成")

        self._run_task(work, done)

    def _browse_analysis_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开用于强度过滤和聚类的点云", "", "LAS/LAZ 点云 (*.las *.laz)")
        if path:
            self.analysis_file_edit.setText(path)

    def _load_analysis_cloud(self):
        filepath = self.analysis_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择点云文件。")
            return
        self._set_busy("正在打开点云...")

        def work():
            return _read_las_cloud(filepath)

        self._run_task(work, self._analysis_cloud_loaded)

    def _analysis_cloud_loaded(self, result):
        points, intensities = result
        self._replace_current_cloud(points, intensities)
        self.coord_info.setText("已打开新点云；如需坐标转换，请回到功能一重新拟合/应用。")
        self._set_ready(f"已打开点云：{len(points):,} 个点")

    def _replace_current_cloud(self, points, intensities):
        self.detector.points = points
        self.detector.intensities = intensities
        self.detector.filtered_points = None
        self.detector.filtered_intensities = None
        self.detector.cluster_labels = None
        self.detector.clusters.clear()
        self.detector.removed_clusters.clear()
        self.transform_applied = False
        self._applied_rotation = np.eye(3)
        self._applied_origin = np.zeros(3)
        self.current_rotation = np.eye(3)
        self.current_origin = np.zeros(3)
        self.axes = None
        self.centroid = None
        self.lines = []
        self.corners = np.empty((0, 3))
        self.frame_scale = 1.0
        self._last_snap_devs = (0.0, 0.0, 0.0)
        if hasattr(self, 'snap_info'):
            self.snap_info.setText("")
        self.view.clear_guides()
        self.view.clear_compare()
        self.view.clear_reg()
        self._set_cloud(points, intensities)
        self.status_label.setText(f"已加载 {len(points):,} 个点")
        initial = int(np.percentile(intensities, self.args.percentile))
        self.threshold_slider.setRange(int(np.min(intensities)), int(np.max(intensities)))
        self.threshold_spin.setRange(self.threshold_slider.minimum(), self.threshold_slider.maximum())
        self.threshold_slider.setValue(initial)
        self.threshold_spin.setValue(initial)
        self._reset_intensity_widgets(initial)
        self.cluster_table.setRowCount(0)
        self.cluster_summary.setText("尚未运行聚类")

    def _reset_intensity_widgets(self, initial):
        self.histogram.clear()
        if not self._has_intensities():
            self.keep_label.setText("未加载点云")
            self.threshold_line = pg.InfiniteLine(pos=initial, angle=90, pen=pg.mkPen("#d64545", width=2))
            self.histogram.addItem(self.threshold_line)
            return
        integer_intensity = np.asarray(self.detector.intensities, dtype=np.uint16)
        self.intensity_counts = np.bincount(integer_intensity, minlength=65536)
        self.intensity_tail_counts = np.cumsum(self.intensity_counts[::-1], dtype=np.int64)[::-1]
        counts, edges = np.histogram(integer_intensity, bins=256)
        curve = pg.PlotCurveItem(edges, counts, stepMode="center", fillLevel=0,
                                 brush=(0, 121, 107, 100), pen=(0, 121, 107))
        self.histogram.addItem(curve)
        self.threshold_line = pg.InfiniteLine(pos=initial, angle=90, pen=pg.mkPen("#d64545", width=2))
        self.histogram.addItem(self.threshold_line)

    def _preview_threshold(self, value):
        if self.threshold_line is None or not self._has_cloud() or not self._has_intensities():
            self.keep_label.setText("未加载点云")
            return
        self.threshold_line.setValue(value)
        sample_ids = _sample_indices(len(self.detector.points), MAX_VIEW_POINTS)
        sample_intensity = self.detector.intensities[sample_ids]
        keep = sample_intensity >= value
        self._set_cloud(self.detector.points[sample_ids][keep], sample_intensity[keep], limit=MAX_VIEW_POINTS, size=3)
        count_index = min(max(int(value), 0), len(self.intensity_tail_counts) - 1)
        count = int(self.intensity_tail_counts[count_index])
        self.keep_label.setText(f"预计保留 {count:,} / {len(self.detector.points):,} 个点")

    def _apply_threshold(self):
        if not self._has_cloud() or not self._has_intensities():
            QtWidgets.QMessageBox.warning(self, "错误", "请先打开点云。")
            return
        threshold = float(self.threshold_spin.value())
        self.detector.filter_by_intensity(threshold)
        self._set_cloud(self.detector.filtered_points, self.detector.filtered_intensities, size=3)
        self.tabs.setCurrentIndex(2)
        self._set_ready(f"强度过滤完成：保留 {len(self.detector.filtered_points):,} 个点")

    def _run_clustering(self):
        if not self._has_cloud():
            QtWidgets.QMessageBox.warning(self, "错误", "请先打开点云。")
            return
        if self.detector.filtered_points is None:
            self._apply_threshold()
            if self.detector.filtered_points is None:
                return
        auto_eps = self.auto_eps.isChecked()
        eps_value = self.eps_spin.value()
        min_samples = self.min_samples.value()
        self._set_busy("正在后台运行 DBSCAN...")

        def work():
            eps = self.api.estimate_eps(self.detector.filtered_points) if auto_eps else eps_value
            self.detector.cluster(eps, min_samples)
            return eps

        self._run_task(work, self._cluster_done)

    def _cluster_done(self, eps):
        self.eps_spin.setValue(eps)
        self.detector.removed_clusters.clear()
        self.cluster_table.blockSignals(True)
        self.cluster_table.setRowCount(len(self.detector.clusters))
        for row, (label, info) in enumerate(sorted(self.detector.clusters.items())):
            keep = QtWidgets.QTableWidgetItem(); keep.setFlags(keep.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            keep.setCheckState(QtCore.Qt.CheckState.Checked); keep.setData(QtCore.Qt.ItemDataRole.UserRole, label)
            # 构建表格行数据
            count = info["count"]
            intensities = info["intensities"]
            intensity_min, intensity_max = float(intensities.min()), float(intensities.max())
            center = info["center"]
            radius = info.get("radius", 0.0)
            # 获取点号：显示前几个样本点号 + 省略号
            indices = info.get("indices", np.array([]))
            if len(indices) > 0:
                sample_count = min(5, len(indices))
                sample_list = ",".join(str(int(i)) for i in indices[:sample_count])
                if len(indices) > sample_count:
                    sample_list += f"...({len(indices)}个)"
            else:
                sample_list = "无"
            items = [
                str(label),
                str(count),
                f"{intensity_min:.1f}-{intensity_max:.1f}",
                f"{center[0]:.4f}",
                f"{center[1]:.4f}",
                f"{center[2]:.4f}",
                sample_list,
                f"R≈{radius:.4f}"
            ]
            self.cluster_table.setItem(row, 0, keep)
            for column, value in enumerate(items, 1):
                self.cluster_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.cluster_table.blockSignals(False)
        self.cluster_summary.setText(f"发现 {len(self.detector.clusters)} 个聚类；取消勾选即可剔除")
        self._show_clusters()
        self._set_ready("聚类完成")

    def _show_clusters(self):
        if self.detector.filtered_points is None:
            return
        self._set_cloud(self.detector.filtered_points, self.detector.filtered_intensities,
                        limit=MAX_CLUSTER_VIEW_POINTS, size=2)
        centers = [(label, info["center"]) for label, info in sorted(self.detector.clusters.items())]
        self.view.show_cluster_centers(centers, self.detector.removed_clusters)

    def _cluster_check_changed(self, item):
        if item.column() != 0:
            return
        label = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self.detector.removed_clusters.discard(label)
        else:
            self.detector.removed_clusters.add(label)
        self.cluster_summary.setText(f"共 {len(self.detector.clusters)} 个聚类，已剔除 {len(self.detector.removed_clusters)} 个")
        self._show_clusters()

    def _tab_changed(self, index):
        if not self._has_cloud() and index in (0, 1, 2):
            self._set_cloud(self.detector.points, self.detector.intensities)
            return
        if index == 0 and self.lines:
            self._set_cloud(self.detector.points, self.detector.intensities)
            if not self.transform_applied:
                self._preview_coordinate_frame()  # 内部调用 show_box 并高亮
            else:
                self.view.clear_guides()
                self.view.show_coordinate_frame(self.current_origin, self.current_rotation, self.frame_scale)
        elif index == 1:
            self._preview_threshold(self.threshold_spin.value())
            # 在强度过滤时也显示已选择的坐标系框
            if self.lines and not self.transform_applied:
                self._preview_coordinate_frame()  # 显示高亮的坐标框
        elif index == 2 and self.detector.clusters:
            self._show_clusters()
            # 清除聚类审核时的坐标系显示，避免干扰
            self.view.clear_guides()
            self.view.clear_frame()
        elif index == 4:
            self._on_compare_tab_shown()
        elif index == 5:
            self._on_reg_tab_shown()

    def _export(self):
        if not self._has_cloud():
            QtWidgets.QMessageBox.warning(self, "错误", "请先打开点云或完成聚类结果。")
            return
        prefix = self.output_edit.text().strip() or "ball_centers"
        export_csv = self.export_csv.isChecked()
        export_cloud = self.export_cloud.isChecked()
        export_pcd = self.export_pcd.isChecked()
        export_clusters = self.export_clusters.isChecked()
        close_after_export = self.close_after_export.isChecked()
        if not any((export_csv, export_cloud, export_pcd, export_clusters)):
            QtWidgets.QMessageBox.information(self, "没有输出", "请至少选择一种导出格式。")
            return
        self._set_busy("正在后台导出结果...")

        def work():
            if export_csv:
                self.detector.export_results(prefix)
            if export_cloud:
                self.api._export_transformed_las(self.detector, prefix + "_cloud.las")
            if export_pcd:
                self.api._export_transformed_pcd(self.detector, prefix + "_cloud.pcd")
            if export_clusters:
                self.api._export_cluster_las(self.detector, prefix + "_clusters.las")
            return str(Path(prefix).resolve().parent), close_after_export

        self._run_task(work, self._export_done)

    def _export_done(self, result):
        output_directory, close_after_export = result
        self.export_info.setText(f"导出完成：{output_directory}")
        self.close_button.setEnabled(True)
        self._set_ready("结果导出完成")
        if close_after_export:
            QtCore.QTimer.singleShot(250, self.close)


    # ═══════════════════════════════════════════════════════════════════════
    # Tab 5: 点云对比
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════════
    # Tab 5: 点云对比
    # ═══════════════════════════════════════════════════════════════════════════════

    def _build_compare_tab(self):
        page, box = self._page()
        intro = QtWidgets.QLabel(
            "对比两份同一标定房的点云。"
            "两份点云分别进行独立的坐标系变换（RANSAC拟合），"
            "再做最近邻误差统计；超过阀値的点对（新增物体等）将被排除。")
        intro.setWordWrap(True)
        box.addWidget(intro)

        # A0. 直接选择两份点云用于误差分析
        direct_group = QtWidgets.QGroupBox("① 直接选择两份点云用于误差分析")
        direct_layout = QtWidgets.QFormLayout(direct_group)
        c1_row = QtWidgets.QHBoxLayout()
        self.compare1_file_edit = QtWidgets.QLineEdit()
        c1_browse = QtWidgets.QPushButton("浏览...")
        c1_browse.clicked.connect(self._browse_compare1_file)
        c1_load = QtWidgets.QPushButton("加载第一份")
        c1_load.clicked.connect(self._load_compare1_direct)
        c1_row.addWidget(self.compare1_file_edit)
        c1_row.addWidget(c1_browse)
        c1_row.addWidget(c1_load)
        direct_layout.addRow("第一份点云", c1_row)
        c2_direct_row = QtWidgets.QHBoxLayout()
        self.compare2_direct_file_edit = QtWidgets.QLineEdit()
        c2_direct_browse = QtWidgets.QPushButton("浏览...")
        c2_direct_browse.clicked.connect(self._browse_compare2_direct_file)
        c2_direct_load = QtWidgets.QPushButton("加载第二份")
        c2_direct_load.clicked.connect(self._load_compare2_direct)
        c2_direct_row.addWidget(self.compare2_direct_file_edit)
        c2_direct_row.addWidget(c2_direct_browse)
        c2_direct_row.addWidget(c2_direct_load)
        direct_layout.addRow("第二份点云", c2_direct_row)
        self.compare_direct_info = QtWidgets.QLabel("未直接加载；默认第一份使用当前主点云。")
        self.compare_direct_info.setWordWrap(True)
        direct_layout.addRow("状态", self.compare_direct_info)
        box.addWidget(direct_group)

        # A. 加载第二份点云并拟合坐标系
        load_group = QtWidgets.QGroupBox("② 可选：加载第二份点云并拟合坐标系")
        load_layout = QtWidgets.QFormLayout(load_group)
        file_row = QtWidgets.QHBoxLayout()
        self.compare_file_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_compare_file)
        file_row.addWidget(self.compare_file_edit)
        file_row.addWidget(browse_btn)
        load_layout.addRow("点云文件", file_row)
        load_btn = QtWidgets.QPushButton("加载并 RANSAC 拟合坐标系")
        load_btn.clicked.connect(self._load_compare2_and_fit)
        load_layout.addRow("", load_btn)
        self.c2_fit_info = QtWidgets.QLabel("尚未加载")
        self.c2_fit_info.setWordWrap(True)
        load_layout.addRow("拟合状态", self.c2_fit_info)
        box.addWidget(load_group)

        # B. 坐标系配置（与主点云 Tab1 一致）
        def _gi2(r, g, b):
            pix = QtGui.QPixmap(14, 14)
            pix.fill(QtGui.QColor(int(r*255), int(g*255), int(b*255)))
            return QtGui.QIcon(pix)
        _ic2 = [_gi2(0.91,0.30,0.24), _gi2(0.18,0.80,0.44), _gi2(0.20,0.60,0.86)]

        axis_group = QtWidgets.QGroupBox("③ 可选：配置第二份点云坐标系")
        axis_form = QtWidgets.QFormLayout(axis_group)
        self.c2_x_axis = QtWidgets.QComboBox()
        self.c2_y_axis = QtWidgets.QComboBox()
        for combo in (self.c2_x_axis, self.c2_y_axis):
            combo.addItem(_ic2[0], "红色方向（组 0）")
            combo.addItem(_ic2[1], "绿色方向（组 1）")
            combo.addItem(_ic2[2], "蓝色方向（组 2）")
            combo.addItem("自定义")
        self.c2_x_axis.setCurrentIndex(3)
        self.c2_y_axis.setCurrentIndex(3)
        self.c2_x_sign = QtWidgets.QComboBox(); self.c2_x_sign.addItems(["正向", "反向"])
        self.c2_y_sign = QtWidgets.QComboBox(); self.c2_y_sign.addItems(["正向", "反向"])
        self.c2_x_custom = QtWidgets.QLineEdit("1,0,0")
        self.c2_y_custom = QtWidgets.QLineEdit("0,1,0")
        self.c2_x_dir_label = QtWidgets.QLabel("—"); self.c2_x_dir_label.setStyleSheet("color:#555; font-size:11px;")
        self.c2_y_dir_label = QtWidgets.QLabel("—"); self.c2_y_dir_label.setStyleSheet("color:#555; font-size:11px;")
        axis_form.addRow("X 方向", self.c2_x_axis);   axis_form.addRow("X 正反向", self.c2_x_sign)
        axis_form.addRow("X 自定义", self.c2_x_custom); axis_form.addRow("X 实际方向", self.c2_x_dir_label)
        axis_form.addRow("Y 方向", self.c2_y_axis);   axis_form.addRow("Y 正反向", self.c2_y_sign)
        axis_form.addRow("Y 自定义", self.c2_y_custom); axis_form.addRow("Y 实际方向", self.c2_y_dir_label)
        self.c2_axis_warning = QtWidgets.QLabel()
        self.c2_axis_warning.setWordWrap(True)
        self.c2_axis_warning.setStyleSheet("color:#c0392b; font-weight:bold; padding:4px;")

        orig_group2 = QtWidgets.QGroupBox("原点与附加旋转（第二份点云）")
        orig_form2 = QtWidgets.QFormLayout(orig_group2)
        self.c2_origin_combo = QtWidgets.QComboBox()
        self.c2_origin_combo.addItems(
            ["世界原点 (0,0,0)", "拟合盒中心"] + [f"角点 {i}" for i in range(8)])
        self.c2_rot_x = QtWidgets.QDoubleSpinBox()
        self.c2_rot_y = QtWidgets.QDoubleSpinBox()
        self.c2_rot_z = QtWidgets.QDoubleSpinBox()
        for sp in (self.c2_rot_x, self.c2_rot_y, self.c2_rot_z):
            sp.setRange(-360, 360); sp.setDecimals(2); sp.setSuffix("°")
        orig_form2.addRow("坐标原点", self.c2_origin_combo)
        orig_form2.addRow("绕 X 旋转", self.c2_rot_x)
        orig_form2.addRow("绕 Y 旋转", self.c2_rot_y)
        orig_form2.addRow("绕 Z 旋转", self.c2_rot_z)

        for w in (self.c2_x_axis, self.c2_y_axis, self.c2_x_sign, self.c2_y_sign, self.c2_origin_combo):
            w.currentIndexChanged.connect(self._preview_compare2_frame)
        for w in (self.c2_x_custom, self.c2_y_custom):
            w.editingFinished.connect(self._preview_compare2_frame)
        for w in (self.c2_rot_x, self.c2_rot_y, self.c2_rot_z):
            w.valueChanged.connect(lambda _v: self._preview_compare2_frame())

        c2_apply_btn = QtWidgets.QPushButton("应用第二份点云坐标系")
        c2_apply_btn.setObjectName("primary")
        c2_apply_btn.clicked.connect(self._apply_compare2_frame)
        self.c2_coord_info = QtWidgets.QLabel("请先加载第二份点云并完成拟合")
        self.c2_coord_info.setWordWrap(True)

        box.addWidget(axis_group)
        box.addWidget(self.c2_axis_warning)
        box.addWidget(orig_group2)
        box.addWidget(c2_apply_btn)
        box.addWidget(self.c2_coord_info)

        # C. 对比参数
        cmp_group = QtWidgets.QGroupBox("④ 对比参数")
        cmp_layout = QtWidgets.QFormLayout(cmp_group)
        self.compare_max_dist = QtWidgets.QDoubleSpinBox()
        self.compare_max_dist.setRange(0.0001, 100.0); self.compare_max_dist.setValue(0.05)
        self.compare_max_dist.setDecimals(4); self.compare_max_dist.setSuffix(" m（超出则排除）")
        self.compare_voxel = QtWidgets.QDoubleSpinBox()
        self.compare_voxel.setRange(0.001, 10.0); self.compare_voxel.setValue(0.02)
        self.compare_voxel.setDecimals(4); self.compare_voxel.setSuffix(" m（体素降采样）")
        cmp_layout.addRow("最大匹配距离", self.compare_max_dist)
        cmp_layout.addRow("体素大小",     self.compare_voxel)
        box.addWidget(cmp_group)

        run_btn = QtWidgets.QPushButton("⑤ 运行点云对比分析")
        run_btn.setObjectName("primary"); run_btn.clicked.connect(self._run_compare)
        box.addWidget(run_btn)
        self.compare_result_label = QtWidgets.QLabel("尚未运行对比分析")
        self.compare_result_label.setWordWrap(True)
        self.compare_result_label.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace; font-size:12px; "
            "background:#f8f8f8; padding:6px; border:1px solid #ddd;")
        box.addWidget(self.compare_result_label)
        self.tabs.addTab(page, "5 点云对比")

    # ── 第二份点云坐标系辅助方法 ────────────────────────────────────────────────────────────

    def _browse_compare1_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择第一份点云", "", "点云文件 (*.las *.laz *.pcd);;LAS/LAZ 点云 (*.las *.laz);;PCD 点云 (*.pcd)")
        if path:
            self.compare1_file_edit.setText(path)

    def _browse_compare2_direct_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择第二份点云", "", "点云文件 (*.las *.laz *.pcd);;LAS/LAZ 点云 (*.las *.laz);;PCD 点云 (*.pcd)")
        if path:
            self.compare2_direct_file_edit.setText(path)

    def _load_compare1_direct(self):
        filepath = self.compare1_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择第一份点云文件。")
            return
        self._set_busy("正在加载第一份对比点云...")

        def work():
            return _read_point_cloud(filepath)

        def done(result):
            pts, intensities = result
            self.compare1_pts = pts
            self.compare1_intensities = intensities
            self._set_cloud(pts, intensities, limit=80_000, size=1)
            if self.compare2_pts is not None:
                self.view.set_compare_cloud(self.compare2_pts)
            self.compare_direct_info.setText(f"第一份已加载：{len(pts):,} 点")
            self._set_ready("第一份对比点云已加载")

        self._run_task(work, done)

    def _load_compare2_direct(self):
        filepath = self.compare2_direct_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择第二份点云文件。")
            return
        self._set_busy("正在加载第二份对比点云...")

        def work():
            return _read_point_cloud(filepath)[0]

        def done(pts):
            self.compare2_raw_pts = pts
            self.compare2_pts = pts
            self.compare2_lines = []
            base = self.compare1_pts if self.compare1_pts is not None else self.detector.points
            base_i = self.compare1_intensities if self.compare1_intensities is not None else self.detector.intensities
            if base is not None and len(base):
                self._set_cloud(base, base_i, limit=80_000, size=1)
            else:
                self.view.set_cloud(pts, limit=80_000, size=1)
            self.view.set_compare_cloud(pts)
            first_count = len(base) if base is not None else 0
            self.compare_direct_info.setText(
                f"第一份：{first_count:,} 点；第二份已直接加载：{len(pts):,} 点。可运行误差分析。")
            self.c2_coord_info.setText("第二份点云已直接加载，未进行坐标系拟合/变换。")
            self._set_ready("第二份对比点云已加载")

        self._run_task(work, done)

    def _browse_compare_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择第二份点云", "", "点云文件 (*.las *.laz *.pcd);;LAS/LAZ 点云 (*.las *.laz);;PCD 点云 (*.pcd)")
        if path:
            self.compare_file_edit.setText(path)

    def _load_compare2_and_fit(self):
        filepath = self.compare_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择第二份点云文件。")
            return
        self._set_busy("正在加载并拟合坐标系...")
        self.c2_fit_info.setText("加载中...")

        def work():
            pts = _read_point_cloud(filepath)[0]
            fit_result = self.api.fit_box_planes_pca(pts)
            return pts, fit_result

        self._run_task(work, self._compare2_fit_done)

    def _compare2_fit_done(self, result):
        pts, (axes, centroid, plane_pairs, corners, lines) = result
        self.compare2_raw_pts  = pts
        self.compare2_pts      = None
        self.compare2_axes     = axes
        self.compare2_centroid = centroid
        self.compare2_corners  = corners
        self.compare2_lines    = lines
        frame_scale2 = max(float(np.max(np.ptp(corners, axis=0))) * 0.22, 0.1)
        self.view.set_cloud(pts, limit=120_000, size=2)
        self.view.show_box(lines, corners, highlight_groups=set())
        diag = plane_pairs[0].get("diagnostics", {}) if plane_pairs else {}
        self.c2_fit_info.setText(
            f"已加载 {len(pts):,} 个点；拟合 8 个角点，阈値 {diag.get('distance_threshold',0):.6f}")
        self.c2_coord_info.setText("请在上方选择坐标轴后点击「应用第二份点云坐标系」")
        self._preview_compare2_frame()
        self._set_ready("第二份点云拟合完成")

    def _direction2(self, combo, custom, sign):
        index = combo.currentIndex()
        if index < 3:
            if not self.compare2_lines:
                raise ValueError("第二份点云尚未拟合，请先点击「加载并拟合」。")
            vector = np.asarray(self.compare2_lines[index * 4][0], dtype=float)
        else:
            parts = [float(p.strip()) for p in custom.text().replace("，", ",").split(",")]
            if len(parts) != 3:
                raise ValueError("自定义方向必须是三个逗号分隔的数字。")
            vector = np.asarray(parts, dtype=float)
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("坐标轴方向不能为零向量。")
        return vector / norm * (-1 if sign.currentIndex() else 1)

    def _selected_compare2_frame(self):
        x = self._direction2(self.c2_x_axis, self.c2_x_custom, self.c2_x_sign)
        y = self._direction2(self.c2_y_axis, self.c2_y_custom, self.c2_y_sign)
        if abs(float(np.dot(x, y))) > 0.95:
            raise ValueError("X 和 Y 轴不能平行。")
        rotation = self.api._gs3(np.column_stack((x, y, np.cross(x, y))))
        for axis_idx, angle in enumerate(
                (self.c2_rot_x.value(), self.c2_rot_y.value(), self.c2_rot_z.value())):
            rotation = self.api._rot_mat(rotation[:, axis_idx], angle) @ rotation
            rotation = self.api._gs3(rotation)
        oi = self.c2_origin_combo.currentIndex()
        origin = (np.zeros(3) if oi == 0
                  else (self.compare2_centroid if oi == 1
                        else self.compare2_corners[oi - 2]))
        return rotation, np.asarray(origin, dtype=np.float64)

    def _preview_compare2_frame(self, *_):
        if not self.compare2_lines:
            return
        xv, xe, yv, ye = None, None, None, None
        try: xv = self._direction2(self.c2_x_axis, self.c2_x_custom, self.c2_x_sign)
        except Exception as e: xe = str(e)
        try: yv = self._direction2(self.c2_y_axis, self.c2_y_custom, self.c2_y_sign)
        except Exception as e: ye = str(e)
        self.c2_x_dir_label.setText(
            f"[{xv[0]:+.3f}, {xv[1]:+.3f}, {xv[2]:+.3f}]" if xv is not None else f"错误：{xe}")
        self.c2_y_dir_label.setText(
            f"[{yv[0]:+.3f}, {yv[1]:+.3f}, {yv[2]:+.3f}]" if yv is not None else f"错误：{ye}")
        xg = self.c2_x_axis.currentIndex() if self.c2_x_axis.currentIndex() < 3 else -1
        yg = self.c2_y_axis.currentIndex() if self.c2_y_axis.currentIndex() < 3 else -1
        self.view.show_box(self.compare2_lines, self.compare2_corners,
                           highlight_groups={g for g in (xg, yg) if g >= 0})
        if xv is None or yv is None:
            self.c2_axis_warning.setText(xe or ye); return
        if abs(float(np.dot(xv, yv))) > 0.95:
            self.c2_axis_warning.setText("⚠ X 轴与 Y 轴方向平行，请选择不同颜色的交线。"); return
        self.c2_axis_warning.setText("")
        try:
            rotation, origin = self._selected_compare2_frame()
            fs2 = max(float(np.max(np.ptp(self.compare2_corners, axis=0))) * 0.22, 0.1)
            self.view.show_coordinate_frame(origin, rotation, fs2)
        except Exception as exc:
            self.c2_axis_warning.setText(f"⚠ {exc}")

    def _apply_compare2_frame(self):
        if self.compare2_raw_pts is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载并拟合第二份点云。"); return
        try:
            rotation, origin = self._selected_compare2_frame()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "坐标系无效", str(exc)); return
        raw = self.compare2_raw_pts
        self._set_busy("正在变换第二份点云...")

        def work():
            return (rotation.T @ (raw - origin).T).T.astype(np.float64)

        def on_done(pts2_xf):
            self.compare2_pts = pts2_xf
            self._set_cloud(self.detector.points, self.detector.intensities, limit=80_000, size=1)
            self.view.set_compare_cloud(pts2_xf)
            self.view.clear_guides(); self.view.clear_frame()
            self.c2_coord_info.setText(
                f"坐标系已应用，第二份点云共 {len(pts2_xf):,} 个点。"
                f"现在可点击「④ 运行点云对比分析」。")
            self._set_ready("第二份点云坐标系已应用")

        self._run_task(work, on_done)

    # ── 对比运行 ─────────────────────────────────────────────────────────────

    def _run_compare(self):
        c1_path = self.compare1_file_edit.text().strip() if hasattr(self, "compare1_file_edit") else ""
        c2_path = self.compare2_direct_file_edit.text().strip() if hasattr(self, "compare2_direct_file_edit") else ""
        if (self.compare1_pts is None or self.compare2_pts is None) and c1_path and c2_path:
            self._run_compare_direct_from_files(c1_path, c2_path)
            return
        if self.compare2_pts is None:
            QtWidgets.QMessageBox.warning(
                self, "错误",
                "请先在“直接选择两份点云”中选择两份 LAS，或加载并应用第二份点云坐标系。"); return
        base_pts = self.compare1_pts if self.compare1_pts is not None else self.detector.points
        base_intensities = (self.compare1_intensities
                            if self.compare1_intensities is not None else self.detector.intensities)
        if base_pts is None or not len(base_pts):
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载主点云。"); return
        self._run_compare_for_points(base_pts, base_intensities, self.compare2_pts)

    def _run_compare_direct_from_files(self, first_path, second_path):
        max_dist = self.compare_max_dist.value()
        self._set_busy("正在加载两份点云并运行对比分析...")

        def work():
            p1, i1 = _read_las_cloud(first_path)
            p2 = _read_las_points(second_path)
            return p1, i1, p2

        def done(result):
            p1, i1, p2 = result
            self.compare1_pts = p1
            self.compare1_intensities = i1
            self.compare2_raw_pts = p2
            self.compare2_pts = p2
            self.compare2_lines = []
            self.compare_direct_info.setText(
                f"第一份已加载：{len(p1):,} 点；第二份已加载：{len(p2):,} 点。正在统计误差...")
            self.c2_coord_info.setText("第二份点云已直接加载，未进行坐标系拟合/变换。")
            self._run_compare_for_points(p1, i1, p2, already_busy=True)

        self.compare_result_label.setText(
            f"正在直接对比两份点云；最大匹配距离 {max_dist*1000:.2f} mm...")
        self._run_task(work, done)

    def _run_compare_for_points(self, base_pts, base_intensities, compare_pts, already_busy=False):
        max_dist   = self.compare_max_dist.value()
        voxel_size = self.compare_voxel.value()
        p1_snap    = base_pts.copy()
        p2_snap    = compare_pts.copy()
        i1_snap    = None if base_intensities is None else base_intensities.copy()

        def voxel_down(pts, vsize):
            if len(pts) == 0 or vsize <= 0.001: return pts
            keys = np.floor(pts / vsize).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            return pts[idx]

        def match_plane_axes(pairs1, pairs2):
            best_perm, best_score = None, -1.0
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                         (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                score = sum(abs(float(np.dot(pairs1[i]["normal"], pairs2[perm[i]]["normal"])))
                            for i in range(3))
                if score > best_score:
                    best_perm, best_score = perm, score
            return best_perm

        def plane_analysis(p1_full, p2_full, p2_draw):
            _axes1, _c1, pairs1, _corners1, _lines1 = self.api.fit_box_planes_pca(p1_full)
            _axes2, _c2, pairs2, _corners2, _lines2 = self.api.fit_box_planes_pca(p2_full)
            perm = match_plane_axes(pairs1, pairs2)
            plane_rows = []
            axis_rows = []
            target_planes = []
            for axis_idx, p2_idx in enumerate(perm):
                n1 = np.asarray(pairs1[axis_idx]["normal"], dtype=np.float64)
                n2_raw = np.asarray(pairs2[p2_idx]["normal"], dtype=np.float64)
                sign = 1.0 if float(np.dot(n1, n2_raw)) >= 0.0 else -1.0
                n2 = n2_raw * sign
                angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(n1, n2_raw))), -1.0, 1.0))))
                d1 = np.array([pairs1[axis_idx]["d_lo"], pairs1[axis_idx]["d_hi"]], dtype=np.float64)
                if sign > 0:
                    d2 = np.array([pairs2[p2_idx]["d_lo"], pairs2[p2_idx]["d_hi"]], dtype=np.float64)
                else:
                    d2 = np.array([-pairs2[p2_idx]["d_hi"], -pairs2[p2_idx]["d_lo"]], dtype=np.float64)
                axis_rows.append({
                    "axis": axis_idx,
                    "matched_axis": p2_idx,
                    "angle_deg": angle,
                    "width_delta": float(abs((d2[1] - d2[0]) - (d1[1] - d1[0]))),
                })
                for side_idx, side_name in enumerate(("lo", "hi")):
                    delta = float(d2[side_idx] - d1[side_idx])
                    plane_rows.append({
                        "axis": axis_idx,
                        "matched_axis": p2_idx,
                        "side": side_name,
                        "distance": delta,
                        "abs_distance": abs(delta),
                        "angle_deg": angle,
                    })
                    target_planes.append((n1, float(d1[side_idx])))

            signed = np.column_stack([(p2_draw @ n) - d for n, d in target_planes])
            nearest_ids = np.argmin(np.abs(signed), axis=1)
            plane_deviation = np.abs(signed[np.arange(len(p2_draw)), nearest_ids])
            plane_valid = plane_deviation <= max_dist
            return {
                "plane_rows": plane_rows,
                "axis_rows": axis_rows,
                "plane_deviation": plane_deviation,
                "plane_valid": plane_valid,
                "plane_max_dist": max_dist,
            }

        def work():
            p1 = voxel_down(p1_snap, voxel_size)
            p2 = voxel_down(p2_snap, voxel_size)
            from sklearn.neighbors import KDTree
            tree = KDTree(p1)
            dists, _ = tree.query(p2, k=1)
            dists = dists[:, 0]
            valid = dists <= max_dist
            vd = dists[valid]
            stats = {
                'n1': len(p1), 'n2': len(p2),
                'n_pairs': len(dists), 'n_valid': int(valid.sum()),
                'pct_valid': 100.0 * float(valid.mean()) if len(dists) else 0.0,
                'mean':   float(np.mean(vd))              if len(vd) else 0.0,
                'median': float(np.median(vd))            if len(vd) else 0.0,
                'rmse':   float(np.sqrt(np.mean(vd**2))) if len(vd) else 0.0,
                'p95':    float(np.percentile(vd, 95))    if len(vd) else 0.0,
                'p99':    float(np.percentile(vd, 99))    if len(vd) else 0.0,
                'max_v':  float(np.max(vd))               if len(vd) else 0.0,
                'p1_view': p1_snap, 'i1_view': i1_snap,
                'p2': p2, 'dists': dists, 'valid': valid, 'max_dist': max_dist,
            }
            try:
                stats["plane"] = plane_analysis(p1_snap, p2_snap, p2)
            except Exception as exc:
                stats["plane_error"] = str(exc)
            return stats

        if not already_busy:
            self._set_busy("正在运行点云对比分析...")
        self._run_task(work, self._compare_done)

    def _compare_done(self, stats):
        p2, dists, valid = stats['p2'], stats['dists'], stats['valid']
        n = len(p2)
        colors = np.zeros((n, 4), dtype=np.float32); colors[:, 3] = 0.75
        plane = stats.get("plane")
        if plane is not None and len(plane.get("plane_deviation", ())) == n:
            dev = plane["plane_deviation"]
            plane_valid = plane["plane_valid"]
            cap = max(float(plane.get("plane_max_dist", stats["max_dist"])), 1e-9)
            t = np.clip(dev / cap, 0.0, 1.0).astype(np.float32)
            # 小偏差为浅黄绿；偏差越大越深，超过阈值为深红。
            colors[:, 0] = 0.78 - 0.38 * t
            colors[:, 1] = 0.95 - 0.90 * t
            colors[:, 2] = 0.52 - 0.50 * t
            colors[:, 3] = 0.82
            colors[np.where(~plane_valid)[0]] = [0.18, 0.00, 0.00, 0.92]
        else:
            if valid.any():
                ev = dists[valid]; ev_max = max(float(stats['p95']), 1e-9)
                ev_norm = np.clip(ev / ev_max, 0.0, 1.0)
                vi = np.where(valid)[0]
                colors[vi, 0] = 0.78 - 0.38 * ev_norm
                colors[vi, 1] = 0.95 - 0.90 * ev_norm
                colors[vi, 2] = 0.52 - 0.50 * ev_norm
            colors[np.where(~valid)[0]] = [0.18, 0.00, 0.00, 0.92]
        self._set_cloud(stats['p1_view'], stats['i1_view'], limit=80_000, size=1)
        self.view.set_compare_cloud(p2, error_colors=colors)
        md = stats['max_dist']
        lines = [
            "主要面差异分析",
            "=" * 44,
        ]
        if plane is not None:
            lines.extend([
                "主方向角度差:",
                *[
                    f"  轴 e{row['axis']} ↔ 次点云 e{row['matched_axis']}: "
                    f"{row['angle_deg']:.6f}°，宽度差 {row['width_delta']*1000:.3f} mm"
                    for row in plane["axis_rows"]
                ],
                "",
                "六个主面距离差（次点云面 - 主点云面）:",
                *[
                    f"  e{row['axis']} {row['side']:>2} ↔ e{row['matched_axis']} {row['side']:>2}: "
                    f"{row['distance']*1000:+.4f} mm  |Δ|={row['abs_distance']*1000:.4f} mm  "
                    f"角度={row['angle_deg']:.6f}°"
                    for row in sorted(plane["plane_rows"], key=lambda r: (r["axis"], r["side"]))
                ],
                "",
                f"视图颜色: 浅=主面偏差小，深=主面偏差大，深红=超过 {md*1000:.2f} mm",
                "",
            ])
        else:
            lines.extend([
                f"主要面分析失败: {stats.get('plane_error', '未知错误')}",
                "已回退为最近邻误差着色。",
                "",
            ])
        lines.extend([
            "最近邻点对统计",
            "=" * 44,
            f"主点云（降采样）: {stats['n1']:,} 点",
            f"次点云（降采样）: {stats['n2']:,} 点",
            f"有效点对 (\u2264{md*1000:.2f} mm): "
            f"{stats['n_valid']:,} / {stats['n_pairs']:,}  ({stats['pct_valid']:.1f}%)",
            f"RMSE:        {stats['rmse']*1000:.4f} mm",
            f"平均距离:    {stats['mean']*1000:.4f} mm",
            f"中位距离:    {stats['median']*1000:.4f} mm",
            f"95% 分位距:  {stats['p95']*1000:.4f} mm",
            f"99% 分位距:  {stats['p99']*1000:.4f} mm",
            f"最大有效距:  {stats['max_v']*1000:.4f} mm",
        ])
        text = '\n'.join(lines)
        self.compare_result_label.setText(text)
        status_prefix = "主面分析完成" if plane is not None else "对比完成"
        self._set_ready(
            f"{status_prefix}: RMSE={stats['rmse']*1000:.3f}mm  有效率={stats['pct_valid']:.1f}%")

    def _on_compare_tab_shown(self):
        if self.compare2_pts is not None:
            base = self.compare1_pts if self.compare1_pts is not None else self.detector.points
            base_i = self.compare1_intensities if self.compare1_intensities is not None else self.detector.intensities
            self._set_cloud(base, base_i, limit=80_000, size=1)
            self.view.set_compare_cloud(self.compare2_pts)
        elif self.compare2_raw_pts is not None:
            self.view.set_cloud(self.compare2_raw_pts, limit=120_000, size=2)
            if self.compare2_lines:
                self.view.show_box(self.compare2_lines, self.compare2_corners)
        else:
            if self._has_cloud():
                self._set_cloud(self.detector.points, self.detector.intensities)



    # ═══════════════════════════════════════════════════════════════════════
    # Tab 6: 点云配准（GICP）
    # ═══════════════════════════════════════════════════════════════════════

    def _build_registration_tab(self):
        page, box = self._page()

        # ── 目标点云 ──────────────────────────────────────────────────────
        tgt_group = QtWidgets.QGroupBox("目标点云（绿色，聚类结果）")
        tgt_layout = QtWidgets.QFormLayout(tgt_group)
        tgt_row = QtWidgets.QHBoxLayout()
        self.reg_target_file_edit = QtWidgets.QLineEdit()
        tgt_browse_btn = QtWidgets.QPushButton("浏览...")
        tgt_browse_btn.clicked.connect(self._browse_reg_target_file)
        tgt_row.addWidget(self.reg_target_file_edit)
        tgt_row.addWidget(tgt_browse_btn)
        tgt_layout.addRow("LAS 文件", tgt_row)
        load_tgt_btn = QtWidgets.QPushButton("加载目标点云并显示")
        load_tgt_btn.clicked.connect(self._load_reg_target)
        tgt_layout.addRow("", load_tgt_btn)
        tgt_ctrl_row = QtWidgets.QHBoxLayout()
        self.reg_show_target_cb = QtWidgets.QCheckBox("显示目标点云")
        self.reg_show_target_cb.setChecked(True)
        self.reg_show_target_cb.toggled.connect(self._toggle_reg_target_visible)
        clear_tgt_btn = QtWidgets.QPushButton("清除目标")
        clear_tgt_btn.clicked.connect(self._clear_reg_target)
        tgt_ctrl_row.addWidget(self.reg_show_target_cb)
        tgt_ctrl_row.addWidget(clear_tgt_btn)
        tgt_layout.addRow("", tgt_ctrl_row)
        self.reg_target_info = QtWidgets.QLabel("默认使用功能二中保留的聚类点；也可在这里单独加载聚类点云 LAS。")
        self.reg_target_info.setWordWrap(True)
        tgt_layout.addRow("状态", self.reg_target_info)
        box.addWidget(tgt_group)

        # ── 源点云 ──────────────────────────────────────────────────────
        src_group = QtWidgets.QGroupBox("源点云（红色）")
        src_layout = QtWidgets.QFormLayout(src_group)
        file_row = QtWidgets.QHBoxLayout()
        self.reg_file_edit = QtWidgets.QLineEdit()
        reg_browse_btn = QtWidgets.QPushButton("浏览...")
        reg_browse_btn.clicked.connect(self._browse_reg_file)
        file_row.addWidget(self.reg_file_edit)
        file_row.addWidget(reg_browse_btn)
        src_layout.addRow("LAS/TRC 文件", file_row)

        self.reg_scale_spin = QtWidgets.QDoubleSpinBox()
        self.reg_scale_spin.setRange(1e-9, 1e9)
        self.reg_scale_spin.setValue(1.0)
        self.reg_scale_spin.setDecimals(9)
        self.reg_scale_spin.setToolTip("毫米→米: 0.001  英寸→米: 0.0254")
        src_layout.addRow("单位缩放系数", self.reg_scale_spin)

        load_src_btn = QtWidgets.QPushButton("加载源点云并显示")
        load_src_btn.clicked.connect(self._load_reg_source)
        src_layout.addRow("", load_src_btn)
        src_ctrl_row = QtWidgets.QHBoxLayout()
        self.reg_show_source_cb = QtWidgets.QCheckBox("显示源点云")
        self.reg_show_source_cb.setChecked(True)
        self.reg_show_source_cb.toggled.connect(self._toggle_reg_source_visible)
        clear_src_btn = QtWidgets.QPushButton("清除源")
        clear_src_btn.clicked.connect(self._clear_reg_source)
        src_ctrl_row.addWidget(self.reg_show_source_cb)
        src_ctrl_row.addWidget(clear_src_btn)
        src_layout.addRow("", src_ctrl_row)
        box.addWidget(src_group)

        # ── 视图与坐标系 ──────────────────────────────────────────────
        view_group = QtWidgets.QGroupBox("视图与坐标系")
        view_layout = QtWidgets.QFormLayout(view_group)
        self.reg_view_combo = QtWidgets.QComboBox()
        self.reg_view_combo.addItems(["自由视图", "俯视 XY", "前视 XZ", "右视 YZ", "等轴测"])
        self.reg_view_combo.currentIndexChanged.connect(self._change_reg_view_mode)
        view_layout.addRow("视图", self.reg_view_combo)
        self.reg_drag_step = QtWidgets.QDoubleSpinBox()
        self.reg_drag_step.setRange(1e-6, 1000.0)
        self.reg_drag_step.setDecimals(6)
        self.reg_drag_step.setValue(0.005)
        self.reg_drag_step.setSuffix(" m/px")
        self.reg_drag_step.setToolTip("鼠标平移时每拖动 1 像素对应的真实坐标位移。毫米点云可用 1~5，米点云可用 0.001~0.01。")
        view_layout.addRow("平移步长", self.reg_drag_step)
        self.reg_frame_info = QtWidgets.QLabel("右侧显示世界坐标系：红=X，绿=Y，蓝=Z。")
        self.reg_frame_info.setStyleSheet("color:#555; font-size:11px;")
        self.reg_frame_info.setWordWrap(True)
        view_layout.addRow("坐标系", self.reg_frame_info)
        box.addWidget(view_group)

        # ── 鼠标交互变换（类 CloudCompare） ────────────────────────────
        im_group = QtWidgets.QGroupBox("鼠标交互变换（类 CloudCompare）")
        im_layout = QtWidgets.QVBoxLayout(im_group)
        self.reg_tm_btn = QtWidgets.QPushButton("激活交互变换模式")
        self.reg_tm_btn.setCheckable(True)
        self.reg_tm_btn.setObjectName("primary")
        self.reg_tm_btn.clicked.connect(self._toggle_reg_transform_mode)
        im_layout.addWidget(self.reg_tm_btn)
        hint_lines = [
            "激活后在右侧视图操作红色点云:",
            "  左键拖拽 → 旋转（绕相机轴）",
            "  Shift+左键 / 右键拖拽 → 按当前视图平面做真实坐标平移",
            "激活期间相机视角固定；完成后点击「提交变换」。",
        ]
        hint = QtWidgets.QLabel("\n".join(hint_lines))
        hint.setStyleSheet("color:#555; font-size:11px;")
        im_layout.addWidget(hint)
        commit_row = QtWidgets.QHBoxLayout()
        commit_btn = QtWidgets.QPushButton("✓ 提交变换")
        commit_btn.clicked.connect(self._commit_interactive_transform)
        cancel_btn = QtWidgets.QPushButton("✗ 取消变换")
        cancel_btn.clicked.connect(self._cancel_interactive_transform)
        commit_row.addWidget(commit_btn)
        commit_row.addWidget(cancel_btn)
        im_layout.addLayout(commit_row)
        box.addWidget(im_group)

        # ── 精确旋转 & 平移 ───────────────────────────────────────────
        rot_group = QtWidgets.QGroupBox("精确旋转 & 平移（增量叠加）")
        rot_layout = QtWidgets.QFormLayout(rot_group)
        self.reg_rot_x = QtWidgets.QDoubleSpinBox()
        self.reg_rot_y = QtWidgets.QDoubleSpinBox()
        self.reg_rot_z = QtWidgets.QDoubleSpinBox()
        for sp in (self.reg_rot_x, self.reg_rot_y, self.reg_rot_z):
            sp.setRange(-360.0, 360.0); sp.setDecimals(3); sp.setSuffix("°")
        rot_layout.addRow("绕 X 旋转（增量）", self.reg_rot_x)
        rot_layout.addRow("绕 Y 旋转（增量）", self.reg_rot_y)
        rot_layout.addRow("绕 Z 旋转（增量）", self.reg_rot_z)
        apply_rot_btn = QtWidgets.QPushButton("应用旋转")
        apply_rot_btn.clicked.connect(self._apply_reg_rotation)
        rot_layout.addRow("", apply_rot_btn)
        self.reg_tx = QtWidgets.QDoubleSpinBox()
        self.reg_ty = QtWidgets.QDoubleSpinBox()
        self.reg_tz = QtWidgets.QDoubleSpinBox()
        for sp in (self.reg_tx, self.reg_ty, self.reg_tz):
            sp.setRange(-1e6, 1e6); sp.setDecimals(4); sp.setSuffix(" m")
        rot_layout.addRow("沿 X 平移（增量）", self.reg_tx)
        rot_layout.addRow("沿 Y 平移（增量）", self.reg_ty)
        rot_layout.addRow("沿 Z 平移（增量）", self.reg_tz)
        apply_t_btn = QtWidgets.QPushButton("应用平移")
        apply_t_btn.clicked.connect(self._apply_reg_translation)
        rot_layout.addRow("", apply_t_btn)
        reset_all_btn = QtWidgets.QPushButton("重置所有旋转 & 平移")
        reset_all_btn.clicked.connect(self._reset_reg_rotation)
        rot_layout.addRow("", reset_all_btn)
        self.reg_rot_info = QtWidgets.QLabel("当前变换: 无")
        self.reg_rot_info.setWordWrap(True)
        self.reg_rot_info.setStyleSheet("color:#444; font-size:11px;")
        rot_layout.addRow("状态", self.reg_rot_info)
        box.addWidget(rot_group)

        # ── GICP ────────────────────────────────────────────────────────
        gicp_group = QtWidgets.QGroupBox("GICP 精配准（需要 open3d: pip install open3d）")
        gicp_layout = QtWidgets.QFormLayout(gicp_group)
        self.gicp_max_dist = QtWidgets.QDoubleSpinBox()
        self.gicp_max_dist.setRange(0.001, 10000.0)
        self.gicp_max_dist.setValue(0.5)
        self.gicp_max_dist.setDecimals(4)
        self.gicp_max_dist.setSuffix(" m")
        gicp_layout.addRow("最大对应距离", self.gicp_max_dist)
        gicp_btn = QtWidgets.QPushButton("运行 GICP 配准")
        gicp_btn.setObjectName("primary")
        gicp_btn.clicked.connect(self._run_gicp)
        gicp_layout.addRow("", gicp_btn)
        box.addWidget(gicp_group)

        # ── 结果 ─────────────────────────────────────────────────────────
        self.reg_result_text = QtWidgets.QTextEdit()
        self.reg_result_text.setReadOnly(True)
        self.reg_result_text.setMinimumHeight(180)
        self.reg_result_text.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace; font-size:11px;")
        self.reg_result_text.setPlaceholderText("配准结果（完整位姿变换）将在此显示...")
        box.addWidget(self.reg_result_text, 1)
        self.tabs.addTab(page, "6 点云配准")

    def _browse_reg_target_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择目标聚类点云 LAS 文件", "", "LAS/LAZ 点云 (*.las *.laz)")
        if path:
            self.reg_target_file_edit.setText(path)

    def _load_reg_target(self):
        filepath = self.reg_target_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择目标点云文件。")
            return
        self._set_busy("正在加载目标点云...")

        def work():
            return _read_point_cloud(filepath)[0]

        def done(pts):
            self.reg_target_pts = pts
            self.reg_show_target = True
            if hasattr(self, "reg_show_target_cb"):
                self.reg_show_target_cb.setChecked(True)
            self.reg_target_info.setText(f"已加载目标点云：{len(pts):,} 点")
            self._show_reg_view()
            self._set_ready("目标点云已加载")

        self._run_task(work, done)

    def _toggle_reg_target_visible(self, checked):
        self.reg_show_target = checked
        self._show_reg_view()

    def _toggle_reg_source_visible(self, checked):
        self.reg_show_source = checked
        self._show_reg_view()

    def _clear_reg_target(self):
        self.reg_target_pts = None
        if hasattr(self, "reg_target_file_edit"):
            self.reg_target_file_edit.clear()
        self.reg_target_info.setText("目标点云已清除；若功能二有保留聚类，将默认使用聚类球心。")
        self.view.set_reg_target(None)
        self._show_reg_view()

    def _clear_reg_source(self):
        self.reg_source_pts = None
        self.reg_manual_rot = np.eye(3)
        self.reg_translation = np.zeros(3)
        self.reg_interactive_R = np.eye(3)
        self.reg_interactive_t = np.zeros(3)
        if hasattr(self, "reg_file_edit"):
            self.reg_file_edit.clear()
        if hasattr(self, "reg_tm_btn") and self.reg_tm_btn.isChecked():
            self.reg_tm_btn.setChecked(False)
            self.view.set_transform_mode(False)
        self._update_reg_rot_info()
        self.view.set_reg_source(None)
        self._show_reg_view()

    def _change_reg_view_mode(self, index):
        modes = ("free", "top", "front", "right", "iso")
        self.reg_view_mode = modes[index] if 0 <= index < len(modes) else "free"
        if self.reg_view_mode == "top":
            self.view.setCameraPosition(azimuth=-90, elevation=90)
        elif self.reg_view_mode == "front":
            self.view.setCameraPosition(azimuth=-90, elevation=0)
        elif self.reg_view_mode == "right":
            self.view.setCameraPosition(azimuth=0, elevation=0)
        elif self.reg_view_mode == "iso":
            self.view.setCameraPosition(azimuth=45, elevation=28)
        self.view.update()

    def _browse_reg_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择源点云文件", "",
            "支持的源点云 (*.las *.laz *.trc);;TRC 动捕文件 (*.trc);;LAS/LAZ 点云 (*.las *.laz);;所有文件 (*)")
        if path:
            self.reg_file_edit.setText(path)

    def _load_reg_source(self):
        filepath = self.reg_file_edit.text().strip()
        if not filepath:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择源点云文件。")
            return
        scale = self.reg_scale_spin.value()
        self._set_busy("正在加载源点云...")

        def work():
            return _read_source_points(filepath, scale=scale)

        self._run_task(work, self._reg_source_loaded)

    def _reg_source_loaded(self, pts):
        self.reg_source_pts    = pts
        self.reg_show_source   = True
        self.reg_manual_rot    = np.eye(3)
        self.reg_translation   = np.zeros(3)
        self.reg_interactive_R = np.eye(3)
        self.reg_interactive_t = np.zeros(3)
        if hasattr(self, 'reg_tm_btn') and self.reg_tm_btn.isChecked():
            self.reg_tm_btn.setChecked(False)
            self.view.set_transform_mode(False)
        if hasattr(self, "reg_show_source_cb"):
            self.reg_show_source_cb.setChecked(True)
        self._update_reg_rot_info()
        self._show_reg_view()
        self._set_ready(f"已加载源点云: {len(pts):,} 个点")

    def _update_reg_rot_info(self):
        R = self.reg_manual_rot
        t = getattr(self, 'reg_translation', np.zeros(3))
        ti = getattr(self, 'reg_interactive_t', np.zeros(3))
        parts = [
            f"R·X: [{R[0,0]:+.4f}, {R[1,0]:+.4f}, {R[2,0]:+.4f}]",
            f"R·Y: [{R[0,1]:+.4f}, {R[1,1]:+.4f}, {R[2,1]:+.4f}]",
            f"R·Z: [{R[0,2]:+.4f}, {R[1,2]:+.4f}, {R[2,2]:+.4f}]",
            f"T:   [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m",
        ]
        if not np.allclose(ti, 0):
            parts.append(f"交互中 ΔT: [{ti[0]:+.4f}, {ti[1]:+.4f}, {ti[2]:+.4f}] m")
        self.reg_rot_info.setText("\n".join(parts))

    def _apply_reg_rotation(self):
        if self.reg_source_pts is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载源点云。")
            return
        self._commit_interactive_transform(silent=True)
        rx = self.reg_rot_x.value()
        ry = self.reg_rot_y.value()
        rz = self.reg_rot_z.value()
        Rx = self.api._rot_mat(np.array([1., 0., 0.]), rx)
        Ry = self.api._rot_mat(np.array([0., 1., 0.]), ry)
        Rz = self.api._rot_mat(np.array([0., 0., 1.]), rz)
        R_inc = Rz @ Ry @ Rx
        self.reg_manual_rot = self.api._gs3(R_inc @ self.reg_manual_rot)
        for sp in (self.reg_rot_x, self.reg_rot_y, self.reg_rot_z):
            sp.setValue(0.0)
        self._update_reg_rot_info()
        self._show_reg_view()

    def _reset_reg_rotation(self):
        self.reg_manual_rot    = np.eye(3)
        self.reg_translation   = np.zeros(3)
        self.reg_interactive_R = np.eye(3)
        self.reg_interactive_t = np.zeros(3)
        self._update_reg_rot_info()
        if self.reg_source_pts is not None:
            self._show_reg_view()

    def _get_reg_target_pts(self):
        """当前检测器中有效聚类的球心点（绿色目标）。"""
        if self.reg_target_pts is not None and len(self.reg_target_pts):
            return self.reg_target_pts
        if not self.detector.clusters:
            return None
        active = {k: v for k, v in self.detector.clusters.items()
                  if k not in self.detector.removed_clusters}
        if not active:
            return None
        return np.vstack([v['center'] for _, v in sorted(active.items())])

    def _get_base_source(self):
        """手动旋转 + 手动平移（不含交互增量）。"""
        if self.reg_source_pts is None:
            return None
        c = np.mean(self.reg_source_pts, axis=0)
        rotated = (self.reg_manual_rot @ (self.reg_source_pts - c).T).T + c
        return rotated + self.reg_translation

    def _get_transformed_source(self):
        """手动变换 + 交互增量（用于 GICP 输入和显示）。"""
        base = self._get_base_source()
        if base is None:
            return None
        c = np.mean(base, axis=0)
        rotated = (self.reg_interactive_R @ (base - c).T).T + c
        return rotated + self.reg_interactive_t

    def _show_reg_view(self):
        """在 3D 视图中同时显示红色源点云和绿色目标点云。"""
        target_pts = self._get_reg_target_pts()
        source_pts = self._get_transformed_source()
        frame_pts = [pts for pts in (target_pts, source_pts) if pts is not None and len(pts)]
        if frame_pts:
            self.view.set_display_origin_from_points(np.vstack(frame_pts))
        self.view.clear_primary_cloud()
        if frame_pts:
            span = np.ptp(np.vstack(frame_pts), axis=0)
            frame_scale = max(float(np.linalg.norm(span)) * 0.18, 0.1)
            self.view.show_coordinate_frame(np.zeros(3), np.eye(3), frame_scale)
        if target_pts is not None and self.reg_show_target:
            self.view.set_reg_target(target_pts)
        else:
            self.view.set_reg_target(None)
        if source_pts is not None and self.reg_show_source:
            self.view.set_reg_source(source_pts)
        else:
            self.view.set_reg_source(None)

    def _run_gicp(self):
        if self.reg_source_pts is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载源点云。")
            return
        target_pts = self._get_reg_target_pts()
        if target_pts is None or not len(target_pts):
            QtWidgets.QMessageBox.warning(
                self, "错误",
                "请先加载目标球心点云，或完成聚类并确保至少有一个有效聚类球心。")
            return

        self._commit_interactive_transform(silent=True)
        source_transformed = self._get_transformed_source()
        max_dist     = self.gicp_max_dist.value()
        R_manual     = self.reg_manual_rot.copy()
        t_manual_ext = self.reg_translation.copy()
        src_pts      = self.reg_source_pts.copy()
        src_ctr      = np.mean(src_pts, axis=0)

        def work():
            try:
                import open3d as o3d
            except ImportError:
                raise RuntimeError(
                    "GICP 需要 open3d，请先安装：\n  pip install open3d")

            estimator = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=300, relative_fitness=1e-7, relative_rmse=1e-7)

            stage_results = []
            dist = float(max_dist)
            dist_step = 0.01
            min_dist = max(dist_step, 1e-5)
            stage_idx = 1
            while dist >= min_dist and stage_idx <= 100:
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(source_transformed.copy())
                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points = o3d.utility.Vector3dVector(target_pts.copy())

                r_nn = max(dist * 3.0, 0.05)
                src_pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=r_nn, max_nn=30))
                tgt_pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=r_nn, max_nn=30))

                result = o3d.pipelines.registration.registration_generalized_icp(
                    src_pcd, tgt_pcd,
                    max_correspondence_distance=dist,
                    init=np.eye(4),
                    estimation_method=estimator,
                    criteria=criteria)
                T_stage = np.asarray(result.transformation)
                fitness = float(result.fitness)
                rmse = float(result.inlier_rmse)
                n_inliers = int(round(fitness * len(source_transformed)))
                stage_results.append({
                    'stage': stage_idx,
                    'max_dist': dist,
                    'T_gicp': T_stage,
                    'fitness': fitness,
                    'inlier_rmse': rmse,
                    'n_inliers': n_inliers,
                })
                dist -= dist_step
                stage_idx += 1

            eligible = [r for r in stage_results if r['fitness'] >= 0.90]
            if eligible:
                best_stage = min(eligible, key=lambda r: r['inlier_rmse'])
                selected_rule = "fitness >= 90% 中 RMSE 最小"
            else:
                best_stage = max(stage_results, key=lambda r: (r['fitness'], -r['inlier_rmse']))
                selected_rule = "无阶段达到 90%，选择 fitness 最高且 RMSE 较低"

            T_gicp = best_stage['T_gicp']  # 4×4

            # 手动变换 4×4: 绕 src_ctr 旋转 + 额外平移
            t_man = (np.eye(3) - R_manual) @ src_ctr + t_manual_ext
            T_manual = np.eye(4)
            T_manual[:3, :3] = R_manual
            T_manual[:3, 3]  = t_man

            # 全变换：T_full = T_gicp @ T_manual
            # 注意：缩放系数已在加载时应用到 src_pts，所以 T_full 基于已缩放坐标
            T_full = T_gicp @ T_manual

            # 对齐后的点（用于可视化）
            src_arr = np.asarray(source_transformed, dtype=np.float64)
            ones = np.ones((len(src_arr), 1))
            aligned = (T_gicp @ np.hstack([src_arr, ones]).T).T[:, :3]

            return {
                'T_gicp': T_gicp,
                'T_manual': T_manual,
                'T_full': T_full,
                'fitness': float(best_stage['fitness']),
                'inlier_rmse': float(best_stage['inlier_rmse']),
                'n_inliers': int(best_stage['n_inliers']),
                'aligned_pts': aligned,
                'stage_results': stage_results,
                'best_stage': best_stage['stage'],
                'selected_rule': selected_rule,
            }

        self._set_busy("正在运行 GICP 配准...")
        self._run_task(work, self._gicp_done)

    def _gicp_done(self, result):
        T_full   = result['T_full']
        T_gicp   = result['T_gicp']
        T_manual = result['T_manual']
        R_full = T_full[:3, :3]
        t_full = T_full[:3, 3]
        aligned = result['aligned_pts']

        # 视图：配准后的源点云改为橙色（接近绿色目标说明对齐好）
        self.view.set_reg_source(aligned, color=(1.0, 0.55, 0.1, 0.85), size=3)

        def fmt_mat(M, lbl):
            rows = "\n".join(
                "  [" + ", ".join(f"{v:+10.6f}" for v in row) + "]"
                for row in M)
            return f"{lbl}:\n{rows}"

        scale = self.reg_scale_spin.value()
        scale_note = (f"  (源点云已预乘缩放系数 {scale:.6g}，"
                      f"T_full 中 t 的单位与目标点云一致)") if scale != 1.0 else ""
        stage_lines = []
        for stage in result.get('stage_results', []):
            marker = "  <-- 采用" if stage['stage'] == result.get('best_stage') else ""
            stage_lines.append(
                f"  #{stage['stage']:02d}  maxDist={stage['max_dist']:.6f} m  "
                f"匹配点={stage['n_inliers']}/{len(aligned)}  "
                f"fitness={stage['fitness']:.6f}  "
                f"RMSE={stage['inlier_rmse']*1000:.4f} mm{marker}"
            )
        stage_text = "\n".join(stage_lines) if stage_lines else "  <无阶段记录>"

        text = (
            f"{'='*50}\n"
            f"多阶段 GICP 配准结果\n"
            f"{'='*50}\n"
            f"选择规则: {result.get('selected_rule', 'fitness >= 90% 中 RMSE 最小')}\n"
            f"采用阶段: #{result.get('best_stage', 1)}\n"
            f"fitness (内点比例) : {result['fitness']:.6f}\n"
            f"inlier RMSE       : {result['inlier_rmse']*1000:.4f} mm\n"
            f"匹配点数           : {result['n_inliers']} / {len(aligned)}\n"
            f"{'─'*50}\n"
            f"阶段记录（每阶段完全独立：重新建点云/法向，从当前手动初参开始；距离每轮减少 0.01 m）:\n"
            f"{stage_text}\n"
            f"{'─'*50}\n"
            f"{fmt_mat(T_gicp, 'T_gicp  (GICP增量变换 4×4)')}\n\n"
            f"{fmt_mat(T_manual, 'T_manual (手动旋转变换 4×4)')}\n\n"
            f"{'='*50}\n"
            f"红色→绿色 完整位姿变换\n"
            f"T_full = T_gicp @ T_manual\n"
            f"{'='*50}\n"
            f"{fmt_mat(T_full, 'T_full (4×4)')}\n\n"
            f"旋转矩阵 R (3×3):\n"
            f"{fmt_mat(R_full, '  R')}\n\n"
            f"平移向量 t:\n"
            f"  [{t_full[0]:+.6f}, {t_full[1]:+.6f}, {t_full[2]:+.6f}]\n"
            f"{scale_note}\n\n"
            f"使用方法:\n"
            f"  p_green = R @ p_red_scaled + t\n"
            f"  p_red_scaled = p_red_original * {scale:.6g}"
        )
        self.reg_result_text.setPlainText(text)
        self._set_ready(
            f"多阶段GICP完成: 阶段#{result.get('best_stage', 1)}  fitness={result['fitness']:.4f}  "
            f"RMSE={result['inlier_rmse']*1000:.3f}mm")

    def _apply_reg_translation(self):
        if self.reg_source_pts is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载源点云。"); return
        self._commit_interactive_transform(silent=True)
        self.reg_translation += np.array(
            [self.reg_tx.value(), self.reg_ty.value(), self.reg_tz.value()])
        for sp in (self.reg_tx, self.reg_ty, self.reg_tz): sp.setValue(0.0)
        self._update_reg_rot_info()
        self._show_reg_view()

    def _toggle_reg_transform_mode(self, checked):
        if checked:
            if self.reg_source_pts is None:
                self.reg_tm_btn.setChecked(False)
                QtWidgets.QMessageBox.warning(self, "错误", "请先加载源点云。"); return
            self.view.set_transform_mode(
                True,
                rotate_cb=self._on_reg_rotate_drag,
                translate_cb=self._on_reg_translate_drag,
                release_cb=self._on_reg_drag_release)
            self.reg_tm_btn.setText("交互模式已激活（再次点击退出）")
        else:
            self.view.set_transform_mode(False)
            self.reg_tm_btn.setText("激活交互变换模式")

    def _on_reg_rotate_drag(self, dx, dy):
        sensitivity = 0.4
        azim_rad = np.radians(self.view.opts.get('azimuth', 0))
        elev_rad = np.radians(self.view.opts.get('elevation', 30))
        cam_right = np.array([np.cos(azim_rad), np.sin(azim_rad), 0.0])
        cam_up = np.array([-np.sin(azim_rad) * np.sin(elev_rad),
                            np.cos(azim_rad) * np.sin(elev_rad),
                            np.cos(elev_rad)])
        R_yaw   = self.api._rot_mat(cam_up,    dx * sensitivity)
        R_pitch = self.api._rot_mat(cam_right, dy * sensitivity)
        R_delta = self.api._gs3(R_pitch @ R_yaw)
        self.reg_interactive_R = self.api._gs3(R_delta @ self.reg_interactive_R)
        self._update_reg_interactive_view()

    def _on_reg_translate_drag(self, dx, dy):
        step = self.reg_drag_step.value() if hasattr(self, "reg_drag_step") else 0.005
        if self.reg_view_mode == "top":
            delta = np.array([dx, -dy, 0.0]) * step
        elif self.reg_view_mode == "front":
            delta = np.array([dx, 0.0, -dy]) * step
        elif self.reg_view_mode == "right":
            delta = np.array([0.0, dx, -dy]) * step
        else:
            azim_rad = np.radians(self.view.opts.get('azimuth', 0))
            elev_rad = np.radians(self.view.opts.get('elevation', 30))
            cam_right = np.array([np.cos(azim_rad), np.sin(azim_rad), 0.0])
            cam_up = np.array([-np.sin(azim_rad) * np.sin(elev_rad),
                                np.cos(azim_rad) * np.sin(elev_rad),
                                np.cos(elev_rad)])
            delta = (cam_right * dx - cam_up * dy) * step
        self.reg_interactive_t += delta
        self._update_reg_interactive_view()

    def _on_reg_drag_release(self):
        pass

    def _update_reg_interactive_view(self):
        if self.reg_source_pts is None: return
        base = self._get_base_source()
        c = np.mean(base, axis=0)
        pts = (self.reg_interactive_R @ (base - c).T).T + c + self.reg_interactive_t
        if self.reg_show_source:
            self.view.set_reg_source(pts)
        self._update_reg_rot_info()

    def _commit_interactive_transform(self, silent=False):
        if (np.allclose(self.reg_interactive_R, np.eye(3)) and
                np.allclose(self.reg_interactive_t, 0)):
            return
        if self.reg_source_pts is None: return
        c_raw  = np.mean(self.reg_source_pts, axis=0)
        R_m = self.reg_manual_rot
        t_m = (np.eye(3) - R_m) @ c_raw + self.reg_translation
        T_man = np.eye(4); T_man[:3,:3] = R_m; T_man[:3,3] = t_m
        base   = self._get_base_source()
        c_base = np.mean(base, axis=0)
        R_i = self.reg_interactive_R
        t_i = (np.eye(3) - R_i) @ c_base + self.reg_interactive_t
        T_int = np.eye(4); T_int[:3,:3] = R_i; T_int[:3,3] = t_i
        T_comb = T_int @ T_man
        R_new  = self.api._gs3(T_comb[:3,:3])
        t_new  = T_comb[:3, 3]
        self.reg_manual_rot    = R_new
        self.reg_translation   = t_new - (np.eye(3) - R_new) @ c_raw
        self.reg_interactive_R = np.eye(3)
        self.reg_interactive_t = np.zeros(3)
        if not silent:
            self._update_reg_rot_info()
            self._show_reg_view()

    def _cancel_interactive_transform(self):
        self.reg_interactive_R = np.eye(3)
        self.reg_interactive_t = np.zeros(3)
        self._show_reg_view()

    def _on_reg_tab_shown(self):
        if self.reg_source_pts is not None or self.reg_target_pts is not None:
            self._show_reg_view()
        elif self.detector.clusters:
            self._show_reg_view()
        else:
            if self._has_cloud():
                self._set_cloud(self.detector.points, self.detector.intensities)
            self.view.clear_reg()


def run_unified_desktop(detector, args, api):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("LAS 高反球检测")
    app.setStyle("Fusion")
    window = UnifiedWindow(detector, args, api)
    window.show()
    return app.exec()
