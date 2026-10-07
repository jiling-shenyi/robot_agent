"""Tk display of the live MuJoCo model, with visual-only world coordinates.

The widget deliberately creates no GL resources until ``render(episode)``.
Call it on the Tk/main thread, between physics steps: the renderer reads the
episode's actual model/data, without a second simulation or state copy.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import mujoco
import numpy as np

from .viewport_renderer import ViewportRenderer


_IDENTITY = np.eye(3, dtype=np.float64).ravel()
_AXIS_COLORS = (
    (1.0, 0.28, 0.24, 1.0),
    (0.30, 0.92, 0.40, 1.0),
    (0.30, 0.58, 1.0, 1.0),
)


def _geom(scene: mujoco.MjvScene, kind: Any, position: Any, color: Any) -> Any:
    """Allocate a rendering geom; never add collision objects to the model."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("World coordinate geometry exceeds renderer capacity")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.zeros(3, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        _IDENTITY,
        np.asarray(color, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1
    return geom


def _line(scene: Any, start: Any, end: Any, color: Any, width: float = 1.0) -> None:
    geom = _geom(scene, mujoco.mjtGeom.mjGEOM_LINE, start, color)
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )


def _label(scene: Any, position: Any, text: str, color: Any) -> None:
    geom = _geom(scene, mujoco.mjtGeom.mjGEOM_LABEL, position, color)
    geom.label = text


def add_world_coordinates(scene: mujoco.MjvScene, table_top_z: float = 0.4) -> None:
    """Append true world axes and a separately identified tabletop XY grid.

    Axis tick positions are in metres and always measured from (0, 0, 0).
    The grid sits 1 mm above the tabletop to avoid depth flicker; its labels
    explicitly identify the reference plane, rather than pretending it is Z=0.
    """
    for axis, name in enumerate(("X", "Y", "Z")):
        endpoint = np.zeros(3)
        endpoint[axis] = 1.05 if axis != 1 else 0.65
        geom = _geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), _AXIS_COLORS[axis])
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, np.zeros(3), endpoint
        )
        # A pixel-width line keeps the long axis legible at the default zoom,
        # even where the narrow 3D arrow shaft projects to less than one pixel.
        _line(scene, np.zeros(3), endpoint, _AXIS_COLORS[axis], 2.0)
        label_position = endpoint.copy()
        label_position[axis] += 0.05
        _label(scene, label_position, f"{name} (m)", _AXIS_COLORS[axis])
        tick_direction = np.zeros(3)
        tick_direction[1 if axis == 0 else 0] = 0.012
        for value in np.arange(0.2, endpoint[axis], 0.2):
            position = np.zeros(3)
            position[axis] = value
            _line(scene, position - tick_direction, position + tick_direction, _AXIS_COLORS[axis], 2.0)
            # Ground labels behind the table would be drawn over its surface
            # by MuJoCo's text pass. Keep major numbers on the visible ends;
            # the explicitly elevated XY grid supplies intermediate readings.
            visible_major = (
                (axis == 0 and value > 0.9)
                or (axis == 1 and value > 0.5)
                or (axis == 2 and value > 0.7)
            )
            if visible_major:
                _label(scene, position + tick_direction * 3, f"{value:.1f}", _AXIS_COLORS[axis])
    _line(scene, (0, -0.55, 0), (0, 0, 0), _AXIS_COLORS[1], 2.0)
    for value in (-0.4, -0.2):
        _line(scene, (-0.012, value, 0), (0.012, value, 0), _AXIS_COLORS[1], 2.0)
        if value == -0.4:
            _label(scene, (0.035, value, 0), f"{value:.1f}", _AXIS_COLORS[1])
    _label(scene, (-0.18, -0.15, 0.02), "O (0,0,0)", (0.92, 0.92, 0.92, 1))

    z = float(table_top_z) + 0.001
    grid_color = (0.68, 0.70, 0.73, 0.45)
    for x in np.arange(0.2, 1.0, 0.1):
        _line(scene, (x, -0.37, z), (x, 0.37, z), grid_color)
    for y in np.arange(-0.3, 0.4, 0.1):
        _line(scene, (0.18, y, z), (0.92, y, z), grid_color)
    for x in (0.2, 0.5, 0.8):
        _label(scene, (x, -0.41, z), f"x={x:.1f}", _AXIS_COLORS[0])
    for y in (-0.2, 0.0, 0.2):
        _label(scene, (0.99, y, z), f"y={y:.1f}", _AXIS_COLORS[1])
    _label(scene, (0.58, 0.43, z), f"XY grid: z={table_top_z:.2f} m", (0.95, 0.95, 0.95, 1))


def add_home_coordinates(scene: mujoco.MjvScene, snapshot: dict[str, Any]) -> None:
    """Decorate the actual room state; every added geom is rendering-only."""
    width, depth, _ = snapshot["room"]["size_m"]
    z = .004
    color = (.43, .47, .52, .45)
    for x in np.arange(-width / 2 + .5, width / 2, .5):
        _line(scene, (x, -depth / 2 + .04, z), (x, depth / 2 - .04, z), color)
    for y in np.arange(-depth / 2 + .5, depth / 2, .5):
        _line(scene, (-width / 2 + .04, y, z), (width / 2 - .04, y, z), color)
    for axis, name in enumerate(("X", "Y", "Z")):
        start = np.array((0., 0., z))
        end = start.copy()
        end[axis] += 1.
        geom = _geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, start, _AXIS_COLORS[axis])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .012, start, end)
        _label(scene, end + np.array((.04, .04, .04)), f"{name} (m)", _AXIS_COLORS[axis])
    _label(scene, (0, -depth / 2 + .30, .01), "Floor grid: z=0 m, step=0.5 m", (.96, .96, .96, 1))
    for name, obj in snapshot.get("objects", {}).items():
        position = np.array(obj["position_m"], dtype=float)
        position[2] += float(obj["half_size_m"][2]) + .08
        _label(scene, position, name, (.98, .98, .98, 1))
    for risk in snapshot.get("risks", []):
        geom = _geom(scene, mujoco.mjtGeom.mjGEOM_BOX, risk["position_m"], (1., .24, .07, .16))
        geom.size[:] = risk["half_size_m"]


class WorldView(ttk.Frame):
    """Embedded live viewer; drag to orbit, wheel to zoom, double-click to reset.

    Render resolution follows the actual canvas size, including maximization;
    window resizing reuses the GL context and preserves the current camera.

    ``render`` returns the RGB frame, or ``None`` on error. The readable error
    remains in ``last_error`` and on the canvas, allowing the host to stop an
    interactive run instead of silently continuing without visualization.
    ``close`` is idempotent and a later render may open a new renderer.
    """

    def __init__(self, master: Any, width: int = 800, height: int = 520, **kwargs: Any):
        super().__init__(master, **kwargs)
        self.render_width = width
        self.render_height = height
        self.last_error: str | None = None
        self._renderer: ViewportRenderer | None = None
        self._model: mujoco.MjModel | None = None
        self._episode: Any = None
        self._camera_kind = "panda"
        self._photo: tk.PhotoImage | None = None
        self._drag_position: tuple[int, int] | None = None
        self._redraw_pending: str | None = None
        self._resize_pending: str | None = None
        self.camera = mujoco.MjvCamera()
        self.reset_camera()

        self.canvas = tk.Canvas(
            self, width=width, height=height, background="#171e2b", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self._image_item = self.canvas.create_image(width / 2, height / 2, anchor="center")
        self._message_item = self.canvas.create_text(
            width / 2,
            height / 2,
            text="请选择地图，再加载模拟世界。",
            fill="#d8e2ef",
            font=("Microsoft YaHei UI", 14),
            justify="center",
            width=width - 40,
        )
        self.coordinate_hint = ttk.Label(
            self,
            text="世界坐标 / 米：X 红 · Y 绿 · Z 蓝 | 桌面网格标注实际 Z 高度 | 左键拖动旋转 · 滚轮缩放 · 双击恢复视角",
            anchor="center",
        )
        self.coordinate_hint.pack(fill="x", pady=(3, 0))
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _: setattr(self, "_drag_position", None))
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.canvas.bind("<Button-4>", self._zoom)
        self.canvas.bind("<Button-5>", self._zoom)
        self.canvas.bind("<Double-Button-1>", self.reset_camera)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def reset_camera(self, _event: Any = None) -> None:
        mujoco.mjv_defaultCamera(self.camera)
        if self._camera_kind == "stretch":
            room = getattr(self._episode, "world", None)
            size = room.room.size_m if room is not None else (6., 5., 2.4)
            self.camera.lookat[:] = (0., 0., .6)
            self.camera.distance = max(size[:2]) * 1.35
            self.camera.azimuth = 125.
            self.camera.elevation = -55.
        else:
            self.camera.lookat[:] = (0.45, 0.0, 0.40)
            self.camera.distance = 2.1
            self.camera.azimuth = 135.0
            self.camera.elevation = -27.0
        if hasattr(self, "canvas"):
            self._queue_redraw()

    def _on_configure(self, event: Any) -> None:
        for item in (self._image_item, self._message_item):
            self.canvas.coords(item, event.width / 2, event.height / 2)
        self.canvas.itemconfigure(self._message_item, width=max(100, event.width - 40))
        size = (max(1, event.width), max(1, event.height))
        if size != (self.render_width, self.render_height):
            self.render_width, self.render_height = size
            if self._resize_pending is not None:
                self.after_cancel(self._resize_pending)
                self._resize_pending = None
            if self._episode is not None:
                # Coalesce a burst of native window sizing events. Execution
                # frames still render at the current size between these events.
                self._resize_pending = self.after(100, self._resize_redraw)

    def _resize_redraw(self) -> None:
        self._resize_pending = None
        self._queue_redraw()

    def _start_drag(self, event: Any) -> None:
        self._drag_position = (event.x, event.y)

    def _drag(self, event: Any) -> None:
        if self._drag_position is None:
            return
        old_x, old_y = self._drag_position
        self.camera.azimuth -= (event.x - old_x) * 0.4
        self.camera.elevation = float(np.clip(self.camera.elevation - (event.y - old_y) * 0.3, -89, -5))
        self._drag_position = (event.x, event.y)
        self._queue_redraw()

    def _zoom(self, event: Any) -> None:
        delta = getattr(event, "delta", 0)
        if not delta:
            delta = 120 if getattr(event, "num", 0) == 4 else -120
        factor = 0.88 if delta > 0 else 1 / 0.88
        maximum = 14. if self._camera_kind == "stretch" else 5.
        self.camera.distance = float(np.clip(self.camera.distance * factor, 0.7, maximum))
        self._queue_redraw()

    def _queue_redraw(self) -> None:
        if self._episode is not None and self._redraw_pending is None:
            self._redraw_pending = self.after_idle(self._redraw)

    def _redraw(self) -> None:
        self._redraw_pending = None
        if self._episode is not None:
            self.render(self._episode)

    def render(self, episode: Any) -> np.ndarray | None:
        try:
            kind = getattr(episode, "robot_kind", "panda")
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            width = width if width > 1 else self.render_width
            height = height if height > 1 else self.render_height
            if self._model is not episode.model or self._renderer is None:
                self.close()
                self._episode = episode
                if kind != self._camera_kind:
                    self._camera_kind = kind
                    self.reset_camera()
                self._renderer = ViewportRenderer(
                    episode.model, width=width, height=height,
                    max_geom=max(1000, episode.model.ngeom * 2 + 200),
                )
                self._model = episode.model
            else:
                self._renderer.resize(width, height)
            if self._resize_pending is not None:
                self.after_cancel(self._resize_pending)
                self._resize_pending = None
            self._episode = episode
            self._renderer.update_scene(episode.data, camera=self.camera)
            if kind == "stretch":
                add_home_coordinates(self._renderer.scene, episode.snapshot())
                self.coordinate_hint.configure(text="世界坐标 / 米：X 红 · Y 绿 · Z 蓝 | 地面网格 z=0，间距0.5米 | 标签和风险区域读取当前状态 | 拖动旋转 · 滚轮缩放 · 双击恢复")
            else:
                add_world_coordinates(self._renderer.scene, getattr(episode, "table_top_z", 0.4))
                self.coordinate_hint.configure(text="世界坐标 / 米：X 红 · Y 绿 · Z 蓝 | 桌面网格标注实际 Z 高度 | 拖动旋转 · 滚轮缩放 · 双击恢复")
            pixels = self._renderer.render()
            height, width, _ = pixels.shape
            ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + pixels.tobytes()
            if self._photo is None:
                self._photo = tk.PhotoImage(master=self.canvas, data=ppm, format="PPM")
            else:
                self._photo.configure(data=ppm, format="PPM")
            self.canvas.itemconfigure(self._image_item, image=self._photo, state="normal")
            self.canvas.itemconfigure(self._message_item, text="", state="hidden")
            self.last_error = None
            return pixels
        except Exception as exc:
            # A failed allocation can invalidate the native framebuffer even
            # when Python's previous dimensions still look usable. Rebuild the
            # display context on the next successful render.
            self.close()
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.canvas.itemconfigure(self._image_item, state="hidden")
            self.canvas.itemconfigure(
                self._message_item,
                text=f"模拟世界渲染失败，执行已停止。\n{self.last_error}",
                state="normal",
            )
            return None

    def close(self) -> None:
        if self._resize_pending is not None:
            self.after_cancel(self._resize_pending)
            self._resize_pending = None
        if self._redraw_pending is not None:
            self.after_cancel(self._redraw_pending)
            self._redraw_pending = None
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._model = None
        self._episode = None

    def _on_destroy(self, event: Any) -> None:
        if event.widget is self:
            self.close()
