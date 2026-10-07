"""Resizable RGB rendering for the Demo's existing Tk viewport.

Only the display-owned OpenGL context/buffer changes size. The executed model,
its visual settings and its data remain untouched.
"""
from __future__ import annotations

import mujoco
import numpy as np


class ViewportRenderer:
    """Reuse a scene and GL context while the canvas changes dimensions."""

    def __init__(self, model, *, width: int, height: int, max_geom: int):
        self.model = model
        self.scene = mujoco.MjvScene(model=model, maxgeom=max_geom)
        self._scene_option = mujoco.MjvOption()
        self._gl_context = None
        self._mjr_context = None
        self.width = self.height = 0
        try:
            # The hidden native window only owns the context; pixels are drawn
            # into the separately sized offscreen framebuffer.
            self._gl_context = mujoco.GLContext(1, 1)
            self._gl_context.make_current()
            self._mjr_context = mujoco.MjrContext(
                model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
            self.resize(width, height)
        except Exception:
            self.close()
            raise

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return
        if self._mjr_context is None:
            raise RuntimeError("Cannot resize a closed viewport renderer")
        self._gl_context.make_current()
        mujoco.mjr_resizeOffscreen(width, height, self._mjr_context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_context)
        error = mujoco.mjr_getError()
        if error:
            raise RuntimeError(f"Cannot allocate {width}×{height} viewport buffer (OpenGL error {error})")
        self.width, self.height = width, height
        self._rect = mujoco.MjrRect(0, 0, width, height)

    def update_scene(self, data, *, camera) -> None:
        mujoco.mjv_updateScene(self.model, data, self._scene_option, None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)

    def render(self) -> np.ndarray:
        if self._mjr_context is None:
            raise RuntimeError("Cannot render a closed viewport renderer")
        self._gl_context.make_current()
        pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_render(self._rect, self.scene, self._mjr_context)
        mujoco.mjr_readPixels(pixels, None, self._rect, self._mjr_context)
        # OpenGL uses a bottom-left origin; Tk images use a top-left origin.
        pixels[:] = np.flipud(pixels)
        return pixels

    def close(self) -> None:
        if self._mjr_context is not None:
            self._gl_context.make_current()
            self._mjr_context.free()
            self._mjr_context = None
        if self._gl_context is not None:
            self._gl_context.free()
            self._gl_context = None
