"""Optional native Viewer hooks; importing this module initializes no display."""
from __future__ import annotations
import atexit
import time
import mujoco
import numpy as np
from embodied_agent.simulation.model import SAMPLE_EVERY_STEPS
from embodied_agent.simulation.errors import ViewerClosed

class EpisodeViewerMixin:
    """Viewer reads the same model/data stepped by the simulator."""

    def _open_viewer(self) -> None:
        if not self.enable_viewer:
            return
        from mujoco import viewer as mujoco_viewer
        # Keep visualization attached to the exact model/data pair being stepped.
        self.viewer = mujoco_viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=self.show_ui,
            show_right_ui=self.show_ui,
        )
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.lookat[:] = np.array([0.48, 0.0, 0.39])
            self.viewer.cam.distance = 1.6
            self.viewer.cam.azimuth = 135.0
            self.viewer.cam.elevation = -24.0
        self.viewer.sync()
        self.next_viewer_deadline = time.perf_counter()
        atexit.register(self._close_viewer)
        print("Live viewer opened; episode is paced to simulation time. Close the window to stop.")

    def _close_viewer(self) -> None:
        if self.viewer is not None:
            handle, self.viewer = self.viewer, None
            try:
                handle.close()
            except Exception:
                pass

    def set_viewer_status(self, title: str, detail: str) -> None:
        """Show a persistent status on the same live model/data pair as the episode."""
        if self.viewer is None or not self.viewer.is_running():
            return
        self.viewer.set_texts((None, None, title, detail))
        self.viewer.sync()

    def wait_for_viewer_close(self) -> None:
        """Keep a completed episode visible until the user closes its viewer."""
        if self.viewer is None:
            return
        try:
            if self.viewer.is_running():
                print("Viewer remains open for inspection; close the MuJoCo window to finish this command.")
                while self.viewer.is_running():
                    self.viewer.sync()
                    time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            print("Viewer closed from the terminal.")
        finally:
            self._close_viewer()

    def _sync_viewer(self) -> None:
        if self.viewer is None or self.total_steps % SAMPLE_EVERY_STEPS != 0:
            return
        if not self.viewer.is_running():
            raise ViewerClosed
        self.viewer.sync()
        self.next_viewer_deadline += float(self.model.opt.timestep) * SAMPLE_EVERY_STEPS
        time.sleep(max(0.0, self.next_viewer_deadline - time.perf_counter()))
