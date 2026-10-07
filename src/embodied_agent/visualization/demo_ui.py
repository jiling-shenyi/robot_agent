"""Single-window MuJoCo demo with serialized scene operations and live controls."""

from __future__ import annotations
from contextvars import copy_context

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

import mujoco

from .world_view import WorldView


class ViewerClosed(RuntimeError):
    """The visible demo was closed while an action was running."""


class _PumpingPlanner:
    def __init__(self, planner: Any, wait: Callable):
        self.delegate, self.wait = planner, wait
        self.kind = getattr(planner, "kind", getattr(planner, "mode", "robot"))

    def plan(self, *args, **kwargs):
        return self.wait(lambda: self.delegate.plan(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class DemoApp:
    """Physics/rendering stay on the Tk thread; only language requests use a worker."""

    def __init__(self, session: Any, cases: list[dict], *, initial_map="home_living_room", batch=False):
        self.session, self.cases, self.batch = session, cases, batch
        self._prime_graphics_before_tk()
        try:
            self.window = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(f"无法创建可视化窗口；自由模式需要 Tk 图形桌面：{exc}") from exc
        self.window.title("Robot Agent · MuJoCo 通用测试台")
        self.window.geometry("1180x940")
        self.window.minsize(960, 800)
        self.window.protocol("WM_DELETE_WINDOW", self.request_close)
        self.busy = False
        self.closing = False
        self.closed = False
        self.batch_finished = False
        self.batch_pass = False
        self.last_frame_step = 0
        self.last_episode = None
        self.frame_deadline = time.perf_counter()
        self.control_widgets: list[ttk.Widget] = []
        self.map_names: dict[str, str] = {}
        self.case_names: dict[str, dict] = {}
        self.agent_names = {"机器人 Agent": "robot"}
        self.status = tk.StringVar(value="请选择地图并点击「加载地图」，随后显示模拟世界。")
        self.world_info = tk.StringVar(value="坐标单位：米。地图编辑保存为初始环境，重置恢复该环境。")
        try:
            self._build(initial_map)
        except Exception:
            self.window.destroy()
            raise
        self.session.on_frame = self.on_frame
        self.session.on_status = self.on_status
        instruction_agent = getattr(self.session, "instruction_agent", None)
        if instruction_agent is not None:
            self.session.instruction_agent = _PumpingPlanner(instruction_agent, self._wait_language)
        environment = self.session.environment_agent
        original_plan = environment.plan
        environment.plan = lambda *a, **kw: self._wait_language(lambda: original_plan(*a, **kw))

    @staticmethod
    def _prime_graphics_before_tk() -> None:
        """Initialize GLFW before Tk so Windows DPI awareness is stable."""
        if os.name != "nt":
            return
        context = mujoco.GLContext(1, 1)
        try:
            context.make_current()
        finally:
            # Keep GLFW initialized for the renderer created after map selection,
            # but release this temporary native window and its GL context.
            context.free()

    def _button(self, parent, text, command, **pack):
        button = ttk.Button(parent, text=text, command=command)
        button.pack(**pack)
        self.control_widgets.append(button)
        return button

    def _build(self, initial_map):
        style = ttk.Style(self.window)
        style.configure("TButton", padding=(8, 5))
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))
        maps = self.session.store.list_maps()
        labels = []
        initial_label = ""
        for world in maps:
            label = f"{world.name} [{world.map_id}]"
            self.map_names[label] = world.map_id
            labels.append(label)
            if world.map_id == initial_map:
                initial_label = label
        if not initial_label:
            raise ValueError(f"地图不存在: {initial_map}")
        ttk.Label(toolbar, text="地图").pack(side="left", padx=(0, 6))
        self.map_choice = ttk.Combobox(toolbar, values=labels, state="readonly", width=25)
        self.map_choice.set(initial_label)
        self.map_choice.pack(side="left")
        self.control_widgets.append(self.map_choice)
        self._button(toolbar, "加载地图", lambda: self._execute(self._select_map), side="left", padx=6)
        self._button(toolbar, "重置环境", lambda: self._execute(self._reset), side="left")
        # The case selector is deliberately anchored at the top right.
        case_box = ttk.Frame(toolbar)
        case_box.pack(side="right")
        self._button(case_box, "执行用例", lambda: self._execute(self._run_selected_case), side="right", padx=(6, 0))
        self.case_choice = ttk.Combobox(case_box, state="readonly", width=36)
        self._update_case_choices(initial_map)
        self.case_choice.pack(side="right")
        self.control_widgets.append(self.case_choice)
        ttk.Label(case_box, text="测试用例  ").pack(side="right")

        if self.batch:
            batch_bar = ttk.Frame(outer)
            batch_bar.pack(fill="x", pady=(0, 6))
            ttk.Label(batch_bar, text="批量可视模式：编辑使用输出目录内地图副本；每例从初始状态执行。").pack(side="left")
            self._button(batch_bar, "开始批量测试", lambda: self._execute(self._run_batch), side="right")

        self.view = WorldView(outer, width=960, height=400)
        self.view.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.world_info, wraplength=1120).pack(anchor="w", pady=(6, 2))
        ttk.Label(outer, textvariable=self.status, wraplength=1120, foreground="#185980").pack(anchor="w", pady=(0, 6))
        inputs = ttk.Frame(outer)
        inputs.pack(fill="x")
        inputs.columnconfigure(0, weight=1)
        inputs.columnconfigure(1, weight=1)
        robot_box = ttk.LabelFrame(inputs, text=f"机器人 Agent / {self.session.instruction_agent.kind}", padding=8)
        robot_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        agent_row = ttk.Frame(robot_box)
        agent_row.pack(fill="x", pady=(0, 5))
        ttk.Label(agent_row, text="Agent").pack(side="left")
        self.agent_choice = ttk.Combobox(agent_row, values=list(self.agent_names), state="readonly", width=15)
        self.agent_choice.set("机器人 Agent")
        self.agent_choice.pack(side="left", padx=8)
        self.control_widgets.append(self.agent_choice)
        self.robot_hint = ttk.Label(agent_row, text="例如：把茶几上的遥控器送到餐桌。")
        self.robot_hint.pack(side="left")
        self.robot_input = tk.Text(robot_box, height=2, wrap="word", font=("Microsoft YaHei UI", 10))
        self.robot_input.insert("1.0", "把茶几上的遥控器送到餐桌。" if initial_map == "home_living_room" else "把方块放到 A 区。")
        self.robot_input.pack(fill="x")
        self._button(robot_box, "执行机器人指令", lambda: self._execute(self._run_robot), side="right", pady=(6, 0))
        env_box = ttk.LabelFrame(inputs, text=f"模拟世界环境修改 Agent / {self.session.environment_agent.mode}", padding=8)
        env_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.env_hint = ttk.Label(env_box, text="家居编辑只改变对象初始位置、名称或描述；危险状态与权限不能由描述修改。", wraplength=520)
        self.env_hint.pack(anchor="w", pady=(0, 5))
        self.env_input = tk.Text(env_box, height=2, wrap="word", font=("Microsoft YaHei UI", 10))
        self.env_input.insert("1.0", "将remote初始位置的x轴提高0.01米" if initial_map == "home_living_room" else "将目标方块的初始位置调整到 (0.42, -0.26, 0.425)")
        self.env_input.pack(fill="x")
        self._button(env_box, "修改并保存初始环境", lambda: self._execute(self._edit_environment), side="right", pady=(6, 0))
        ttk.Label(outer, text="运行记录（执行期间地图、重置和提交按钮暂停使用）").pack(anchor="w", pady=(8, 2))
        self.log = tk.Text(outer, height=5, state="disabled", wrap="word", font=("Microsoft YaHei UI", 9))
        self.log.pack(fill="x")

    def register_agent(self, name: str, handler: Callable):
        """Register future agents without changing the UI or the batch runner."""
        self.session.register_agent(name, handler)
        self.agent_names[name] = name
        self.agent_choice.configure(values=[*self.agent_choice.cget("values"), name])

    def _update_case_choices(self, map_id: str) -> None:
        current = self.case_choice.get()
        available = [case for case in self.cases if case.get("map_id", map_id) == map_id]
        self.case_names = {f"{case['case_id']} · {case.get('instruction', case.get('name', ''))[:24]}": case
                           for case in available}
        self.case_choice.configure(values=list(self.case_names))
        if current in self.case_names:
            self.case_choice.set(current)
        elif self.case_names:
            self.case_choice.current(0)
        else:
            self.case_choice.set("")

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _require_world(self):
        if self.session.world is None:
            raise ValueError("请先点击「加载地图」。")

    def _select_map(self):
        world_id = self.map_names[self.map_choice.get()]
        # MuJoCo's first GL renderer can cause Windows/Tk to recalculate the
        # top-level window bounds. Preserve the user's actual window state at
        # click time, then restore it after the scene and its first frame load.
        previous_state = self.window.state()
        previous_geometry = self.window.geometry()
        previous_scaling = self.window.tk.call("tk", "scaling")
        try:
            self.session.select_map(world_id)
            self._refresh_world()
            self.status.set(f"已加载 {world_id}。可输入机器人指令，或选择测试用例。")
            self._append(f"已加载地图 {world_id}。")
        finally:
            self._restore_window_bounds(previous_state, previous_geometry, previous_scaling)

    def _restore_window_bounds(self, state: str, geometry: str, scaling: str) -> None:
        self.window.tk.call("tk", "scaling", scaling)
        self.window.update_idletasks()
        if state == "zoomed":
            self.window.state("zoomed")
        elif state == "normal":
            self.window.state("normal")
            self.window.geometry(geometry)
        else:
            self.window.state(state)
        self.window.update_idletasks()

    def _reset(self):
        self._require_world()
        self.session.reset()
        self._refresh_world()
        self._append("已恢复所选地图最新保存的初始状态。")

    def _refresh_world(self):
        world = self.session.world
        if world is None:
            return
        for label, map_id in self.map_names.items():
            if map_id == world.map_id:
                self.map_choice.set(label)
        self._update_case_choices(world.map_id)
        self._update_world_info(self.session.episode)
        is_home = getattr(self.session.episode, "robot_kind", None) == "stretch"
        self.robot_hint.configure(text="例如：把茶几上的遥控器送到餐桌。" if is_home else "例如：把方块放到 A 区。")
        self.env_hint.configure(text=("家居编辑只改变对象初始位置、名称或描述；危险状态与权限不能由描述修改。" if is_home else "例如：将危险区扩大到原来的1.2倍；坐标使用米。"))
        defaults = {"把茶几上的遥控器送到餐桌。", "把方块放到 A 区。"}
        if self.robot_input.get("1.0", "end").strip() in defaults:
            self.robot_input.delete("1.0", "end")
            self.robot_input.insert("1.0", "把茶几上的遥控器送到餐桌。" if is_home else "把方块放到 A 区。")
        env_defaults = {"将remote初始位置的x轴提高0.01米", "将目标方块的初始位置调整到 (0.42, -0.26, 0.425)"}
        if self.env_input.get("1.0", "end").strip() in env_defaults:
            self.env_input.delete("1.0", "end")
            self.env_input.insert("1.0", "将remote初始位置的x轴提高0.01米" if is_home else "将目标方块的初始位置调整到 (0.42, -0.26, 0.425)")
        if not self.closing:
            self.on_frame(self.session.episode, pace=False)

    def _update_world_info(self, episode):
        if getattr(episode, "robot_kind", None) == "stretch":
            snapshot = episode.snapshot()
            robot = snapshot["robot"]
            point = ", ".join(f"{value:.3f}" for value in robot["position_m"])
            self.world_info.set(
                f"地图：{episode.world.name} | 修订 {snapshot['map_revision']} / 世界版本 {snapshot['world_version']} | "
                f"Stretch 底盘 xyz=({point}) m | 模式 {robot['mode']} / {robot.get('stage', '')} | "
                f"持物 {robot['held_object'] or '无'} | 风险 {len(snapshot['risks'])} | "
                f"对象 {len(snapshot['objects'])} / 家具 {len(snapshot['furniture'])}")
        else:
            world = self.session.world
            if world is not None:
                self.world_info.set(f"地图：{world.name} | 修订 {world.revision} | 初始方块 xyz={list(world.cube_position_m)} m | 世界坐标：红 X / 绿 Y / 蓝 Z")

    def _report_result(self, result):
        status = result.get("status", "UNKNOWN")
        code = result.get("error_code") or ""
        expected = ""
        if "passed" in result:
            expected = f" | 用例判定：{'通过' if result['passed'] else '未通过'}"
        detail = result.get("error_message") or result.get("message") or ""
        self.status.set(f"{status} {code}{expected} {detail}")
        self._append(self.status.get())
        if result.get("task_record_path"):
            self._append(f"任务记录：{result['task_record_path']}")
        if result.get("recording_error") or result.get("recording_errors"):
            self._append("任务记录写入失败；实际执行结果已保留，记录可能尚未完整结束。")
        self._refresh_world()

    def _run_robot(self):
        self._require_world()
        instruction = self.robot_input.get("1.0", "end").strip()
        self._append(f"[{self.agent_choice.get()}] {instruction}")
        self._report_result(self.session.run_agent(instruction, agent=self.agent_names.get(self.agent_choice.get(), self.agent_choice.get())))

    def _edit_environment(self):
        self._require_world()
        instruction = self.env_input.get("1.0", "end").strip()
        self._append(f"[环境编辑] {instruction}")
        self._report_result(self.session.edit_environment(instruction))

    def _run_selected_case(self):
        self._require_world()
        case = self.case_names.get(self.case_choice.get())
        if case is None:
            raise ValueError("请选择测试用例。")
        self._append(f"[用例 {case['case_id']}] 重置至用例初始状态后执行。")
        self._report_result(self.session.run_case(case))

    def _run_batch(self):
        self._require_world()
        completed = 0
        try:
            for index, case in enumerate(self.cases, 1):
                if self.closing:
                    break
                self._append(f"[{index}/{len(self.cases)}] {case['case_id']}")
                result = self.session.run_case(case)
                completed = index
                self._report_result(result)
                if result["status"] == "ABORTED":
                    break
                # Keep the measured final state visible between cases.
                deadline = time.perf_counter() + 0.8
                while time.perf_counter() < deadline and not self.closing:
                    self.window.update()
                    time.sleep(0.02)
        finally:
            self.session.mark_not_run(self.cases[completed:])
        summary = self.session.finish()
        self.batch_finished = completed == len(self.cases) and not self.closing
        self.batch_pass = bool(summary["all_pass"]) and self.batch_finished
        self._append(f"批量执行结束：{'通过' if self.batch_pass else '未全部通过'}。证据：{self.session.output_dir}")

    def _execute(self, action):
        if self.busy or self.closing:
            return
        self.busy = True
        for widget in self.control_widgets:
            widget.configure(state="disabled")
        try:
            action()
        except Exception as exc:
            message = f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
            self.status.set(message)
            self._append(message)
        finally:
            self.busy = False
            try:
                self.session.finish()
            finally:
                if self.closing:
                    self._close()
                else:
                    for widget in self.control_widgets:
                        widget.configure(state="readonly" if isinstance(widget, ttk.Combobox) else "normal")

    def _wait_language(self, call):
        """Keep Tk responsive without touching physics or map persistence off-thread."""
        result_queue = queue.Queue(maxsize=1)

        def work():
            try:
                result_queue.put((True, call()))
            except BaseException as exc:
                result_queue.put((False, exc))

        context = copy_context()
        threading.Thread(target=lambda: context.run(work), name="demo-language", daemon=True).start()
        next_planning_frame = time.perf_counter()
        while result_queue.empty():
            self.window.update()
            if self.closing:
                self.session.cancel_task("Viewer closed during model planning", code="VIEWER_CLOSED")
                raise ViewerClosed("用户关闭了测试窗口；已取消后续执行。")
            if hasattr(self.session, "pump_planning_hold"):
                self.session.pump_planning_hold(seconds=.02)
            if (getattr(self.session, "_active_instruction_adapter", None) is not None
                    and self.session.episode is not None and time.perf_counter() >= next_planning_frame):
                self.on_frame(self.session.episode, pace=False)
                next_planning_frame = time.perf_counter() + .1
            task_budget = getattr(self.session, "active_task_budget", None)
            if task_budget is not None:
                task_budget.check()
            time.sleep(0.02)
        success, value = result_queue.get()
        if self.closing:
            raise ViewerClosed("用户关闭了测试窗口；已取消后续执行。")
        if not success:
            raise value
        self.frame_deadline = time.perf_counter()
        return value

    def on_frame(self, episode, *, pace=True):
        if self.closing:
            raise ViewerClosed("用户关闭了测试窗口。")
        now = time.perf_counter()
        steps = int(episode.total_steps)
        if episode is not self.last_episode or steps < self.last_frame_step:
            self.frame_deadline = now
        elif pace and self.busy:
            self.frame_deadline += (steps - self.last_frame_step) * float(episode.model.opt.timestep)
        self.last_episode, self.last_frame_step = episode, steps
        self._update_world_info(episode)
        self.view.render(episode)
        if self.view.last_error:
            raise RuntimeError(f"可视化失败，无法继续自由测试：{self.view.last_error}")
        self.window.update()
        if pace and self.busy:
            # Wait in short increments so window close and camera input stay responsive.
            while time.perf_counter() < self.frame_deadline and not self.closing:
                self.window.update()
                time.sleep(min(0.01, max(0.0, self.frame_deadline - time.perf_counter())))
        if self.closing:
            raise ViewerClosed("用户关闭了测试窗口。")

    def on_status(self, title, detail):
        self.status.set(f"{title} · {detail}")
        self._append(self.status.get())
        self.window.update_idletasks()

    def request_close(self):
        self.closing = True
        if hasattr(self.session, "cancel_task"):
            self.session.cancel_task("User closed the viewer", code="VIEWER_CLOSED")
        if self.busy:
            self.status.set("正在结束本次请求并保存运行证据…")
        else:
            self._close()

    def _close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.session.finish()
        finally:
            try:
                self.view.close()
                self.session.close()
            finally:
                self.window.destroy()

    def run(self):
        if self.session.episode is None:
            # Commit the requested initial size before map loading preserves it.
            # Otherwise the first idle callback captures Tk's temporary minimum.
            self.window.update_idletasks()
            self.window.after_idle(lambda: self._execute(self._select_map))
        try:
            self.window.mainloop()
        finally:
            if not self.closed:
                self._close()
        if self.batch:
            return 0 if self.batch_finished and self.batch_pass else 1
        return 0
