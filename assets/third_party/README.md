# Third-party model sources

## Franka Emika Panda

- **Source:** [MuJoCo Menagerie, `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
- **Python package:** `mujoco-menagerie==2026.9.2`
- **Menagerie repository commit:** `c96a32d28fb5da84da38c1da4d749e7a13212855`
- **Model entry OID:** `3d2262eeb81ecec19abfa31dd509e35abbb33e67`
- **Upstream license:** Apache-2.0; retain the upstream `LICENSE` with the model files.
- **Upstream entry XML:** `panda.xml`; source model directory: `franka_emika_panda/`.
- **Project copy:** `assets/third_party/franka_emika_panda/` (to be populated and verified in M1).
- **Project-specific changes:** keep upstream `panda.xml` unchanged. Put M1's actuator grouping, end-effector site, and mocap-weld additions in a separate derivative XML, and document those differences here.

The values above were read from the installed Menagerie package metadata on 2026-09-27. The model has not yet been copied into the repository at the M0 baseline; M1 must verify the copied files against the package-cached entry and record the derivative XML changes.
