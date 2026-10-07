# Stretch 2 project adapter

The original `stretch.xml`, assets, `scene.xml` and Clear BSD `LICENSE` are
vendored unchanged from `mujoco-menagerie==2026.9.2`. The locked source tree is
`d37c1fc83fc259f0681856608f3c59df66ce7c1d` in Menagerie commit
`c96a32d28fb5da84da38c1da4d749e7a13212855`. Package/archive/file hashes and the
official source link are recorded in `MODEL_PROVENANCE.json`. These added
project notes are separate from the upstream model.

The adapter merges that model with the independently registered HomeWorld
room/furniture/free objects. It sets a 0.002 s timestep, gives unnamed robot
geoms explicit names, and compiles from the vendored asset directory to avoid
MuJoCo's Windows path decoding problem with Unicode workspace names. It adds
no welds, mocap drivers, object attachments or robot-furniture contact
exclusions. Only scene initialization assigns base/joint qpos. Task execution
uses the eight original actuators.

| Actuator | Control meaning | Control range |
| --- | --- | --- |
| `forward` | Tendon motor input; gear 3, wheel tendon coefficients +0.5/+0.5 | [-1, 1] |
| `turn` | Tendon motor input; gear 3, left/right coefficients -0.5/+0.5 | [-1, 1] |
| `lift` | Lift displacement in metres | [-0.5, 0.6] |
| `arm_extend` | Sum of four telescope slides in metres | [0, 0.52] |
| `wrist_yaw` | General actuator position input in radians | [-1.75, 4] |
| `grip` | Finger slider position in metres; positive opens | [-0.005, 0.04] |
| `head_pan` | Head pan position in radians | [-3.9, 1.5] |
| `head_tilt` | Head tilt position in radians | [-1.53, 0.79] |

Positive `forward` and `turn` were physically measured to produce negative
base x velocity and negative world yaw velocity. The adapter negates target
wheel rates. Nominal wheel radius is 0.05 m, centre separation is 0.3407 m.
Wheel-rate feedback, measured base yaw feedback and base-origin feedback
during turns account for the asymmetric mast/payload load. No kinematic base
teleportation is used for navigation or docking.

World +z is up. Base +x is forward and default arm extension is base -y.
At zero wrist yaw, the nominal closed fingertip centre in the base frame is
approximately (-0.021385, -0.34036 - extension, 0.532 + lift). Every object
reachability check uses the measured base rotation matrix to transform world
coordinates into this frame. Table-front operation points use yaw pi so the
arm extends toward world +y. The initial table tops are 0.6 m. The demonstrated
remote/book/cup are independent 0.05 m box proxies of mass 0.08 kg; the cup is
not a liquid, mesh perception or general cup grasp simulation.

The validated grasp envelope is box full width 0.038–0.060 m, full height
0.038–0.070 m, mass <= 0.15 kg, lateral alignment error <= 0.018 m and telescope
extension 0.06–0.49 m. Side grasp approaches with an open 0.035 m slider and a
raised lift. It aims 0.020 m above the object's centre and 0.008 m short of the
nominal tip plane to clear tabletop and proximal metal finger links. Closing
uses -0.005 m slider input. The 0.15 m lift must achieve >= 0.10 m measured
object rise, bilateral rubber contact, and no independent support. Then the
telescope retracts to 0.02 m. Grasped objects stay free bodies throughout.

The official wrist general actuator's gain is 1 Nm/rad. The adapter commands
that same actuator with external position/velocity feedback (additional
position gain 18, velocity gain 0.9) to maintain yaw 0 during grasp, carry and
release. Commands remain within the official control range and original
actuator force limit of 5 Nm. This eliminated a measured 35.7 mm release offset
after carrying; the complete remote transfer then placed within 4.94 mm.

`footprint_evidence()` projects all registered robot collision mesh vertices
or conservative primitive corners, together with the actual rotated and
offset payload, into the horizontal plane about the measured base origin.
The enclosing radius adds 0.015 m clearance. Initial open-hand radius is
approximately 0.415 m including clearance; the nominal base radius alone is
insufficient. Planning uses the full measured collision envelope.

Every physics step checks finite state, MuJoCo warnings, base tilt <= 0.12 rad,
and actual contact force. Contact with any furniture or room wall by the robot
is blocked above 0.02 N, including contacts by gripper fingers. Robot contact
with objects is allowed only on the registered left/right finger subtrees
and only for the current target or held object. Grasp verification requires
the two rubber tip sides separately, each with measured normal force > 0.05 N.
Held-object contact with another object is always blocked above 0.02 N.
Held-object contact with furniture is allowed only with the measured original
support during `pick_lift`, or the declared target support during
`place_lower`/`place_release`; all other furniture/wall contacts are blocked.
Errors retain actual geom IDs/names, contact distance, normal force and the
0.02 N normal-force threshold.

Held-object wrist-relative drift > 0.035 m or loss of bilateral rubber contact
for more than 100 steps (0.2 simulated seconds) stops ordinary execution.
Navigation requires telescope input <= 0.035 m; carry additionally requires
an unsupported verified grasp. Docking verifies final xy error <= 0.02 m,
and manipulation requires base linear speed <= 0.01 m/s and yaw speed
<= 0.03 rad/s. Placement currently supports the validated horizontal tables,
requires 0.025 m footprint edge margin, lowers the measured grasp to 0.003 m
above independent support, opens the slider to 0.020 m, withdraws 0.08 m
along the approach axis, and verifies complete robot separation plus target
support contact before clearing the held-object marker. Only then does it
complete the opening arc to 0.035 m, raise the hand, retract and settle. This
avoids the metal finger-link sweep that displaced the cup during a full-open
release in place. The `release_verified` event records actual support and
hand/object contact pairs before clearing the held-object marker. The release
transition retains the held-object marker and target support authorization
until measured separation; reaching an open actuator setting alone does not
declare release. Existing contact guards continue throughout this trajectory,
including robot/furniture and held-object/other-object checks. Success requires
no robot contact, target support contact, object speed <= 0.01 m/s and xy
error <= 0.025 m; no object state is directly set to satisfy these conditions.

Normal stop continues all guards and holds arm/gripper targets while braking
through wheel motors. Latched emergency stop continues actual physics and
wheel braking, then independently verifies linear speed <= 0.01 m/s and yaw
speed <= 0.03 rad/s. With a held object, both modes additionally require
bilateral rubber contact, no furniture support and <= 0.035 m wrist-relative
drift; failure is explicitly `STOP_HOLD_FAILED`. The skill does not count an
empty-handed stop as successful payload retention.
