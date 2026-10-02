# Robot Specifications — High-Level Specification

## 1. Purpose and layout

This document describes the robots and the files needed to import them into
MuJoCo, reproducing the real hardware's geometry, physical properties, motion,
sensors and control interfaces as documented by their manufacturers. Together with the
robot files it lists (§2) it is self-contained. Each robot has its own file,
`robots_specs/<id>.md`, named exactly by its robot id (amended 2026-10-02: the robot
sections moved out of this document into those files).

Each robot file defines a stable **robot id**, its identity and kind, its **embodiment**
(the exact hardware variant, its controller, and its camera and lidar hardware), its
authoritative sources and boot, and its required files. Each robot's folder is always
`robots_specs/<id>/`, with the folder name exactly matching its robot id. Paths are relative to the
repository root. These are required asset paths, not a statement that the files
have already been downloaded: meshes are not committed, and the simulator's
`run.sh setup` fetches them from the pinned sources, verified against
`robots_specs/meshes.sha256`.

Each robot folder must contain:

* The official URDF and every mesh or other asset it references, preserving upstream
  filenames and relative paths.
* Exactly one of `ros.yml` (ROS 1) or `ros2.yml` (ROS 2), for the ROS dialect of the
  recorded boot; the filename identifies that dialect.
* An official MJCF and its assets where one exists for the same hardware embodiment.
  A model of a different hardware variant must not be presented as that robot's official model.

The ROS interface descriptions record every authoritative node, topic, service,
action and parameter, including message types, frames, rates, camera streams,
optional features, each command's units and limits, and the stop command. The stop
command is recorded for every velocity-driven motion (driving, walking); every other
motion ends on reaching its goal or end state. A topic published only on events or on demand
is recorded as non-periodic and has no rate; every other topic records its rate. Record whether each command has a watchdog or timeout and its interval,
established by measurement on the real robot running the authoritative boot, with the
method and date of the measurement. Record the ROS distribution, hardware boot
configuration and pinned source revisions used to establish the interface. Do not
invent a manufacturer ROS interface when none exists; identify an approved community
interface explicitly. Describe the robot's real-hardware interface, separately
from transport or discovery services added by a simulation application.

Each robot file names its **authoritative boot**: the launch and parameter files, at
the pinned revision, that the vendor documentation starts for normal operation, and the
vendor data files they load (such as action groups and gait parameters). The interface
description records that boot. Endpoints that exist only when an optional plugin is
installed or through a launch other than the authoritative boot, such as mapping and
navigation, are recorded as `optional`. Each camera's raw image stream is marked as a
camera.

### MuJoCo import and real-hardware fidelity

Use the official MJCF for the exact hardware variant when available. Otherwise,
import the official URDF and retain a reproducible MJCF conversion in the robot's
folder as `model.xml`, with all referenced assets. Label converted models as
derived, not manufacturer-provided. Record conversion steps, source revisions
and any adaptations in `import.md` in the same folder.

Where the authoritative boot's `robot_description` and the model disagree on joint
names, axes, limits or zero offsets, the boot's `robot_description` governs; `import.md`
lists each difference as an adaptation. An official MJCF is never modified: when it
lacks an element the interface requires, such as a camera the boot publishes, a derived
`model.xml` adds it, labelled as derived, with the element's pose source.

The imported model must preserve:

* Link dimensions, mesh scales, joint origins and axes, joint types, ranges,
  velocity and effort limits, and the manufacturer's coordinate frames.
* Masses, centres of mass and inertia tensors; collision geometry and contact
  properties appropriate to the real wheels, feet, grippers and body.
* Actuator mappings, gearing, command units, control modes and limits needed to
  reproduce real arm, gripper, wheel or walking behaviour.
* Sensor poses and documented characteristics, including camera resolution and
  field of view, lidar geometry, and feedback frames and rates.

Use pinned manufacturer models, real-hardware software and documentation as the
sources for these properties. Document missing values, conflicting sources and
necessary estimates in `import.md`; distinguish estimates from manufacturer
specifications. Conversion must not silently change the robot's embodiment or
exceed its documented capabilities.

## 2. Robots

Every robot is one file in this folder, `robots_specs/<id>.md`, listed below; the robot
files (every `.md` here with a robot id, this document and `SCHEMA.md` aside) are the robot
registry, and this table must list exactly them. Robots are single bodies: no robot is assembled from other robots
(amended 2026-10-02: the `myagv_mycobot280` assembly of myAGV and myCobot 280 is no longer
supported).

| Robot id | Name | Kind | File |
| --- | --- | --- | --- |
| `myagv` | myAGV | `mobile_base` | [`myagv.md`](myagv.md) |
| `so101` | SO-101 | `arm` | [`so101.md`](so101.md) |
| `ainex` | AiNex | `humanoid` | [`ainex.md`](ainex.md) |
| `mycobot280` | myCobot 280 | `arm` | [`mycobot280.md`](mycobot280.md) |
| `rosmaster_x3_plus` | ROSMASTER X3 PLUS | `mobile_manipulator` | [`rosmaster_x3_plus.md`](rosmaster_x3_plus.md) |
