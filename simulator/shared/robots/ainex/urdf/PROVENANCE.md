# AiNex vendor description — provenance

Source: [`Hiwonder/ainex`](https://github.com/Hiwonder/ainex), branch `main`, commit
`e8fe2a816797cf83054135160df5a82ec3596a69`, package
`src/ainex_simulations/ainex_description`.

> **No licence stated.** The repository carries no LICENSE file despite describing itself
> as fully open source, and that covers the URDF, the meshes and `servo_controller.yaml` —
> not just the action groups. This is the only robot here whose vendor files are not under
> an identified licence; see the note in [molmospaces/robots/URDF.md](../../../../molmospaces/robots/URDF.md).

| File | What it is |
|---|---|
| `ainex.urdf.xacro`, `materials.xacro`, `transmissions.xacro` | vendor source, verbatim |
| `ainex.urdf` | the flattened xacro — **this is what MuJoCo loads** |
| `meshes/*.STL` | 25 binary STLs, 5.4 MB, verbatim |
| `flatten_xacro.py` | regenerates `ainex.urdf` from the xacro |
| `UPSTREAM_README.md` | the vendor's own README, verbatim |

## Why the xacro is flattened offline

`ainex.urdf.xacro` is not loadable by MuJoCo directly, and expanding it at load time would
make a ROS toolchain a hard dependency of viewing a robot — a non-starter on the macOS
stack this simulator runs on (see the top-level `CLAUDE.md` on why MuJoCo needs the
Homebrew framework Python). So it is expanded once and the result committed.

`flatten_xacro.py` is not a general xacro implementation and does not need to be. The
AiNex xacro uses only three constructs: seven scalar `<xacro:property>` definitions, one
arithmetic substitution (`${M_PI/2}`), and two `<xacro:include>`s — `materials.xacro`
(colour definitions, no macros) and `transmissions.xacro` (one macro instantiated 24
times). No conditionals, no loops, no parameterised geometry, so the expansion is
deterministic.

```bash
python shared/robots/ainex/urdf/flatten_xacro.py \
    <ainex_description dir> shared/robots/ainex/urdf/ainex.urdf
```

Re-run it after pulling a newer vendor description, and re-check the figures below.

## Verified after flattening

| | |
|---|---|
| unresolved `${...}` or `<xacro:` | none |
| links | 28 — 25 with meshes, plus `base_link`, `camera_link`, `imu_link` |
| joints | 24 revolute + 3 fixed |
| transmissions | 24 (ROS-control only; MuJoCo ignores them) |
| total mass | **2.3475 kg**, against the vendor's published 2.45 kg for the assembled robot |

## Two things about this description that the simulator corrects

**MuJoCo merges *two* links into the worldbody, not one.**
[`molmospaces/robots/URDF.md`](../../../../molmospaces/robots/URDF.md) records that a
jointless URDF root gets merged; here it happens twice, because `base_link` is jointless
*and* `body_link` hangs off it by a fixed joint. Compiling the file as-is yields **five
disconnected root bodies** — both `hip_yaw` links, `head_pan_link` and both `sho_pitch`
links — and drops `body_link`'s 0.743 kg out of the tree, leaving 1.6045 kg of the 2.3475.
Adding the virtual base joints to `body_link` is what makes it a real body again; after
that there is one root and the mass is right. `shared/ainex_model.py` does this (it moved
there from `molmospaces/robots/ainex/ainex.py` when both engines started compiling the
same robot), and `test_attach.py` checks both the single root and the total mass, because
a silent regression here looks like a robot that merely falls apart.

**The vendor's `init_pose` is not a straight stand, and the difference goes on the
torso.** Its left leg reads hip_pitch −0.828, knee +1.192, ank_pitch +0.625, which on
these axes sums to **−14.95°** rather than to zero — the vendor's `hip_pitch_offset`,
which on the real robot tips the *body* forward over feet that stay flat. This simulator's
torso is bolted to planar joints, so `ainex_model.stance_lean` measures that angle off the
compiled model and gives it to the `base_pitch` joint. With it at zero the 14.95° landed
on the feet instead: soles tilted toe-up, toes 37.6 mm off the surface, the robot standing
on two heel corners while every gap check read a perfect 0.00 mm, because they all measure
the lowest single vertex and that vertex was the heel.

**Every joint carries the same placeholder limits:** `lower="-2.09" upper="2.09"
effort="6" velocity="100"`. ±2.0944 rad is *exactly* the 240° full travel of an HX-series
servo, which is the tell that these are a default rather than a per-joint calibration —
the hips' HX-35HM has 360° and is not distinguished either. The real per-joint limits live
in `ainex_kinematics/config/servo_controller.yaml` as raw servo counts, and
`robots/ainex/servos.py` transcribes them. `test_attach.py` asserts that no joint still
has the placeholder range, since a failure to apply the table is otherwise invisible.

## Verbatim, and checkable

`shared/tests/ainex_provenance_check.py` recomputes these and fails on any difference:

```bash
python shared/tests/ainex_provenance_check.py            # verify
python shared/tests/ainex_provenance_check.py --update   # after a vendor pull
```

They are **git blob hashes** — sha1 over `blob <bytes>\0` plus the content — so a row
here can be checked against `git hash-object <file>` or against GitHub's own blob id for
the upstream file, with no tooling on either side. This robot is the one where it matters
most: its description is the only hardware source in this repo under no stated licence,
so "verbatim" is the entire basis on which it is vendored, and a re-export or a nudged
mesh would otherwise be indistinguishable from the vendor's own file for ever after.

| File | git blob sha1 | |
|---|---|---|
| `ainex.urdf.xacro` | `e0c4b4301ac1d9d574e7ef73033cc41f5eca1863` | vendor |
| `materials.xacro` | `3b656eaee4b1ec5d83f43f2ee315d1967bf94e0c` | vendor |
| `transmissions.xacro` | `5908d61f28319985f771d9f6a42c3bc647edfe5f` | vendor |
| `UPSTREAM_README.md` | `e3b7c153d22fd0fdf4e0979f8e8e1f8f0b372bc5` | vendor |
| `ainex.urdf` | `240884730f10585af0e8c5e2da820c3604ba175e` | generated by `flatten_xacro.py` |
| `meshes/body_link.STL` | `fa6e14a0a993d80aa3ee397193ab9e3f4c14679d` | vendor |
| `meshes/head_pan_link.STL` | `ade52d35166f5eb7306066b4cfac076931e83b13` | vendor |
| `meshes/head_tilt_link.STL` | `4c5acb3bbae1f0f5e7b13a875f7ed784d5ff51ff` | vendor |
| `meshes/l_ank_pitch_link.STL` | `2fbfd3310d42ffefdf313cc050595f12175613b0` | vendor |
| `meshes/l_ank_roll_link.STL` | `50c43ec81113bb90c1ba286395976d8a61953998` | vendor |
| `meshes/l_el_pitch_link.STL` | `82e5823b887b79c65f31dcc2ae20505b42980474` | vendor |
| `meshes/l_el_yaw_link.STL` | `4608a8cddfde37412910c9ee9d034334a3863dd2` | vendor |
| `meshes/l_gripper_link.STL` | `894440e90048d88f9780d16e0aec7418542e64eb` | vendor |
| `meshes/l_hip_pitch_link.STL` | `e4988fb40055469eeea0f0009e8ee4a3254b2309` | vendor |
| `meshes/l_hip_roll_link.STL` | `7771747bbd3d5207fe7beef5d908206978f6cc31` | vendor |
| `meshes/l_hip_yaw_link.STL` | `fe320850720a2e6c71b84a2667f2d0eac18913d5` | vendor |
| `meshes/l_knee_link.STL` | `f4dbf95da6749cee6e61457505b4987eb77cd789` | vendor |
| `meshes/l_sho_pitch_link.STL` | `5fecdff5be6f1ff58913ecefdca9bdba231378f3` | vendor |
| `meshes/l_sho_roll_link.STL` | `e6f2f0f0eca4c3b5ee41754caf1fc784f678791c` | vendor |
| `meshes/r_ank_pitch_link.STL` | `e62ab7aab1fbf5d74a4ffeac8f1503e97c648d12` | vendor |
| `meshes/r_ank_roll_link.STL` | `e7f91a36add45d700e5babccc46227c65f817428` | vendor |
| `meshes/r_el_pitch_link.STL` | `1c76d4e6462e8dc3edcba8a7da522ada900e58e9` | vendor |
| `meshes/r_el_yaw_link.STL` | `b2cefc5bb3ebf9dc327ea72f6bf898e0ec1891fe` | vendor |
| `meshes/r_gripper_link.STL` | `fff4afdde9bfec16fe60e81199ef675d1b4bce23` | vendor |
| `meshes/r_hip_pitch_link.STL` | `3bcd2a55100c37dd2567aab31fa82f39ed3b9479` | vendor |
| `meshes/r_hip_roll_link.STL` | `83e5c673d677975ec2e2743c8c6d8e1ff02ccc38` | vendor |
| `meshes/r_hip_yaw_link.STL` | `e8a92622863bb2ae8c41cbcc85ae2a198f3e1fb5` | vendor |
| `meshes/r_knee_link.STL` | `f85f968ad058bce4d1d937217a82a84155da243a` | vendor |
| `meshes/r_sho_pitch_link.STL` | `4bc5dbf0f3177c9c2b80a25eadf5bdc0af57ba96` | vendor |
| `meshes/r_sho_roll_link.STL` | `3ead3cea10eeb33494a3cd300a63d92beb04fa37` | vendor |
