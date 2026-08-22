# AiNex vendor description — provenance

Source: [`Hiwonder/ainex`](https://github.com/Hiwonder/ainex), branch `main`, commit
`e8fe2a816797cf83054135160df5a82ec3596a69`, package
`src/ainex_simulations/ainex_description`.

> **No licence stated.** The repository carries no LICENSE file despite describing itself
> as fully open source, and that covers the URDF, the meshes and `servo_controller.yaml` —
> not just the action groups. This is the only robot here whose vendor files are not under
> an identified licence; see the note in [../../URDF.md](../../URDF.md).

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
python robots/ainex/urdf/flatten_xacro.py \
    <ainex_description dir> robots/ainex/urdf/ainex.urdf
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

**MuJoCo merges *two* links into the worldbody, not one.** `robots/URDF.md` records that a
jointless URDF root gets merged; here it happens twice, because `base_link` is jointless
*and* `body_link` hangs off it by a fixed joint. Compiling the file as-is yields **five
disconnected root bodies** — both `hip_yaw` links, `head_pan_link` and both `sho_pitch`
links — and drops `body_link`'s 0.743 kg out of the tree, leaving 1.6045 kg of the 2.3475.
Adding the three virtual planar joints to `body_link` is what makes it a real body again;
after that there is one root and the mass is right. `robots/ainex/ainex.py` does this, and
`test_attach.py` checks both the single root and the total mass, because a silent
regression here looks like a robot that merely falls apart.

**Every joint carries the same placeholder limits:** `lower="-2.09" upper="2.09"
effort="6" velocity="100"`. ±2.0944 rad is *exactly* the 240° full travel of an HX-series
servo, which is the tell that these are a default rather than a per-joint calibration —
the hips' HX-35HM has 360° and is not distinguished either. The real per-joint limits live
in `ainex_kinematics/config/servo_controller.yaml` as raw servo counts, and
`robots/ainex/servos.py` transcribes them. `test_attach.py` asserts that no joint still
has the placeholder range, since a failure to apply the table is otherwise invisible.
