"""The apple-on-plate benchmark definition.

One scene, one fixed layout: the red apple starts on the table at
``(0.30, 0.10, 0.020)`` and must end up on the plate centred at
``(0.226, -0.226)``. Both are **world**-frame coordinates and both are
``CONTRACT.md`` section 4 numbers as amended, confirmed against the compiled
MJCF in ``simulator/so_arm_mujoco/mjcf/task_scene.xml``:

```
body apple  world pos [0.3    0.1    0.02]
body plate  world pos [0.226 -0.226  0.0 ]
apple radius    0.02
plate cylinder  size [0.1 0.0102] at z 0.0102  -> plate top z 0.0204
```

The plate moved from ``(0.40, -0.20)`` by 0.176 m: it is now at bearing -45.0
deg and 0.3196 m from the base, where it was at -26.6 deg and 0.4472 m. Nothing
else in the scene moved and no success clause changed.

The apple is a **20 mm-radius** sphere. An earlier revision of this module
carried 0.03, describing a 60 mm ball that the SO-101's 43 mm jaws could not
close on; that apple no longer exists in the scene. Section 4 of the contract
calls a silent change to this number a breach, so it is pinned here and the
scene target carries it to the policy and the scorers.
"""

from __future__ import annotations

from typing import Any

from inspect_robots import Scene, Target, Task

from robot_console.arm.success import PlateGoal

#: World-frame apple start pose from ``task_scene.xml`` / ``CONTRACT.md`` section 4.
APPLE_XYZ: tuple[float, float, float] = (0.30, 0.10, 0.020)
#: World-frame plate centre and the top-face height the *gate* is built on.
#: The z is 0.020, not the MJCF collision top of 0.0204, because 0.020 is what
#: ``so_arm_mujoco/task_manager.py`` uses to derive ``RESTING_Z = 0.040``. The
#: 0.4 mm difference sits well inside clause 2's +/- 0.015 band; matching the
#: arbiter matters more than matching the mesh.
PLATE_XYZ: tuple[float, float, float] = (0.226, -0.226, 0.020)

INSTRUCTION = (
    "Move the arm towards the red apple, grasp it, lift it up, and place it on "
    "the white plate."
)

#: Apple sphere radius in ``task_scene.xml``, and the contract's declared value.
APPLE_RADIUS = 0.020


def resolve_instruction(text: str | None = None, path: str | None = None) -> str:
    """Pick the instruction from a literal, a file, or the benchmark's default.

    The file form exists because a long prompt does not survive shell quoting
    intact, and a prompt silently mangled by the shell is indistinguishable from
    a policy that misunderstood it.
    """
    if text is not None and path is not None:
        raise ValueError("give either an instruction or a file holding one, not both")
    if path is not None:
        from pathlib import Path as _Path

        return _Path(path).read_text(encoding="utf-8").strip()
    return INSTRUCTION if text is None else text


def instruction_warning(instruction: str, *, policy_ignores_it: bool = False) -> str | None:
    """Return the caveat owed to a run given a non-default instruction, or None.

    Changing the instruction changes what the policy is *told*. It does not
    change what is *measured*: the three scorers compute apple-on-plate geometry
    whatever the text says. So a custom-goal run scores 0 against this task, and
    that 0 is a statement about apple-on-plate geometry rather than about
    whether the policy did the thing it was asked to do. Reporting it as a task
    failure would be a false claim, which is why this is loud rather than a
    footnote in the JSON.
    """
    if instruction == INSTRUCTION:
        return None
    lines = [
        "=" * 78,
        "CUSTOM INSTRUCTION — READ WHAT THIS DOES NOT CHANGE",
        "",
        f"  asked for: {instruction}",
        f"  benchmark: {INSTRUCTION}",
        "",
        "The scorers are unchanged: apple_on_plate_success, sim_task_success and",
        "apple_plate_distance all measure apple-on-plate geometry. A 0 from this run",
        "means the apple did not end up on the plate — it is NOT a measurement of",
        "whether the policy achieved the goal typed above. This probes instruction",
        "following, not task success.",
    ]
    if policy_ignores_it:
        lines += [
            "",
            "Worse here: SO101WaypointPolicy never reads the instruction. It executes a",
            "fixed apple->plate plan, so this run is identical to one with the default",
            "text. The text is recorded as provenance and has no other effect.",
        ]
    lines.append("=" * 78)
    return "\n".join(lines)


def apple_on_plate_scene(
    scene_id: str = "apple-on-plate",
    *,
    apple_radius: float = APPLE_RADIUS,
    apple_xy: tuple[float, float] = (APPLE_XYZ[0], APPLE_XYZ[1]),
    plate_xyz: tuple[float, float, float] = PLATE_XYZ,
    goal: PlateGoal | None = None,
    instruction: str = INSTRUCTION,
) -> Scene:
    """Build the single fixed scene, carrying its geometry in the target spec.

    ``apple_radius`` sets both where the apple's centre rests on the table and
    the success gate derived from it, so a diagnostic run against a modified
    scene stays self-consistent. At the contract's own radius this reproduces
    the section 5 gate exactly.
    """
    apple_xyz = (apple_xy[0], apple_xy[1], apple_radius)
    resolved = goal or PlateGoal.for_apple(
        apple_radius=apple_radius,
        plate_top_z=plate_xyz[2],
        center_xy=(plate_xyz[0], plate_xyz[1]),
        spawn_xy=apple_xy,
    )
    return Scene(
        id=scene_id,
        instruction=instruction,
        target=Target(
            kind="object_on_receptacle",
            spec={
                "object": "apple",
                "receptacle": "plate",
                "apple_xyz": list(apple_xyz),
                "plate_xyz": list(plate_xyz),
                "plate_center_xy": list(resolved.center_xy),
                "max_horizontal_distance": resolved.radius,
                "placed_z_range": [
                    resolved.center_z - resolved.z_tolerance,
                    resolved.center_z + resolved.z_tolerance,
                ],
                "max_speed_mps": resolved.max_speed,
                "min_displacement_m": resolved.min_displacement,
                "apple_radius": apple_radius,
            },
        ),
        metadata={"source": "so_arm_mujoco/mjcf/task_scene.xml", "frame": "world"},
    )


def apple_on_plate(
    max_steps: int = 220,
    epochs: int = 1,
    apple_radius: float = APPLE_RADIUS,
    instruction: str = INSTRUCTION,
    **_: Any,
) -> Task:
    """The benchmark: one scene, geometric success plus the simulator's own verdict.

    ``instruction`` is what the policy is *told* to do. It changes nothing
    about scoring: the three scorers measure apple-on-plate geometry whatever
    the text says, so a run given some other goal scores 0 against *this*
    task and that 0 is not a verdict on the goal that was typed.

    It is declared explicitly rather than left to ``**_`` because that
    catch-all swallows a misspelled keyword in silence -- the run would then
    use the default text while the caller believed otherwise, and nothing in
    the log would say so. Callers should still read the instruction back off
    the returned scene; see the guard in ``scripts/molmoact_eval.py``.
    """
    text = str(instruction)
    return Task(
        name="apple_on_plate",
        scenes=[apple_on_plate_scene(apple_radius=float(apple_radius), instruction=text)],
        scorer=["apple_on_plate_success", "sim_task_success", "apple_plate_distance"],
        max_steps=int(max_steps),
        epochs=int(epochs),
        metadata={
            "robot": "SO-101",
            "transport": "rosbridge",
            "instruction": text,
            "apple_radius_m": float(apple_radius),
        },
    )
