"""The ``apple_on_plate`` task: one scene, one instruction, one scorer.

The simulator stages it (simulator spec §2.3): a red apple and a white plate on the
worktop in front of the SO-101. It passes when the camera-verdict scorer
(``scorer.apple_on_plate``) measures, for at least 1.0 s at the episode's end, the apple
resting released on the plate. An episode ends when the policy reports the task done, or
after `MAX_POLICY_STEPS` policy steps.

The scene carries no geometry to grade against: where the plate is, and where the apple
ended up, are measured from the rig's cameras, so a layout the console was never told
about is graded the same way as the standard one.
"""

from __future__ import annotations

from inspect_robots import Scene, Target, Task

INSTRUCTION = (
    "Move the arm towards the red apple, grasp it, lift it up, and place it on "
    "the white plate."
)

#: An episode ends after this many policy steps if the policy has not said it is done.
MAX_POLICY_STEPS = 220

#: The one scorer, by its registered name.
SCORER = "apple_on_plate"

#: The arm pose every episode starts from: five arm joints, radians, contract order.
#: Mirrors the simulator task's ``START_ARM_QPOS``; a test holds the two together.
START_ARM_QPOS: tuple[float, float, float, float, float] = (0.0, 0.0, -1.5708, 1.0008, -1.5221)


def resolve_instruction(text: str | None = None, path: str | None = None) -> str:
    """Pick the instruction from a literal, a file, or the task's default."""
    if text is not None and path is not None:
        raise ValueError("give either an instruction or a file holding one, not both")
    if path is not None:
        from pathlib import Path as _Path

        return _Path(path).read_text(encoding="utf-8").strip()
    return INSTRUCTION if text is None else text


def instruction_warning(instruction: str) -> str | None:
    """The caveat owed to a run given a non-default instruction, or None.

    Changing the instruction changes what the policy is *told*, not what is *measured*:
    the scorer grades apple-on-plate whatever the text says.
    """
    if instruction == INSTRUCTION:
        return None
    return (
        f"custom instruction: {instruction!r}. The {SCORER} scorer still measures "
        "apple-on-plate, so a 0 means the apple did not end up released on the plate -- "
        "not whether the policy did what it was told."
    )


def apple_on_plate_scene(instruction: str = INSTRUCTION) -> Scene:
    return Scene(
        id="apple-on-plate",
        instruction=instruction,
        target=Target(kind="object_on_receptacle", spec={"object": "apple", "receptacle": "plate"}),
        metadata={"frame": "scene/worktop"},
    )


def apple_on_plate(
    max_steps: int = MAX_POLICY_STEPS,
    epochs: int = 1,
    instruction: str = INSTRUCTION,
) -> Task:
    """The task. Unknown ``-T`` options are an error, not silently swallowed."""
    text = str(instruction)
    return Task(
        name="apple_on_plate",
        scenes=[apple_on_plate_scene(instruction=text)],
        scorer=[SCORER],
        max_steps=int(max_steps),
        epochs=int(epochs),
        metadata={"robot": "SO-101", "transport": "rosbridge", "instruction": text},
    )
