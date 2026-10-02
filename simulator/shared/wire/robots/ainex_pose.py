"""Put the AiNex's arms down at its sides in the vendor's `config/init_pose.yaml`.

The boot's init pose is the `/ainex_controller/init_pose` parameter, served from the
recorded interface (robots_specs/ainex/ros.yml), whose arms are relaxed down at the sides
(the vendor file holds the shoulder roll at +-1.293 with the elbows yawed +-1.926, which on
the simulated model keeps the arms raised in the air). The action groups are generated from
the vendor's file (ainex_actions.py), so this brings that file's arms to the same pose
first: their frames stay offsets from the pose the robot actually stands in. Legs, head
and grippers keep the vendor's values.

    ainex_pose.py <init_pose.yaml>
"""

import sys

import yaml

ARMS_DOWN = {"l_sho_pitch": 0.0, "l_sho_roll": -1.45, "l_el_pitch": 0.0, "l_el_yaw": 0.0,
             "r_sho_pitch": 0.0, "r_sho_roll": 1.45, "r_el_pitch": 0.0, "r_el_yaw": 0.0}


def main(path):
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    doc["init_pose"].update(ARMS_DOWN)
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=False)


if __name__ == "__main__":
    main(sys.argv[1])
