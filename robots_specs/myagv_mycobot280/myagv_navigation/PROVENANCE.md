# myagv_navigation, as the composite's navigation launch loads it

Byte-for-byte copies of `myagv_navigation/` in
[elephantrobotics/myagv_ros](https://github.com/elephantrobotics/myagv_ros) at
`c71f3cc574e5ed1973a925238eabe88662cfa701` (branch `myagv_ros_2023Pi`, the revision
`robots.yml` pins for this robot), laid out as upstream. They are what
`launch/composite_robot_navigation_active.launch` starts `map_server_for_test`, `amcl` and
`move_base` with: the map `/map` serves, and the parameter files the launch loads
(`amcl.yaml` from `param/base_local_planner_param/`, the path `ros.yml` records as the
user-fixed one). `tests/test_robots_specs.py` holds each file to its hash below.

| file | sha256 |
| --- | --- |
| launch/composite_robot_navigation_active.launch | d38942eead3d781b12a102b6249d4a0969122d6a41f10a36d01b05593084ddaa |
| map/composite_robot_map.pgm | d120a2f4a395f647d4ca3924118bc19c4f9b1ba39fb2184189a3bf47ca6ce018 |
| map/composite_robot_map.yaml | 1eb9e468a4942b47f71fbff3a43067ef3860246ce8e79f538bdb57c5198c0975 |
| param/base_local_planner_param/amcl.yaml | 38b7f8a65fc0fb2d4c6e67aa707175756c08e7d23cc30440e0d7492cd60484e1 |
| param/base_local_planner_param/base_global_planner_params.yaml | 2d0eda8810c035894580d31f516c3548daeaccbace8eaba4443b1957e5854cb7 |
| param/base_local_planner_param/base_local_planner_params.yaml | c6dc365bbbe3c599f77ae9b3b065f6b6132cd001c43d9fb31ee4fdd11f0e2e8a |
| param/base_local_planner_param/costmap_common_params.yaml | 97745a30f08b1e06403108ea23697449ac3f7b0feb640feaaa4f9f47666a2299 |
| param/base_local_planner_param/global_costmap_params.yaml | 92889c5ee52708af64aefad8ca5baef2901360dc2a72cc98830c25be8849746d |
| param/base_local_planner_param/local_costmap_params.yaml | 95770956b0b9679c59bd6867290bae77aaa6a196084de6a0b86f30519d250bfd |
