import argparse
import os
from pathlib import Path

from pydrake.geometry import StartMeshcat
from pydrake.visualization import ModelVisualizer

ROBOTS = {
    "go2": {
        "package_name": "go2_description",
        "model_file": "urdf/go2_description.urdf",
    },
    "g1": {
        "package_name": "g1_description",
        "model_file": "g1_23dof.urdf",
    },
    "humanoid": {
        "package_name": "humanoid",
        "model_file": "humanoid.xml"
    },
    "smpl": {
        "package_name": "smpl",
        "model_file": "smpl.xml"
    }
}

def main(robot: str = None):
    if robot is None:
        parser = argparse.ArgumentParser(description="Visualize robot model in Meshcat")
        parser.add_argument(
            "--robot",
            type=str,
            default="smpl",
            choices=list(ROBOTS.keys()),
            help="Robot to visualize (default: go2)",
        )
        args = parser.parse_args()
        robot = args.robot

    if robot not in ROBOTS:
        raise ValueError(f"Unknown robot: {robot}. Must be one of: {list(ROBOTS.keys())}")

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    robot_config = ROBOTS[robot]
    robot_pkg = project_root / "urdf" / robot_config["package_name"]
    pkg_xml = robot_pkg / "package.xml"

    meshcat = StartMeshcat()
    visualizer = ModelVisualizer(meshcat=meshcat)

    pm = visualizer.parser().package_map()
    if pkg_xml.exists():
        pm.AddPackageXml(str(pkg_xml))
    else:
        pm.Add(robot_config["package_name"], str(robot_pkg))

    model_file = robot_pkg / robot_config["model_file"]
    visualizer.parser().AddModels(str(model_file))

    test_mode = "TEST_SRCDIR" in os.environ
    visualizer.Run(loop_once=test_mode)


if __name__ == "__main__":
    main()