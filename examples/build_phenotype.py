"""Build a genotype, validate it, expand it into a MuJoCo phenotype and save the model."""

import argparse
from pathlib import Path

from robogene.config import load_config
from robogene.genotype import Genotype, Node, morphology_class, validate_genotype
from robogene.phenotype import check_phenotype, expand, model_xml
from robogene.schema import default_library


def rover() -> Genotype:
    """Body with four wheels and a three-segment arm."""
    g = Genotype([Node(0, "body_small")], [])
    for module, port in (("wheel_left", "+y_front"), ("wheel_left", "+y_rear"),
                         ("wheel_right", "-y_front"), ("wheel_right", "-y_rear")):
        g.add_node(module, 0, port, "hub")
    a = g.add_node("arm_link", 0, "+z", "end_a")
    b = g.add_node("arm_link", a, "end_b", "end_a", roll_deg=90)
    g.add_node("arm_link", b, "end_b", "end_a")
    return g


def render(xml_path: Path, png_path: Path, settle_steps: int = 800) -> None:
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.subtree_com[1]
    cam.distance, cam.azimuth, cam.elevation = 2.4, 135, -22
    renderer.update_scene(data, cam)
    Image.fromarray(renderer.render()).save(png_path)
    renderer.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/examples/rover")
    p.add_argument("--render", action="store_true", help="save a PNG (headless: MUJOCO_GL=egl)")
    a = p.parse_args()

    library, cfg = default_library(), load_config()
    g = rover()
    ok, reasons = validate_genotype(g, library, cfg["checks"], cfg["ga"])
    print(f"genotype: valid={ok} {reasons} class={morphology_class(g, library)} modules={len(g)}")

    ph = expand(g, library)
    ok, reasons, metrics = check_phenotype(ph, cfg)
    print(f"phenotype: checks={ok} {reasons} mass={metrics['mass']:.2f} kg actuators={ph.n_actuators}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "genotype.json").write_text(g.to_json())
    (out / "model.xml").write_text(model_xml(ph, cfg))
    if a.render:
        render(out / "model.xml", out / "rover.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
