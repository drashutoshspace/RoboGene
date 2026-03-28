import numpy as np
import pytest

from robogene.config import load_config
from robogene.genotype import (
    Genotype,
    Node,
    adjacency_matrix,
    align_tolerance,
    alignment_error,
    arm_chains,
    assembly_genotype,
    connection_rotation,
    edge_transform,
    forward_kinematics,
    module_counts,
    morphology_class,
    validate_genotype,
)
from robogene.schema import default_library, shape_inertia


@pytest.fixture
def lib():
    return default_library()


@pytest.fixture
def cfg():
    return load_config()


def rover(lib) -> Genotype:
    """Body with four wheels and a three-segment arm on top (class 3Arm+4Wheel)."""
    g = Genotype([Node(0, "body_small")], [])
    g.add_node("wheel_left", 0, "+y_front", "hub")
    g.add_node("wheel_left", 0, "+y_rear", "hub")
    g.add_node("wheel_right", 0, "-y_front", "hub")
    g.add_node("wheel_right", 0, "-y_rear", "hub")
    a = g.add_node("arm_link", 0, "+z", "end_a")
    b = g.add_node("arm_link", a, "end_b", "end_a", roll_deg=90)
    g.add_node("arm_link", b, "end_b", "end_a")
    return g


def test_library_entries_are_consistent(lib):
    assert {"body_small", "body_large", "connector", "hinge", "arm_link", "wheel_left", "wheel_right"} <= set(lib.modules)
    for m in lib.modules.values():
        assert m.mass() > 0
        assert np.all(m.inertia() > 0)
        for p in m.ports.values():
            assert np.isclose(np.linalg.norm(p.normal), 1.0)
    assert set(lib.assemblies) == {"arm_3dof", "arm_5dof"}
    wheel = lib.hardware["moonbot_wheel_module"]
    assert len(wheel.joints) == 2 and all(j["velocity_limit"] == pytest.approx(0.654) for j in wheel.joints)
    assert len(lib.hardware["moonbot_limb"].joints) == 7


def test_capsule_inertia_matches_limits():
    # a capsule with no cylinder part is a sphere
    m = 2.0
    sphere = shape_inertia("capsule", [0.1, 0.0], m)
    assert np.allclose(sphere, 0.4 * m * 0.01)


def test_connection_is_antiparallel_and_rolls(lib):
    n_par = np.array([0.0, 1.0, 0.0])
    n_chi = np.array([0.0, 0.0, -1.0])
    for roll in (0, 90, 180, 270):
        R = connection_rotation(n_par, n_chi, roll)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert alignment_error(n_par, R, n_chi) < 1e-9
    assert np.isclose(align_tolerance(5.0), 2 * np.sin(np.deg2rad(2.5)))


def test_ports_coincide_after_transform(lib):
    g = rover(lib)
    for e in g.edges:
        R, t = edge_transform(g, e, lib)
        pp = lib[g.node(e.parent).module].ports[e.parent_port].position
        cp = lib[g.node(e.child).module].ports[e.child_port].position
        assert np.allclose(R @ cp + t, pp)


def test_rover_is_valid_and_named(lib, cfg):
    g = rover(lib)
    ok, why = validate_genotype(g, lib, cfg["checks"], cfg["ga"])
    assert ok, why
    assert morphology_class(g, lib) == "3Arm+4Wheel"
    assert module_counts(g, lib)["wheel"] == 4
    assert arm_chains(g, lib) == [[5, 6, 7]]
    poses = forward_kinematics(g, lib)
    assert np.allclose(poses[1][1], [0.1, 0.125, 0.0])
    assert np.allclose(poses[7][1], [0.0, 0.0, 0.8])
    A = adjacency_matrix(g)
    assert (A == A.T).all() and A.sum() == 2 * len(g.edges)


def test_invalid_genotypes_are_rejected(lib, cfg):
    g = rover(lib)
    bad_roll = g.copy()
    bad_roll.edges[0].roll_deg = 45
    assert not validate_genotype(bad_roll, lib, cfg["checks"], cfg["ga"])[0]

    port_twice = g.copy()
    port_twice.add_node("wheel_left", 0, "+y_front", "hub")
    assert not validate_genotype(port_twice, lib, cfg["checks"], cfg["ga"])[0]

    cycle = g.copy()
    cycle.edges.append(type(g.edges[0])(7, "mid_+x", 5, "mid_+x", 0.0))
    assert not validate_genotype(cycle, lib, cfg["checks"], cfg["ga"])[0]

    task2 = load_config("configs/task2_loco_manip.yaml")
    wheels_only = Genotype([Node(0, "body_small")], [])
    wheels_only.add_node("wheel_left", 0, "+y_front", "hub")
    ok, why = validate_genotype(wheels_only, lib, task2["checks"], task2["ga"])
    assert not ok and "no arm" in why


def test_class_names(lib):
    legged = Genotype([Node(0, "body_small")], [])
    for port in ("+y_front", "-y_front", "+y_rear"):
        legged.add_node("arm_link", 0, port, "end_a")
    assert morphology_class(legged, lib) == "Legged"
    arm = Genotype([Node(0, "body_small")], [])
    arm.add_node("arm_link", 0, "+z", "end_a")
    assert morphology_class(arm, lib) == "Arm Only"
    assert morphology_class(Genotype([Node(0, "body_small")], []), lib) == "Body Only"


def test_hash_ignores_numbering_and_graft(lib):
    g = rover(lib)
    assert g.relabel().canonical_hash() == g.canonical_hash()
    assert Genotype.from_json(g.to_json()).canonical_hash() == g.canonical_hash()
    other = g.copy()
    other.edges[-1].roll_deg = 180
    assert other.canonical_hash() != g.canonical_hash()

    host = Genotype([Node(0, "body_small")], [])
    arm = assembly_genotype(lib, "arm_3dof")
    # graft the chain below its first segment onto the host
    idmap = host.graft(arm, 1, 0, "+z", 0.0)
    assert len(idmap) == 2 and len(host) == 3


def test_remove_subtree(lib):
    g = rover(lib)
    gone = g.remove_subtree(5)
    assert gone == [5, 6, 7]
    assert morphology_class(g, lib) == "4Wheel"
