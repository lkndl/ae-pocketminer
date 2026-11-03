"""Run the installed `ae-pocketminer` command, as dtand's interface-score does, and check its outputs.

The fixture is PDB entry 2PTC, trypsin and BPTI. Its author chains E and I are
label chains A and B, and trypsin has insertion codes (184A, 188A, 221A).
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "2ptc.cif"
# the table `ae-pocketminer` wrote for the fixture once, checked against the
# predictions of the original code (see docs/development.md)
EXPECTED = FIXTURES / "2ptc_expected.tsv"
MODELS = ["ae-pocketminer", "pocketminer"]


def run(*args):
    # the console script installed next to this interpreter, not any other on the PATH
    command = shutil.which("ae-pocketminer", path=os.path.dirname(sys.executable))
    assert command, "the ae-pocketminer command is not installed in this environment"
    return subprocess.run([command, *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def is_amino_acid(residue):
    info = gemmi.find_tabulated_residue(residue.name)
    return info is not None and info.is_amino_acid()


def author_residues(path):
    # (chain, number, insertion code) of every amino acid residue of the first model
    st = gemmi.read_structure(str(path))
    return {(chain.name, residue.seqid.num, residue.seqid.icode.strip())
            for chain in st[0] for residue in chain if is_amino_acid(residue)}


def write_odd_structure(path):
    # 2PTC with E:100 lacking its O, E:101 unknown (UNK) and E:104 selenomethionine (MSE)
    st = gemmi.read_structure(str(FIXTURE))
    for residue in st[0]["E"]:
        if residue.seqid.num == 100:
            for i in reversed(range(len(residue))):
                if residue[i].name in ("O", "OXT"):
                    del residue[i]
        elif residue.seqid.num == 101:
            residue.name = "UNK"
        elif residue.seqid.num == 104:
            assert residue.name == "MET"
            residue.name = "MSE"
            for atom in residue:
                if atom.name == "SD":
                    atom.name = "SE"
                    atom.element = gemmi.Element("Se")
    st.write_pdb(str(path))


def read_track(path):
    track = json.loads(path.read_text(encoding="utf-8"))
    values = {(v["chain"], v["seq"], v.get("ins_code", "")): v["value"] for v in track["values"]}
    return track, values


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    # one run on the fixture, a variant with odd residues and a file that is no structure
    tmp = tmp_path_factory.mktemp("ae-pocketminer")
    odd = tmp / "odd.pdb"
    write_odd_structure(odd)
    broken = tmp / "broken.pdb"
    broken.write_text("not a structure\n")

    out_dir = tmp / "out"
    result = run(FIXTURE, odd, broken, "-o", out_dir, "-j", "--attention")
    return result, out_dir


def test_fixture_chains_differ():
    # the test is only meaningful if author and label chains differ, and insertion codes occur
    st = gemmi.read_structure(str(FIXTURE))
    assert {chain.name for chain in st[0]} == {"E", "I"}
    assert {residue.subchain for residue in st[0]["E"] if is_amino_acid(residue)} == {"A"}
    assert {residue.subchain for residue in st[0]["I"] if is_amino_acid(residue)} == {"B"}
    assert any(icode for _, _, icode in author_residues(FIXTURE))


def test_failed_input_does_not_stop_the_others(results):
    result, out_dir = results
    assert result.returncode == 1, result.stdout + result.stderr
    assert "broken.pdb" in result.stderr
    assert "1 of 3 inputs failed" in result.stderr
    for name in ("2ptc", "odd"):
        for model in MODELS:
            assert (out_dir / f"{name}_{model}.json").is_file()
    assert not list(out_dir.glob("broken*"))


@pytest.mark.parametrize("model", MODELS)
def test_track(results, model):
    _, out_dir = results
    track, values = read_track(out_dir / f"2ptc_{model}.json")

    # the fields af-molstar-viewer requires, see docs/residue-score-track.md there
    assert track["schema_version"] == 1
    assert track["id"] == f"{model}-cryptic-pocket"
    assert track["label"]
    assert track["index"] == "auth"
    assert track["scale"]["kind"] == "sequential"
    assert track["scale"]["domain"] == [0.0, 1.0]
    assert track["palette"] == {"name": "blues"}
    assert "structure_ref" not in track
    provenance = track["provenance"]
    assert provenance["tool"] == "ae-pocketminer"
    assert provenance["tool_version"] not in ("", "unknown")
    assert provenance["source_structure"] == "2ptc.cif"
    assert provenance["parameters"] == {"model": model}

    # one entry per residue, each a probability
    assert len(values) == len(track["values"]), "duplicate residues"
    for entry in track["values"]:
        assert set(entry) <= {"chain", "seq", "ins_code", "value"}
        assert isinstance(entry["seq"], int)
        assert entry.get("ins_code", "x") != ""
        assert 0.0 <= entry["value"] <= 1.0

    # on the author chains and numbers, insertion codes included, every protein residue
    assert set(values) == author_residues(FIXTURE)
    assert {chain for chain, _, _ in values} == {"E", "I"}


@pytest.mark.parametrize("model", MODELS)
def test_values_match_the_recorded_ones(results, model):
    # a broken weight load gives plausible numbers, but not these
    _, out_dir = results
    _, values = read_track(out_dir / f"2ptc_{model}.json")

    with open(EXPECTED, encoding="utf-8", newline="") as fs:
        expected = {(row["chain"], int(row["seq"]), row["ins_code"]): float(row[model])
                    for row in csv.DictReader(fs, delimiter="\t")}

    assert set(values) == set(expected)
    keys = sorted(expected)
    np.testing.assert_allclose([values[key] for key in keys], [expected[key] for key in keys], atol=1e-3)


@pytest.mark.parametrize("model", MODELS)
def test_odd_residues(results, model):
    result, out_dir = results
    _, values = read_track(out_dir / f"odd_{model}.json")

    # left out of the model input and null, but still in the track
    assert values[("E", 100, "")] is None
    assert values[("E", 101, "")] is None
    # a modified residue is predicted as its parent
    assert values[("E", 104, "")] is not None
    assert sum(value is None for value in values.values()) == 2
    assert set(values) == author_residues(FIXTURE)

    assert "odd.pdb: 2 residues" in result.stderr


def test_attention(results):
    _, out_dir = results
    data = np.load(out_dir / "2ptc_ae-pocketminer_attention.npz")
    n = len(author_residues(FIXTURE))
    assert data["attention"].shape == (2, n, n)
    np.testing.assert_allclose(data["attention"].sum(axis=-1), 1.0, atol=1e-4)

    # rows and columns in the order of the residues of the structure
    _, values = read_track(out_dir / "2ptc_ae-pocketminer.json")
    keys = list(zip(data["chain"].tolist(), data["seq"].tolist(), data["ins_code"].tolist()))
    assert keys == list(values)


def test_mismatched_checkpoint_fails(tmp_path):
    # the AE-PocketMiner model from the PocketMiner weights: the attention layers have none
    from ae_pocketminer.predict import load_model, weights_dir

    for path in weights_dir().glob("pocketminer.*"):
        shutil.copy(path, tmp_path / ("ae" + path.name))
    with pytest.raises(RuntimeError, match="does not fit"):
        load_model("ae-pocketminer", tmp_path)


def test_outputs_must_not_collide(tmp_path):
    # x.pdb and x.cif would write the same files
    shutil.copy(FIXTURE, tmp_path / "x.cif")
    gemmi.read_structure(str(FIXTURE)).write_pdb(str(tmp_path / "x.pdb"))

    result = run(tmp_path, "-o", tmp_path / "out")
    assert result.returncode == 2
    assert "would write the same output files" in result.stderr
