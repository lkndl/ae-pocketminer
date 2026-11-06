"""Predict cryptic pockets with AE-PocketMiner and PocketMiner.

For each PDB or mmCIF file, writes a table of the per-residue pocket
probabilities and, with -j, one residue score track per model, in the format
of af-molstar-viewer's docs/residue-score-track.md.

All protein chains of the first model of a file are predicted together, as
one unit. Residues are named by author chain, number and insertion code.
Residues that lack N, CA, C or O (OXT stands in for O), or whose amino acid
has no standard parent, are left out of the model input and are null in the
outputs.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata, resources
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np

# the models, by the names used on the command line and in the track IDs
MODELS = {
    'ae-pocketminer': {'label': "AE-PocketMiner", 'checkpoint': "aepocketminer", 'use_attention': True},
    'pocketminer': {'label': "PocketMiner", 'checkpoint': "pocketminer", 'use_attention': False},
}

# a directory with the checkpoints to use instead of the ones in the package
WEIGHTS_ENV = "AE_POCKETMINER_WEIGHTS"

# the backbone atoms the models see, in this order
BACKBONE = ("N", "CA", "C", "O")

# the index of each amino acid in the models' sequence embedding
LOOKUP = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0,
          'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

# the structure files found in a directory ('.pdb1', '.pdb2', ... are bioassemblies)
RE_STRUCTURE = re.compile(r"\.(pdb[0-9]*|ent|cif|mmcif)(\.gz)?$", re.IGNORECASE)

# how many left-out residues a warning names
MAX_NAMED = 10


@dataclass
class Residue:
    """A protein residue of the input, by author chain, number and insertion code."""
    chain: str
    seq: int
    icode: str
    name: str
    # its position in the model input, None if it was left out
    index: Optional[int] = None

    def __str__(self):
        return f"{self.chain}:{self.seq}{self.icode}"


def package_version():
    # the version of the installed package, derived from git when it was built
    try:
        return metadata.version("ae-pocketminer")
    except metadata.PackageNotFoundError:
        return "unknown"


def weights_dir():
    override = os.environ.get(WEIGHTS_ENV)
    if override:
        return Path(override)
    return Path(str(resources.files("ae_pocketminer") / "weights"))


def load_model(name, directory):
    """One model with its weights. Fails if any weight is missing from the checkpoint."""
    # TensorFlow loads only here, so that --help and argument errors are quick
    import tensorflow as tf
    from .models import MQAModel

    spec = MODELS[name]
    prefix = Path(directory) / spec['checkpoint']
    if not prefix.with_name(prefix.name + ".index").is_file():
        raise FileNotFoundError(f"no checkpoint {prefix}.index")

    model = MQAModel(node_features=(8, 50), edge_features=(1, 32), hidden_dim=(16, 100),
                     num_layers=4, dropout=0.1,
                     use_attention=spec['use_attention'], attention_heads=2)

    # the model alone: tf-keras refuses the legacy optimizer state in the checkpoints
    status = tf.train.Checkpoint(model=model).read(str(prefix))

    # the variables are created, and restored, on the first call
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1, 8, 4, 3)).astype(np.float32)
    S = np.zeros((1, 8), dtype=np.int32)
    model(X, S, np.ones((1, 8), dtype=np.float32), train=False, res_level=True)

    # a variable without a value in the checkpoint keeps its random initial one
    try:
        status.assert_existing_objects_matched()
    except AssertionError as e:
        raise RuntimeError(f"checkpoint {prefix} does not fit the {spec['label']} model: {e}") from None
    finally:
        # the optimizer state is left unread on purpose, no warning about it
        status.expect_partial()

    return model


def read_structure(path):
    """The protein residues of the first model, in file order, and the model input.

    Returns the residues, the coordinates of N, CA, C and O [n, 4, 3] in nm (as
    mdtraj gave them in training), the sequence [n], and the warnings.
    """
    st = gemmi.read_structure(str(path))
    if len(st) == 0:
        raise ValueError("no model in the file")
    st.remove_alternative_conformations()

    notes = []
    if len(st) > 1:
        notes.append(f"{len(st)} models, only the first one is predicted")

    residues, coords, sequence, left_out = [], [], [], []
    for chain in st[0]:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue

            # gemmi's chain names and residue numbers are the author ones, for mmCIF too
            residue = Residue(chain.name, res.seqid.num, res.seqid.icode.strip(), res.name)
            residues.append(residue)

            # a modified residue counts as its parent: MSE as M, SEP as S
            letter = info.one_letter_code.upper()
            atoms = [res.find_atom(name, '*') for name in BACKBONE]
            # at a C-terminus, OXT is as good as O
            if atoms[3] is None:
                atoms[3] = res.find_atom("OXT", '*')
            if (letter not in LOOKUP) or any(atom is None for atom in atoms):
                left_out.append(residue)
                continue

            residue.index = len(coords)
            coords.append([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in atoms])
            sequence.append(LOOKUP[letter])

    if len(coords) == 0:
        raise ValueError("no amino acid residue with N, CA, C and O")

    if len(left_out) > 0:
        named = ", ".join(str(residue) for residue in left_out[:MAX_NAMED])
        more = ", ..." if len(left_out) > MAX_NAMED else ""
        notes.append(f"{len(left_out)} residues lack N, CA, C or O or have no standard parent "
                     f"amino acid; they are left out of the model input and null in the outputs: "
                     f"{named}{more}")

    X = (np.array(coords, dtype=np.float64) / 10.0).astype(np.float32)
    S = np.array(sequence, dtype=np.int32)
    return residues, X, S, notes


def predict(model, X, S):
    """The pocket probability of each residue [n], and the attention weights [heads, n, n] or None."""
    mask = np.ones(S.shape, dtype=np.float32)
    p, attention = model(X[None], S[None], mask[None], train=False, res_level=True,
                         return_attention=True)
    return p.numpy()[0], (None if attention is None else attention.numpy()[0])


def track_entries(residues):
    """One track entry per residue, each with its residue, and the number of duplicates left out."""
    entries, seen, num_duplicates = [], set(), 0
    for residue in residues:
        key = (residue.chain, residue.seq, residue.icode)
        if key in seen:
            num_duplicates += 1
            continue
        seen.add(key)

        entry = {'chain': residue.chain, 'idx': residue.seq, 'res_name': residue.name}
        if residue.icode:
            entry['ins_code'] = residue.icode
        entries.append((entry, residue))

    return entries, num_duplicates


def value(residue, p):
    # null, never 0, for a residue the model did not see
    return None if residue.index is None else round(float(p[residue.index]), 4)


def sha1sum(path):
    """The SHA-1 of a file, as `sha1sum` prints it; of the decompressed bytes if it is gzipped."""
    with open(path, 'rb') as fs:
        gzipped = fs.read(2) == b'\x1f\x8b'
    opener = gzip.open if gzipped else open
    digest = hashlib.sha1()
    with opener(path, 'rb') as fs:
        for chunk in iter(lambda: fs.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_track(filepath, name, entries, p, source, source_sha1sum):
    # one residue score track, following docs/residue-score-track.md of af-molstar-viewer
    track = {
        'schema_version': 2,
        'id': f"{name}-cryptic-pocket",
        'label': f"{MODELS[name]['label']} cryptic pocket",
        'index': "auth",
        'scale': {
            'kind': "sequential",
            # the range of the measure, not of this structure
            'domain': [0.0, 1.0],
            'units': "probability",
        },
        # one palette for both models, since they measure the same thing; light to dark
        'palette': {'name': "blues"},
        'provenance': {
            'tool': "ae-pocketminer",
            'tool_version': package_version(),
            'created': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'source_structure': source,
            'source_sha1sum': source_sha1sum,
            'parameters': {'model': name},
        },
        'values': [],
    }

    # one residue per line: indenting every entry makes the file four times longer
    values = ",\n    ".join(json.dumps(dict(entry, value=value(residue, p)), ensure_ascii=False)
                            for entry, residue in entries)
    text = json.dumps(track, indent=2, ensure_ascii=False)
    text = text.replace('"values": []', '"values": [\n    ' + values + '\n  ]')

    with open(filepath, 'w', encoding='utf-8', newline='\n') as fs:
        fs.write(text + '\n')


def write_table(filepath, residues, predictions):
    # one row per protein residue, in file order, empty where the model did not see it
    with open(filepath, 'w', encoding='utf-8', newline='\n') as fs:
        fs.write("\t".join(["chain", "seq", "ins_code", "resname", *predictions]) + "\n")
        for residue in residues:
            cells = ["" if value(residue, p) is None else f"{value(residue, p):.4f}"
                     for p in predictions.values()]
            fs.write("\t".join([residue.chain, str(residue.seq), residue.icode, residue.name, *cells]) + "\n")


def write_attention(filepath, residues, attention):
    # the residues in the order of the matrix rows and columns, i.e. those the model saw
    seen = [residue for residue in residues if residue.index is not None]
    np.savez(filepath,
             attention=attention.astype(np.float32),
             chain=np.array([residue.chain for residue in seen]),
             seq=np.array([residue.seq for residue in seen], dtype=np.int32),
             ins_code=np.array([residue.icode for residue in seen]),
             resname=np.array([residue.name for residue in seen]))


def output_stem(path):
    """The file name without its last extension and a .gz: x.cif.gz -> x, x.model.pdb -> x.model."""
    name = Path(path).name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


def locate_structures(inputs):
    """The structure files named or found directly in the named directories, and the missing paths."""
    paths, missing = [], []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.is_file() and RE_STRUCTURE.search(p.name))
            if len(found) == 0:
                print(f"WARNING: no structure file directly in {path}, subfolders are not searched",
                      file=sys.stderr)
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            missing.append(item)
    return paths, missing


def run(path, models, out_dir, args):
    """Predict one structure with every model and write its outputs."""
    residues, X, S, notes = read_structure(path)
    for note in notes:
        print(f"WARNING: {path}: {note}", file=sys.stderr)

    predictions, attention = {}, None
    for name, model in models.items():
        predictions[name], weights = predict(model, X, S)
        if MODELS[name]['use_attention']:
            attention = weights

    directory = path.parent if out_dir is None else out_dir
    stem = output_stem(path)

    if not args.no_table:
        write_table(directory / f"{stem}_ae-pocketminer.tsv", residues, predictions)

    if args.json:
        entries, num_duplicates = track_entries(residues)
        if num_duplicates > 0:
            print(f"WARNING: {path}: {num_duplicates} residues share their chain, number and insertion "
                  "code with an earlier one and are left out of the tracks", file=sys.stderr)
        source_sha1sum = sha1sum(path)
        for name, p in predictions.items():
            write_track(directory / f"{stem}_{name}.json", name, entries, p, path.name, source_sha1sum)

    if args.attention:
        write_attention(directory / f"{stem}_ae-pocketminer_attention.npz", residues, attention)

    print(f"{path}: {len(S)} of {len(residues)} protein residues predicted", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ae-pocketminer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", metavar="STRUCTURE",
                        help="PDB or mmCIF files, also gzipped, or directories holding them")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="where to write the results (default: next to each input)")
    parser.add_argument("-m", "--model", choices=[*MODELS, "both"], default="both",
                        help="the model to run (default: %(default)s)")
    parser.add_argument("-j", "--json", action="store_true",
                        help="also write one residue score track per model, as <name>_<model>.json")
    parser.add_argument("--no-table", action="store_true",
                        help="do not write the per-residue table <name>_ae-pocketminer.tsv")
    parser.add_argument("--attention", action="store_true",
                        help="also write AE-PocketMiner's attention weights, as "
                             "<name>_ae-pocketminer_attention.npz, for src/find_key_attention_residues.py")
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_version()}")
    args = parser.parse_args(argv)

    # a file name the console cannot encode (a Windows code page, say) must not fail the run
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    names = list(MODELS) if args.model == "both" else [args.model]
    if args.attention and not any(MODELS[name]['use_attention'] for name in names):
        parser.error("--attention needs the ae-pocketminer model")
    if args.no_table and not (args.json or args.attention):
        parser.error("nothing to write: --no-table without -j or --attention")

    # find the input structures
    paths, missing = locate_structures(args.inputs)
    if len(missing) > 0:
        parser.error("no such file or directory: " + ", ".join(missing))
    if len(paths) == 0:
        parser.error("no structure file to predict on")

    # two inputs must not write to the same files, as x.pdb and x.cif would
    stems = {}
    for path in paths:
        key = (path.parent.resolve() if args.out_dir is None else None, output_stem(path))
        if key in stems:
            parser.error(f"{stems[key]} and {path} would write the same output files")
        stems[key] = path

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # TensorFlow's informational messages on stderr, unless asked for
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # load the models once, for all structures
    try:
        directory = weights_dir()
        models = {name: load_model(name, directory) for name in names}
    except Exception as e:
        print(f"ERROR: loading the models: {e}", file=sys.stderr)
        sys.exit(1)

    # a failed structure does not stop the others
    failed = []
    for path in paths:
        try:
            run(path, models, args.out_dir, args)
        except Exception as e:
            print(f"ERROR: {path}: {e}", file=sys.stderr, flush=True)
            failed.append(path)

    # so that a calling script or a scheduler can tell
    if len(failed) > 0:
        print(f"ERROR: {len(failed)} of {len(paths)} inputs failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
