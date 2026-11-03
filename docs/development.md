# Development

How the `ae-pocketminer` command is built, packaged and tested, so that dtand's
target preparation (`interface-score`, in `dtand/score`) can install it with
`uv tool install` on Linux, macOS and Windows and run it like PeSTo, IARA and
ScanNet. Open work is in [PLAN.md](../PLAN.md).

## Install and test

```bash
uv sync
uv run pytest
```

The tests run the installed `ae-pocketminer` command, as `interface-score`
does. GitHub Actions runs them on Linux, macOS and Windows with Python 3.13, and
on Linux with Python 3.10 ([.github/workflows/tests.yml](../.github/workflows/tests.yml)),
after a `uv tool install` of the checkout, which resolves without `uv.lock`.

## Layout

- `src/ae_pocketminer/` is the package: the command (`predict.py`), the model
  (`models.py`, `gvp.py`) and both checkpoints (`weights/`). This is the only
  copy of the checkpoints.
- The training scripts, `datasets.py`, `util.py` and the two helper scripts
  stay in `src/`, outside the package. The training scripts import the model
  from the package and need the `train` dependency group.
- `weights/pocketminer` is byte-identical to the checkpoint in the original
  PocketMiner repository (`Mickdub/gvp`). Run without attention, this code
  gives the same predictions as that repository to the last bit, so one
  command writes both tracks and `gvp` is not installed.

## The weights load only with Keras 2

The checkpoints were written with Keras 2. From TensorFlow 2.16 on, `tf.keras`
is Keras 3, which names layer variables differently: under TensorFlow 2.21 and
Keras 3, only 13 of PocketMiner's 177 weight tensors reach the model and the
rest keep their random initial values, without an error. Hence:

- `models.py` and `gvp.py` import `tf_keras` (Keras 2), never `tensorflow.keras`.
  This does not depend on `TF_USE_LEGACY_KERAS` being set before TensorFlow is
  first imported. The training scripts set that variable, so that their own
  `tf.keras` is Keras 2 as well.
- `load_model` restores the model alone,
  `tf.train.Checkpoint(model=model).read(path)`: with the optimizer passed in,
  tf-keras refuses the checkpoints' legacy Adam state.
- After a first forward pass, which creates the variables,
  `status.assert_existing_objects_matched()` fails the load if any variable
  found no value. All 187 of AE-PocketMiner's and 177 of PocketMiner's load.
  `test_mismatched_checkpoint_fails` checks that a partial load fails.

## Structures are read with gemmi

gemmi names chains and residues by author ID (`auth_asym_id`, `auth_seq_id`)
in both PDB and mmCIF, with insertion codes, which is what the tracks declare
with `"index": "auth"`. mdtraj, used before, did not do this:

- its mmCIF reader switches to *label* chain IDs whenever a file has more label
  chains than author chains, which every PDB entry with ligands or waters has;
- its residues have no insertion code, so 52 and 52A became duplicates;
- it could not read an mmCIF file with an empty `_entity_poly_seq` loop, and
  needed openmm to read mmCIF at all.

What the model sees did not change. The coordinates of N, CA, C and O go in
nm, as mdtraj gave them in training. On seven structures of 35 to 798
residues, the gemmi input is bit-identical to the mdtraj one, and so are the
predictions of both models.

Each value of the outputs comes from the list of residues built with the
coordinates, so it lands on the right chain, number and insertion code:

- Only amino acid residues of the first model are in scope. Others (ligands,
  water, nucleic acids) are not in the tracks at all.
- A modified residue counts as its parent:
  `gemmi.find_tabulated_residue(name).one_letter_code.upper()` maps MSE to M
  and SEP to S.
- A residue lacking N, CA, C or O, or without a standard parent (UNK), is left
  out of the model input, is null in the tracks and empty in the table, and is
  counted in a warning. OXT stands in for a missing O; the features use only
  N, CA and C anyway (moving O leaves the predictions bit-identical).
- The tracks keep the first of residues that share chain, number and
  insertion code, and warn about the others.

## Outputs

- `<stem>_ae-pocketminer.tsv` (unless `--no-table`), `<stem>_<model>.json`
  with `-j`, `<stem>_ae-pocketminer_attention.npz` with `--attention`, where
  `<stem>` is the file name without its last extension and `.gz`.
- Two inputs of one run that would write the same files, such as `x.pdb` and
  `x.cif`, are refused before anything runs.
- The tracks are `ae-pocketminer-cryptic-pocket` and
  `pocketminer-cryptic-pocket`, `sequential` on `[0, 1]`, both with the
  `blues` palette, since they measure the same thing and PeSTo (greens), IARA
  (purples) and ScanNet (orange-red) do not use it.
- The model never writes into the working directory: `MQAModel` returns the
  attention matrix with `return_attention=True`.
- A failed input is reported, the others are still predicted, and the command
  exits with status 1. Usage errors exit with status 2.

## Tests

`tests/fixtures/2ptc.cif` is PDB entry 2PTC (trypsin and BPTI, from PeSTo's
tests). Its author chains E and I are label chains A and B, and trypsin has
insertion codes (184A, 188A, 221A); there are also a calcium ion and waters.

`tests/fixtures/2ptc_expected.tsv` holds the predictions of both models for
it, written by `ae-pocketminer` on 2026-11-03 and checked bit for bit against
the original mdtraj code with the weight fix. Its PocketMiner values also
match the [web interface](https://pocketminer.azurewebsites.net/) (checked by
hand on 2026-11-03). Later runs must match within
0.001: a broken weight load gives plausible numbers, but not these. Should the
model or its input deliberately change, record it again with
`ae-pocketminer tests/fixtures/2ptc.cif -o <dir>` and copy the table.

## Versions come from git tags

hatch-vcs derives the version from git: tag `v0.2.0` builds as `0.2.0`, later
commits as `0.2.1.devN+g<hash>`. It ends up in `provenance.tool_version` of
every track. Tag each change that dtand should pick up, and never move a tag
once pushed: uv caches a build per commit. Without git, e.g. from a ZIP
download, the version is `0.0.0`.

## Dependencies

- `uv tool install` resolves from `pyproject.toml` and ignores `uv.lock`, so
  every dependency keeps a lower bound there.
- `tf-keras>=2.21,<2.22` pins `tensorflow` to the same minor version, which
  `tensorflow>=2.21,<2.22` states again. Update both together, and only to a
  TensorFlow that still loads the checkpoints (the tests tell).
- `requires-python` stops before 3.14, for which TensorFlow 2.21 has no
  wheels. It has wheels for 3.10 to 3.13 on macOS arm64, Linux x86_64 and
  aarch64 (glibc 2.27) and Windows. Both models give identical predictions on
  Python 3.10 and 3.13.
- TensorFlow from PyPI brings no CUDA libraries unless asked for
  (`tensorflow[and-cuda]`); the command runs on the CPU.

## Releases

The repository's history is about 340 MB, almost all of it the training data
in `data/`, and a `git+` install clones it. For each `v*` tag, CI builds the
wheel (about 17 MB, both checkpoints included, the same for every platform)
and attaches it to the GitHub release of that tag, so that

```bash
uv tool install https://github.com/lkndl/ae-pocketminer/releases/download/v0.2.0/ae_pocketminer-0.2.0-py3-none-any.whl
```

installs without cloning. The sdist holds only the package, its tests and the
top-level files.

## Platform notes

- TensorFlow's informational messages are hidden: the command sets
  `TF_CPP_MIN_LOG_LEVEL=1` unless it is set already.
- The command's messages are ASCII apart from file paths. A character the
  console's encoding lacks (a Windows code page, say) is printed as a
  backslash escape instead of failing the run.
