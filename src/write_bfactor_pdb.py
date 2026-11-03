"""Write a copy of a structure with the values of a residue score track as B-factors.

For viewing the predictions of `ae-pocketminer -j` in PyMOL or Chimera. Every
atom of a residue gets the residue's value; residues the track does not hold,
or holds as null, get --fill. Residues are matched by author chain, number and
insertion code, in every chain and every model.

    python src/write_bfactor_pdb.py input/x.pdb output/x_ae-pocketminer.json
"""

import argparse
import json
import sys
from pathlib import Path

import gemmi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('structure', help='the PDB or mmCIF file the track was computed on')
    parser.add_argument('track', type=Path, help='a residue score track, e.g. <name>_ae-pocketminer.json')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='where to write the structure, as mmCIF if it ends in .cif '
                             '(default: the track with .pdb instead of .json)')
    parser.add_argument('--fill', type=float, default=0.0,
                        help='the B-factor of residues without a value (default: %(default)s)')
    args = parser.parse_args()

    track = json.loads(args.track.read_text(encoding='utf-8'))
    if track.get('index') != 'auth':
        sys.exit(f"{args.track} numbers residues by {track.get('index')!r}, only 'auth' is supported")
    values = {(v['chain'], v['seq'], v.get('ins_code', '')): v['value'] for v in track['values']}

    st = gemmi.read_structure(args.structure)
    found = set()
    for model in st:
        for chain in model:
            for residue in chain:
                key = (chain.name, residue.seqid.num, residue.seqid.icode.strip())
                value = values.get(key)
                if key in values:
                    found.add(key)
                bfactor = args.fill if value is None else float(value)
                for atom in residue:
                    atom.b_iso = bfactor

    if len(found) < len(values):
        print(f"WARNING: {len(values) - len(found)} of the {len(values)} residues of the track "
              f"are not in {args.structure}", file=sys.stderr)

    output = args.output or args.track.with_suffix('.pdb')
    if output.suffix.lower() == '.cif':
        st.setup_entities()
        st.make_mmcif_document().write_file(str(output))
    else:
        # the PDB format has room for two characters of chain name only
        if any(len(chain.name) > 2 for chain in st[0]):
            st.shorten_chain_names()
        st.write_pdb(str(output))
    print(f'Wrote {output}')

    print('\nConsider running the following commands in PyMOL for visualization:')
    print(f'    load {output}, prot')
    print('    spectrum b, blue_white_red, prot, minimum=0.5, maximum=0.8')


if __name__ == '__main__':
    main()
