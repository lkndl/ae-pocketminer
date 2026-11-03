"""
find_key_attention_residues.py

Given the attention weights that `ae-pocketminer --attention` writes
(<name>_ae-pocketminer_attention.npz) and a set of known/apo pocket residues,
this script:

  1. For each pocket residue (as query), averages how much every other
     residue (as key) feeds into its representation -- the "rest of the
     protein -> pocket" direction, i.e. which residues may influence or
     induce opening/closing of the pocket.
  2. Excludes the pocket residues themselves and their immediate sequence
     neighbors (adjacent residues trivially get high attention due to local
     structure), replacing their score with a distinct sentinel value.
  3. Prints the top-N remaining residues by attention score.
  4. Writes out a PDB with these per-residue scores set as the B-factor, so
     the result can be visualized directly in PyMOL/Chimera.

Residues are named by author chain, number and insertion code: A:10, A:52A.
A bare number (10) is accepted where only one chain has it.

Usage:
    ae-pocketminer inputs/your_protein.pdb -o results --attention
    python src/find_key_attention_residues.py \\
        --pdb inputs/your_protein.pdb \\
        --attention-weights results/your_protein_ae-pocketminer_attention.npz \\
        --pocket-resnums A:10 A:14 A:15 \\
        --top-n 40 \\
        --output-pdb results/your_protein-attn_Bfactor.pdb
"""

import argparse
import re
import sys

import gemmi
import numpy as np


def load_attention(attention_path):
    """The attention matrix (N, N), averaged over the heads, and the residue
    (chain, number, insertion code) of each row and column.
    """
    data = np.load(attention_path)
    attn = data['attention']
    if attn.ndim == 3:
        attn = attn.mean(axis=0)  # (heads, N, N) -> (N, N)
    residues = list(zip(data['chain'].tolist(), data['seq'].tolist(), data['ins_code'].tolist()))
    if attn.shape != (len(residues), len(residues)):
        raise ValueError(f"Attention matrix shape {attn.shape} does not match its "
                         f"{len(residues)} residues.")
    return attn, residues


def residue_name(residue):
    chain, seq, icode = residue
    return f"{chain}:{seq}{icode}"


def find_residue(text, residues):
    """The index of a residue given as A:10, A:10A or, if unambiguous, 10."""
    chain, _, number = text.rpartition(':')
    m = re.fullmatch(r'(-?\d+)([A-Za-z]?)', number)
    if m is None:
        raise ValueError(f"Cannot read residue {text!r}; expected e.g. A:10 or A:52A.")
    seq, icode = int(m.group(1)), m.group(2)

    matches = [i for i, (c, s, ic) in enumerate(residues)
               if s == seq and ic == icode and (not chain or c == chain)]
    if len(matches) == 0:
        raise ValueError(f"Residue {text} is not among the residues the model saw.")
    if len(matches) > 1 and not chain:
        chains = ', '.join(residue_name(residues[i]) for i in matches)
        raise ValueError(f"Residue {text} is ambiguous ({chains}); give its chain.")
    return matches[0]


def adjacent_residue_inds(selected_inds, residues):
    # the residues themselves and their sequence neighbours in the same chain
    adjacent = set()
    for i in selected_inds:
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(residues) and residues[j][0] == residues[i][0]:
                adjacent.add(j)
    return np.array(sorted(adjacent))


def compute_key_residue_scores(attn, residues, pocket_residues):
    """Returns (scores, excluded_inds) where `scores` has one entry per
    residue (average attention paid to it by the chosen pocket residues),
    and `excluded_inds` are the array indices of the pocket residues
    themselves plus their immediate neighbors.
    """
    selected_inds = [find_residue(text, residues) for text in pocket_residues]
    scores = attn[selected_inds].mean(axis=0)
    excluded_inds = adjacent_residue_inds(selected_inds, residues)
    return scores, excluded_inds


def print_top_residues(scores, excluded_inds, residues, top_n):
    mask = np.ones_like(scores, dtype=bool)
    mask[excluded_inds] = False

    valid_inds = np.arange(len(scores))[mask]
    order = np.argsort(scores[valid_inds])[::-1][:top_n]
    top_inds = valid_inds[order]

    excluded = ' '.join(residue_name(residues[i]) for i in excluded_inds)
    print(f"\nExcluded residues (pocket residues + immediate neighbors): {excluded}")
    top = ' '.join(residue_name(residues[i]) for i in top_inds)
    print(f"\nTop {top_n} residues influencing the given pocket: {top}")

    return top_inds


def write_bfactor_pdb(pdb_path, residues, scores, excluded_inds, exclude_value, output_pdb):
    """Writes a copy of the structure with per-residue attention scores set as
    the B-factor. Excluded residues (pocket residues + neighbors) get
    `exclude_value` instead, so they stand out distinctly in PyMOL/Chimera.
    """
    score_by_residue = {}
    for residue, score in zip(residues, scores.tolist()):
        score_by_residue.setdefault(residue, score)
    excluded = {residues[i] for i in excluded_inds}

    st = gemmi.read_structure(pdb_path)
    n_missing = 0
    for model in st:
        for chain in model:
            for res in chain:
                key = (chain.name, res.seqid.num, res.seqid.icode.strip())
                if key in excluded:
                    value = exclude_value                     # sentinel -- write as-is, NOT scaled
                elif key in score_by_residue:
                    value = score_by_residue[key] * 100       # real score -- scale for B-factor range
                else:
                    value = exclude_value                     # not seen by the model -- also not scaled
                    info = gemmi.find_tabulated_residue(res.name)
                    if info is not None and info.is_amino_acid():
                        n_missing += 1
                for atom in res:
                    atom.b_iso = value

    if n_missing:
        print(f"\nWarning: {n_missing} amino acid residue(s) in the structure had no attention "
              f"score (missing backbone atoms?) and were set to {exclude_value}.")

    if any(len(chain.name) > 2 for chain in st[0]):
        st.shorten_chain_names()
    st.write_pdb(output_pdb)
    print(f"\nWrote B-factor-annotated PDB to {output_pdb}")


def pymol_selection(residues):
    return ' or '.join(f"(chain {c} and resi {s}{i})" for c, s, i in residues) or 'none'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdb', required=True,
                        help='the PDB or mmCIF file the attention weights were computed on')
    parser.add_argument('--attention-weights', required=True,
                        help='the <name>_ae-pocketminer_attention.npz of `ae-pocketminer --attention`')
    parser.add_argument('--pocket-resnums', required=True, nargs='+', metavar='RESIDUE',
                        help='the residues of the known/apo pocket, e.g. A:10 A:14 A:15')
    parser.add_argument('--top-n', type=int, default=40,
                        help='number of top-attended residues to print (default: 40)')
    parser.add_argument('--exclude-value', type=float, default=-100.0,
                        help='B-factor value assigned to pocket residues and their '
                             'immediate neighbors, so they stand out distinctly when '
                             'visualized (default: -100.0)')
    parser.add_argument('--output-pdb', default=None,
                        help='where to write the B-factor-annotated PDB; defaults to '
                             '{pdb}_key_residues.pdb next to the input structure')
    args = parser.parse_args()

    output_pdb = args.output_pdb or args.pdb.rsplit('.', 1)[0] + '_key_residues.pdb'

    attn, residues = load_attention(args.attention_weights)
    try:
        scores, excluded_inds = compute_key_residue_scores(attn, residues, args.pocket_resnums)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    top_inds = print_top_residues(scores, excluded_inds, residues, args.top_n)
    write_bfactor_pdb(args.pdb, residues, scores, excluded_inds, args.exclude_value, output_pdb)

    pocket = [residues[find_residue(text, residues)] for text in args.pocket_resnums]
    excluded = [residues[i] for i in excluded_inds]
    allosteric = [residues[i] for i in top_inds]

    print('\nConsider running the following commands in PyMOL for visualization:')
    print(f'    load {output_pdb}, prot')
    print(f'    select pocket, {pymol_selection(pocket)}')
    print(f'    select excluded, {pymol_selection(excluded)}')
    print(f'    select allosteric, {pymol_selection(allosteric)}')
    print( '    color gray70, excluded')
    print(f'    spectrum b, blue_white_red, prot and not excluded, minimum=0, maximum=1')
