from rdkit import Chem
from fragment_mol.dove.dove_k import graph_dove, TokenizerNew, MolGraph
from fragment_mol.datasets.dataset_frag_graph import Smiles2Fragment


if __name__ == "__main__":
    fname = 'dove/chembl29_selected.txt'

    vocab_len = 500
    order = 1

    vocab_path = f"chembl29_vocab_dove{order}_{vocab}.txt"
    selected_smis, details = graph_dove(fname, vocab_len, vocab_path, order=order)
