from collections import defaultdict

import numpy as np
import pandas as pd 

from rdkit.Chem.Scaffolds import MurckoScaffold

from fragment_mol.utils.chem_utils import BENCHMARK_NAME, BASE_PATH



class ScaffoldSplitter:
    def __init__(self, dataset: str):
        benchmark_name = BENCHMARK_NAME[dataset]
        base_path = BASE_PATH[benchmark_name]
        self.dataset = dataset
        self.benchmark_name = benchmark_name
        self.base_path = base_path
        self.dataset_folder = base_path / dataset

        dataset_file = base_path / dataset / f"{dataset}.csv"
        df = pd.read_csv(dataset_file)
        
        self.smiles = df["smiles"].values.tolist()
    
    
    def split_datset(self, scaffold_id):
        split_file = self.dataset_folder / "splits" / f"scaffold-{scaffold_id}.npy"
        if split_file.exists() and 0:
            return 
        train_idx, valid_idx, test_idx = self.scaffold_split(self.smiles, random_seed=2516+scaffold_id)
        
        print(f"{self.dataset}, Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}, scaffold_id: {scaffold_id}")
        split_dict = {'train': train_idx, 'valid':valid_idx, 'test': test_idx}
        new_split = np.array([np.array(split_dict['train']), np.array(split_dict['valid']), np.array(split_dict['test'])], dtype=object)
        
        print(f"Saving split to {split_file}")
        np.save(split_file, new_split)

    def generate_scaffold(self, smiles, include_chirality=False):
        """
        Obtain Bemis-Murcko scaffold from smiles
        :param smiles:
        :param include_chirality:
        :return: smiles of scaffold
        """
        # reduces each molecule to a scaffold by iteratively removing monovalent atoms until none remain
        # https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality)
        return scaffold

    def scaffold_split(self, smiles_list, frac_train=0.8, frac_valid=0.1, random_seed=12):

        all_scaffolds = defaultdict(list)
        for i, smiles in enumerate(smiles_list):
            scaffold = self.generate_scaffold(smiles, include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        # Sort from largest to smallest scaffold sets
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        np.random.seed(random_seed)
        
        np.random.shuffle(all_scaffold_sets) 

        train_cutoff = frac_train * len(smiles_list)
        valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        return train_idx, valid_idx, test_idx
