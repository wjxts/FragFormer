from dgllife.utils.featurizers import (ConcatFeaturizer, 
                                       bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot, 
                                       atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, 
                                       atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, 
                                       atom_total_num_H_one_hot, atom_is_chiral_center, atom_chirality_type_one_hot, 
                                       atom_mass)
from functools import partial
import random 
import rdkit 
from rdkit import Chem 
from pathlib import Path 

from typing_extensions import Literal

ATOM_NUMS = 101
ATOM_FEATURE_DIM = 137
BOND_FEATURE_DIM = 14
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
INF = 100000000

SPLIT_TO_ID = {'train':0, 'valid':1, 'test':2}

BENCHMARK = {'moleculenet': ['bbbp', 'bace', 'clintox', 'sider', 'toxcast', 'tox21', 'estrogen', 'metstab', #
                             'freesolv', 'esol', 'lipo', 
                             'muv', 'pcba', 'hiv', 
                             'qm7', 'qm8', 'qm9',
                             'chembl29'], # 'chembl29' is for pretrain       
             'long_range': ['peptide_func', 'peptide_struct'],
             'pharmabench': ['pharm_ames', 'pharm_bbb', # cls
                             'pharm_cyp2c9', 'pharm_cyp2d6', 'pharm_cyp3a4', 
                             'pharm_hum', 'pharm_mou', 'pharm_rat', # LMC exp
                             'pharm_logd', 'pharm_ppb', 'pharm_sol'], } 

CLS_TASKS = {'bbbp', 'bace', 'clintox', 'sider', 'toxcast', 'tox21', 'estrogen', 'metstab', 'muv', 'pcba', 'hiv', 
             'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 
            'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels',
            'cyp2d6_veith', 'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith',
            'herg', 'ames', 'dili',
            'peptide_func',
            'CLASS_EGFR', 'CLASS_KIT', 'CLASS_LOK', 'CLASS_RET', 'CLASS_SLK',
            'hi_drd2', 'hi_hiv', 'hi_kdr', 'hi_sol',
            '3MR', 'benzene', 'mutagenicity', 'liver_1', 'liver_2', 
            'pharm_ames', 'pharm_bbb', }
MULTI_CLS_TASKS = {'liver'}

CLS_METRICS = ['rocauc', 'ap', 'acc', 'f1'] 
MULTI_CLS_METRICS = ['acc'] 
REG_METRICS = ['rmse', 'mae', 'r2']
METRIC_BEST_TYPE = {'rocauc': 'max', 'ap': 'max', 'acc': 'max', 'f1': 'max', 'rmse': 'min', 'mae': 'min', 'r2': 'max'}
METRICS = {'cls': CLS_METRICS, 'reg': REG_METRICS, 'multi_cls': MULTI_CLS_METRICS}


def get_task_type(dataset) -> Literal['cls', 'reg', 'multi_cls']:
    if dataset in CLS_TASKS:
        return 'cls'
    elif dataset in MULTI_CLS_TASKS:
        return 'multi_cls'
    else:
        return 'reg'

def get_task_metrics(dataset):
    return METRICS[get_task_type(dataset)]

def get_split_name(dataset, scaffold_id):
    if (dataset in BENCHMARK['moleculenet'] or 
        dataset in BENCHMARK['pharmabench']):
        return f"scaffold-{scaffold_id}"
    elif dataset in BENCHMARK['long_range']:
        return "split"
    else:
        raise ValueError(f"wrong dataset: {dataset}")
    
DATASET_TASKS = {'bbbp':1,
                 'bace':1,
                 'clintox':2,
                 'sider':27,
                 'toxcast':617,
                 'tox21':12,
                 'estrogen':2,
                 'metstab':2,
                 'peptide_func': 10,
                 'peptide_struct': 11,
                 'qm8': 12,
                 'qm9': 3,
                 'muv': 17,
                 'pcba': 128,
                 }

# 其余数据集的任务数均为1
for name, ds in BENCHMARK.items():
    for d in ds:
        if d not in DATASET_TASKS:
            DATASET_TASKS[d] = 1

BENCHMARK_NAME = {}
for name, ds in BENCHMARK.items():
    for d in ds:
        BENCHMARK_NAME[d] = name 
        
BASE_PATH = {'moleculenet': Path("dataset_base_path/KPGT/datasets"),
             'long_range': Path('dataset_base_path/long_range'),
             'pharmabench': Path('dataset_base_path/pharmabench'), }

bond_featurizer_all = ConcatFeaturizer([ # 14
    partial(bond_type_one_hot, encode_unknown=True), # 5 
    bond_is_conjugated, # 1  
    bond_is_in_ring, # 1 
    partial(bond_stereo_one_hot, encode_unknown=True) # 7
    ])

# atom_featurizer_atom_type_only = ConcatFeaturizer([ 
#     partial(atomic_number_one_hot, encode_unknown=True), #101
#     ])

atom_featurizer_all = ConcatFeaturizer([ # 137
    partial(atomic_number_one_hot, encode_unknown=True), #101
    partial(atom_degree_one_hot, encode_unknown=True), # 12
    atom_formal_charge, # 1
    partial(atom_num_radical_electrons_one_hot, encode_unknown=True), # 6  #"Radical electrons"指的是某个原子上的未配对电子, 一般来说是0
    partial(atom_hybridization_one_hot, encode_unknown=True), # 6
    atom_is_aromatic, # 1
    partial(atom_total_num_H_one_hot, encode_unknown=True), # 6
    atom_is_chiral_center, # 1
    atom_chirality_type_one_hot, # 2
    atom_mass, # 1
    ])


def random_smiles(mol: rdkit.Chem.Mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)

def get_result_file(base_dir:str, dataset:str, scaffold: str, seed: int):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    file = base_dir / f"{dataset}_{scaffold}_{seed}.json"
    return file 