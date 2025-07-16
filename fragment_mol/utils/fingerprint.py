import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.AtomPairs import Torsions
import numpy as np
from tqdm import tqdm 
from multiprocessing import Pool
from functools import partial 

from fragment_mol.utils.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from joblib import Parallel, delayed

generator = RDKit2DNormalized()
# other fp
# topological_torsion
FP_FUNC_DICT = {}
FP_DIM = {"ecfp": 1024,
          "rdkfp": 1024,
          "maccs": 167, 
          "atom_pair": 1024, 
          "pharm": 3348, 
          "torsion": 1024, 
          'md': 200,
          'e3fp': 256,
          'e3fp_rb': 1024}


def register_fp(name):
    def decorator(fp_func):
        FP_FUNC_DICT[name] = fp_func
        return fp_func
    return decorator 

@register_fp("ecfp")
def ecfp_fp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM["ecfp"])
    fp = np.array(fp)
    assert len(fp) == FP_DIM["ecfp"]
    return fp
    
@register_fp("rdkfp")
def rdk_fp(mol):
    fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=FP_DIM["rdkfp"])
    fp = np.array(fp)
    assert len(fp) == FP_DIM["rdkfp"]
    return fp
    
@register_fp("maccs")
def maccs_fp(mol):
    fp = MACCSkeys.GenMACCSKeys(mol) 
    fp = np.array(fp) # shape: (167, )
    assert len(fp) == FP_DIM["maccs"]
    return fp 
    
@register_fp("atom_pair")
def atom_pair_fp(mol):
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FP_DIM["atom_pair"])
    fp = np.array(fp)
    assert len(fp) == FP_DIM["atom_pair"]
    return fp 

def fp_to_np(fp, nbits):
        np_fps = np.zeros((nbits,))
        for k in fp:
            # np_fps[k % 1024] += v
            np_fps[k % nbits] = 1
        return np_fps
    
@register_fp("torsion")
def torsion_fp(mol):
    fp = Torsions.GetTopologicalTorsionFingerprintAsIds(mol) # 是hash到的整数
    np_fps = fp_to_np(fp, FP_DIM['torsion'])
    assert len(np_fps) == FP_DIM['torsion']
    return np_fps 

@register_fp("pharm")
def pharmacophore_fp(mol):
    fdefName = rdkit.RDConfig.RDDataDir + '/BaseFeatures.fdef' # 2988 / 3348
    # fdefName = rdkit.RDConfig.RDDocsDir + '/Book/data/MinimalFeatures.fdef' # 990
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    # Create a SigFactory object
    # sigFactory = SigFactory(factory, minPointCount=2, maxPointCount=3) # for BaseFeatures.fdef
    sigFactory = SigFactory(factory, minPointCount=2, maxPointCount=3, trianglePruneBins=False) # For MinimalFeatures.fdef

    # Restrict the features that should be considered when generating the 2D pharmacophore fingerprints
    sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
    sigFactory.Init()
    # print(sigFactory.GetSigSize())
    # Generate the pharmacophore fingerprint
    fp = Generate.Gen2DFingerprint(mol, sigFactory)
    fp = np.array(fp)
    assert len(fp) == FP_DIM["pharm"]
    return fp 

@register_fp("md")
def mol_descriptor(smiles):
    md = generator.process(smiles)
    md = np.array(md[1:])
    assert len(md) == FP_DIM["md"]
    return md 


from e3fp.pipeline import fprints_from_smiles

def idx2arr(idx, nbits):
    fp = np.zeros(nbits)
    for i in idx:
        fp[i] = 1
    return fp 

@register_fp("e3fp")
def e3_fp(smiles, nbits=FP_DIM["e3fp"]):
    confgen_params = {'max_energy_diff': 20.0, 'first': 1}
    fprint_params = {'bits': nbits, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
    fprints = fprints_from_smiles(smiles, "ritalin", confgen_params=confgen_params, fprint_params=fprint_params)
    fp_bits = fprints[0]
    fp = idx2arr(fp_bits.indices, nbits)
    assert len(fp) == FP_DIM["e3fp"]
    return fp

def merge_idx_common(fprints):
    idx = set(fprints[0].indices)
    for fp in fprints[1:]:
        idx = idx & set(fp.indices)
    return list(idx)

def merge_idx_union(fprints):
    idx = set(fprints[0].indices)
    for fp in fprints[1:]:
        idx = idx | set(fp.indices)
    return list(idx)

@register_fp("e3fp_rb")
def e3_fp_robust(smiles, nbits=FP_DIM["e3fp"], n_sample=3):
    confgen_params = {'max_energy_diff': 20.0, 'first': n_sample}
    fprint_params = {'bits': nbits, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
    fprints = fprints_from_smiles(smiles, "ritalin", confgen_params=confgen_params, fprint_params=fprint_params)
    # fp_bits = fprints[0]
    fp = idx2arr(merge_idx_common(fprints), nbits)
    assert len(fp) == FP_DIM["e3fp"]
    return fp


def get_fp(smiles, fp_name="ecfp"):
    assert fp_name in FP_FUNC_DICT, f"fp_name: {fp_name} not in FP_FUNC_DICT"
    if fp_name in ['md', 'e3fp', 'e3fp_rb']:
        return FP_FUNC_DICT[fp_name](smiles)
    else:
        mol = Chem.MolFromSmiles(smiles)
        return FP_FUNC_DICT[fp_name](mol)

def get_batch_fp(smiless, fp_name, n_jobs=16):

    fp_list = Parallel(n_jobs=n_jobs)(delayed(get_fp)(smiles, fp_name) for smiles in tqdm(smiless))

    return np.stack(fp_list, axis=0)

    
def get_batch_fp_multiprocess(smiless, fp_name, n_jobs=16):
    fp_func = partial(get_fp, fp_name=fp_name)
    fp_list = list(Pool(n_jobs).imap(fp_func, tqdm(smiless)))
    return np.stack(fp_list, axis=0)

def get_full_fp(smiless):
    res = {}
    selected_fp_name = ['ecfp', 'rdkfp', 'maccs', 'atom_pair', 'torsion', 'md', "e3fp"]
    for fp_name in selected_fp_name:
       
        res[fp_name] = get_batch_fp_multiprocess(smiless, fp_name)
    return res 


    
    
    
        