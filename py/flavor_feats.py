"""PharmaGNN 的分子特征提取（RDKit + numpy）。

与原作者 `flavor/model.py` 的数学口径一致：

- 原子特征 11 维：`get_atom_features_enhanced`（价数/芳香/成环/质量/度数 + N O S F Cl Br）
- 键特征 4 维：键类型 one-hot（SINGLE / DOUBLE / TRIPLE / AROMATIC）。
  注意 checkpoint `best_model_pharma33.pt` 的 `edge_encoder.0.weight` 形状是 (128, 4)，
  即训练时用的是 4 维键特征（model.py 后来扩到 15 维，与本 checkpoint 不匹配）。
- 药效团特征：每个药效团 11 维（类型 + 平均坐标 + 电荷/MR/LogP/TPSA + 中心距离），
  max/mean/sum 三通道池化拼接成 33 维。

图特征取自**去氢**分子（等价于原代码里 `Chem.MolFromSmiles(smiles)`），
药效团特征取自**带氢的 3D 构象**（等价于原代码里 `load_mol_from_sdf` + Gasteiger 电荷）。
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import BondType, HybridizationType

RDLogger.DisableLog("rdApp.*")

# 与 checkpoint 一致的维度
NODE_DIM = 11
EDGE_DIM = 4
PHARMA_RAW_DIM = 11
PHARMA_DIM = PHARMA_RAW_DIM * 3  # 33

FAMILIES = [
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
    "ZnBinder",
]

BOND_TYPE_CHOICES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
HYBRID_CHOICES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]


def _one_hot(value, choices) -> list:
    return [int(value == c) for c in choices]


def atom_features(atom) -> list:
    """11 维原子特征（与 model.py 的 get_atom_features_enhanced 一致）。"""
    symbol = atom.GetSymbol()
    return [
        atom.GetTotalValence(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetMass(),
        atom.GetTotalDegree(),
        int(symbol == "N"),
        int(symbol == "O"),
        int(symbol == "S"),
        int(symbol == "F"),
        int(symbol == "Cl"),
        int(symbol == "Br"),
    ]


def bond_features(bond) -> list:
    """4 维键特征：键类型 one-hot。"""
    return _one_hot(bond.GetBondType(), BOND_TYPE_CHOICES)


class PharmacophoreExtractor:
    """RDKit 药效团特征提取（与 model.py 的 AdvancedPharmacophoreExtractor 一致）。"""

    def __init__(self):
        fdef = f"{RDConfig.RDDataDir}/BaseFeatures.fdef"
        self.factory = AllChem.BuildFeatureFactory(fdef)
        self.last_count = 0  # 最近一次 extract() 识别到的药效团数量

    # ---------- 子特征 ----------

    @staticmethod
    def _atom_props(mol, atom_ids):
        """(平均电荷, 总 MR, 平均 LogP, 局部 TPSA)"""
        if not atom_ids:
            return (0.0, 0.0, 0.0, 0.0)
        try:
            crippen = rdMolDescriptors._CalcCrippenContribs(mol)
            tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
        except Exception:
            crippen = [(0.0, 0.0)] * mol.GetNumAtoms()
            tpsa = [0.0] * mol.GetNumAtoms()

        charges, logps, mrs, local_tpsa = [], [], [], 0.0
        for idx in atom_ids:
            atom = mol.GetAtomWithIdx(idx)
            try:
                charge = atom.GetDoubleProp("_GasteigerCharge")
            except Exception:
                charge = 0.0
            if charge is None or np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
            logps.append(crippen[idx][0])
            mrs.append(crippen[idx][1])
            local_tpsa += tpsa[idx]
        return (float(np.mean(charges)), float(np.sum(mrs)), float(np.mean(logps)), float(local_tpsa))

    @staticmethod
    def _average_xyz(mol, atom_ids):
        if not atom_ids:
            return (0.0, 0.0, 0.0)
        conf = mol.GetConformer()
        acc = np.zeros(3)
        for idx in atom_ids:
            pos = conf.GetAtomPosition(idx)
            acc += np.array([pos.x, pos.y, pos.z])
        avg = acc / len(atom_ids)
        return (float(avg[0]), float(avg[1]), float(avg[2]))

    @staticmethod
    def _center_distances(mol, atom_ids):
        """(最远距离, 最近距离, 平均距离)：药效团原子到自身几何中心的距离统计。"""
        if not atom_ids or len(atom_ids) <= 1:
            return (0.0, 0.0, 0.0)
        conf = mol.GetConformer()
        coords = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in atom_ids]
        )
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        return (float(dists.max()), float(dists.min()), float(dists.mean()))

    # ---------- 主流程 ----------

    def extract(self, mol_3d) -> np.ndarray:
        """带氢的 3D 分子 → 33 维药效团特征向量。"""
        feats = []
        for feat in self.factory.GetFeaturesForMol(mol_3d, confId=-1):
            family = feat.GetFamily()
            if family not in FAMILIES:
                continue
            try:
                atom_ids = list(feat.GetAtomIds())
                vec = [
                    float(FAMILIES.index(family)),
                    *self._average_xyz(mol_3d, atom_ids),
                    *self._atom_props(mol_3d, atom_ids),
                    *self._center_distances(mol_3d, atom_ids),
                ]
                feats.append(vec)
            except Exception:
                continue
        self.last_count = len(feats)
        if not feats:
            return np.zeros(PHARMA_DIM, dtype=np.float32)
        arr = np.asarray(feats, dtype=np.float32)
        return np.concatenate([arr.max(axis=0), arr.mean(axis=0), arr.sum(axis=0)]).astype(np.float32)


def graph_arrays(mol_graph, pharma_vec: np.ndarray):
    """由分子（去氢）与药效团向量构建 GNN 输入：x / edge_index / edge_attr / pharma。"""
    x = np.asarray([atom_features(a) for a in mol_graph.GetAtoms()], dtype=np.float32)
    if x.size == 0:
        x = np.zeros((0, NODE_DIM), dtype=np.float32)

    src, dst, attrs = [], [], []
    for bond in mol_graph.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [f, f]
    if src:
        edge_index = np.asarray([src, dst], dtype=np.int64)
        edge_attr = np.asarray(attrs, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_DIM), dtype=np.float32)
    return x, edge_index, edge_attr, pharma_vec.reshape(1, -1).astype(np.float32)


def graph_mol(mol_3d):
    """取用于 GNN 的分子（去掉显式氢，保留键级/芳香性），等价于 MolFromSmiles 的重原子图。"""
    return Chem.RemoveHs(mol_3d)


def prepare_mol_3d(mol):
    """补全 3D 构象与 Gasteiger 电荷（保证药效团特征可算）。"""
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        ok = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if ok != 0:
            ok = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if ok != 0:
            raise ValueError("无法为该分子生成 3D 构象")
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    elif not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        # 场景里由 xyz 一类文件载入的分子没有显式氢；药效团特征（供体/疏水等）依赖氢的位置，
        # 这里按已有 3D 坐标补出氢（与原训练数据“带氢的 3D 构象”口径一致）。
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass
    AllChem.ComputeGasteigerCharges(mol)
    return mol


def mol_from_smiles(smiles: str):
    """SMILES → 带氢的 3D 分子（与论文一致：RDKit 生成 3D + MMFF94 优化）。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    return prepare_mol_3d(mol)


def mol_from_block(block: str):
    """SDF / MOL block（可能含 3D 坐标）→ 分子；优先用文件自带的坐标与键级。"""
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError("无法解析分子文件")
        Chem.SanitizeMol(mol)
    return prepare_mol_3d(mol)


NO_H_WARNING = (
    "分子没有显式氢、也读不到原始分子文件，键级无法推断（按单键处理），预测结果仅供参考；"
    "建议改用 SMILES 输入，或使用带键级/带氢的 SDF、MOL、XYZ 文件"
)


def mol_from_atoms(syms, xyzs, bonds, charge=0):
    """由原子符号 / 坐标 / 连接表构建分子（用于主程序场景里的分子）。

    主程序给出的连接表没有键级，这里用 RDKit `rdDetermineBonds.DetermineBonds`
    依据 3D 坐标和总电荷推断键级与芳香性。

    :return: (分子, 警告文本或 None)
    """
    from rdkit.Chem import rdDetermineBonds

    def skeleton():
        rw = Chem.RWMol()
        for sym in syms:
            rw.AddAtom(Chem.Atom(str(sym).capitalize()))
        for i, j in bonds:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        m = rw.GetMol()
        conf = Chem.Conformer(m.GetNumAtoms())
        for idx, (x, y, z) in enumerate(xyzs):
            conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
        m.AddConformer(conf, assignId=True)
        return m

    def has_hydrogen(mol):
        return any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())

    # 1) 有显式氢时，DetermineBonds 能准确还原键级与芳香性
    mol = skeleton()
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
        return prepare_mol_3d(mol), None
    except Exception:
        pass

    # 2) 没有显式氢时先按几何补氢，再试一次（有时能救回来）
    if not has_hydrogen(mol):
        try:
            candidate = Chem.AddHs(skeleton(), addCoords=True)
            rdDetermineBonds.DetermineBonds(candidate, charge=int(charge))
            return prepare_mol_3d(candidate), None
        except Exception:
            pass

    # 3) 兜底：干净的骨架（避免上一次推断残留的电荷/键级）+ 跳过价键检查的校正，
    #    再按几何补氢，尽量保住药效团信息；结果仅供参考。
    mol = skeleton()
    try:
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception:
        pass
    if not has_hydrogen(mol):
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass
    return prepare_mol_3d(mol), NO_H_WARNING
