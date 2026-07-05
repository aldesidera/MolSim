"""
MolSim Final v3.3 — Análise Molecular Avançada
• Visualização 3D das moléculas
• Layout mais compacto e profissional
• Similarity Map aprimorado
• Suporte a PDF com PNGs
• Type hints, constantes e logging (v3.3)
"""

import logging
from typing import Tuple, Dict, Optional, List, Any
import base64
import io

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, rdMolDescriptors, Crippen, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem import Mol

# ============================================================================
# CONSTANTES
# ============================================================================

FINGERPRINT_BITS = 2048
MORGAN_RADIUS = 2
MACC_KEYS = 166

LOGD_PH_COEFFICIENT = 0.12
NEUTRAL_PH = 7
PH_RANGE_START = 0
PH_RANGE_END = 15

SVG_SIZE_MOLECULE = 400
SVG_SIZE_SIMILARITY_MAP = 800
PNG_SIZE_MOLECULE = 400
PNG_SIZE_SIMILARITY_MAP = 800
FINGERPRINT_IMAGE_SIZE = 300

SIMILARITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Tanimoto": {
        "Muito Alta": 0.85,
        "Alta": 0.65,
        "Moderada": 0.40,
        "Baixa": 0.20,
        "Muito Baixa": 0.0,
    },
    "Dice": {
        "Muito Alta": 0.90,
        "Alta": 0.70,
        "Moderada": 0.50,
        "Baixa": 0.30,
        "Muito Baixa": 0.0,
    },
}

VALID_FINGERPRINT_TYPES = {"Morgan2", "RDKit", "MACCS"}
VALID_METRICS = {"Tanimoto", "Dice"}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_fingerprint_type(fp_type: str) -> bool:
    return fp_type in VALID_FINGERPRINT_TYPES


def validate_metric(metric: str) -> bool:
    return metric in VALID_METRICS


def get_mol(smiles: str) -> Tuple[Optional[Mol], Optional[str]]:
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, "SMILES inválido – não foi possível parsear."

        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        logger.info(f"Molécula carregada com sucesso: {smiles[:50]}")
        return mol, None
    except Exception as e:
        err_str = str(e).lower()
        if "valence" in err_str:
            error_msg = "Erro de Valência: átomo com número inválido de ligações."
        elif any(x in err_str for x in ["ring", "kekul", "aromatic"]):
            error_msg = "Erro de anel ou Kekulização: estrutura aromática inválida."
        else:
            error_msg = f"Erro ao processar SMILES: {str(e)[:100]}"
        logger.error(f"Erro ao carregar molécula: {error_msg}")
        return None, error_msg


def get_properties(mol: Optional[Mol]) -> Optional[Dict[str, Any]]:
    if not mol:
        logger.warning("Tentativa de calcular propriedades com mol=None")
        return None

    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)

        return {
            "Peso Molecular (g/mol)": round(mw, 2),
            "Coeficiente de Partição (LogP)": round(logp, 3),
            "Área de Superfície Polar (Å²)": round(tpsa, 2),
            "Doadores de H (HBD)": int(hbd),
            "Receptores de H (HBA)": int(hba),
            "Ligações Rotacionais": int(rot_bonds),
        }
    except Exception as e:
        logger.error(f"Erro ao calcular propriedades: {str(e)}")
        return None


def get_fingerprint(mol: Optional[Mol], fp_type: str) -> Optional[Any]:
    if not mol:
        logger.warning("Tentativa de gerar fingerprint com mol=None")
        return None
    if not validate_fingerprint_type(fp_type):
        logger.error(f"Tipo de fingerprint inválido: {fp_type}")
        return None

    try:
        if fp_type == "Morgan2":
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=MORGAN_RADIUS, nBits=FINGERPRINT_BITS
            )
        elif fp_type == "RDKit":
            fp = Chem.RDKFingerprint(mol)
        elif fp_type == "MACCS":
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            return None
        return fp
    except Exception as e:
        logger.error(f"Erro ao gerar fingerprint {fp_type}: {str(e)}")
        return None


def fingerprint_to_png(mol: Optional[Mol], fp_type: str, size: int = FINGERPRINT_IMAGE_SIZE) -> Optional[str]:
    try:
        fp = get_fingerprint(mol, fp_type)
        if fp is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DCairo(size, 120)
        drawer.DrawString(f"Fingerprint: {fp_type}")
        drawer.FinishDrawing()
        return base64.b64encode(drawer.GetDrawingText()).decode('utf-8')
    except Exception as e:
        logger.error(f"Erro ao gerar imagem do fingerprint: {str(e)}")
        return None


def calc_similarity(fp1: Any, fp2: Any, metric: str) -> float:
    if not validate_metric(metric):
        logger.error(f"Métrica inválida: {metric}")
        return 0.0

    try:
        if metric == "Dice":
            return round(DataStructs.DiceSimilarity(fp1, fp2), 4)
        return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)
    except Exception as e:
        logger.error(f"Erro ao calcular similaridade: {str(e)}")
        return 0.0


def classify_similarity(similarity: float, metric: str) -> str:
    thresholds = SIMILARITY_THRESHOLDS.get(metric, SIMILARITY_THRESHOLDS["Tanimoto"])
    for classification, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
        if similarity >= threshold:
            return classification
    return "Muito Baixa"


def mol_to_svg(mol: Optional[Mol], size: int = SVG_SIZE_MOLECULE) -> str:
    try:
        if not mol:
            return ""
        mol_no_h = Chem.RemoveHs(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.prepareMolsBeforeDrawing = True
        options.bondLineWidth = 2.5
        options.minFontSize = 14
        options.annotationFontScale = 0.85
        drawer.DrawMolecule(mol_no_h)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace(f'width="{size}px"', 'width="100%"').replace(f'height="{size}px"', 'height="100%"')
    except Exception as e:
        logger.error(f"Erro ao gerar SVG: {str(e)}")
        return ""


def generate_similarity_map(mol_ref: Optional[Mol], mol_test: Optional[Mol], size: int = SVG_SIZE_SIMILARITY_MAP) -> Optional[str]:
    try:
        if not mol_ref or not mol_test:
            return None
        mol_ref_no_h = Chem.RemoveHs(mol_ref)
        mol_test_no_h = Chem.RemoveHs(mol_test)
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        drawer.drawOptions().bondLineWidth = 3
        drawer.drawOptions().minFontSize = 14
        SimilarityMaps.GetSimilarityMapForFingerprint(
            mol_ref_no_h,
            mol_test_no_h,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(m, i, radius=MORGAN_RADIUS, useFeatures=True, useChirality=True),
            draw2d=drawer,
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace(f'width="{size}px"', 'width="100%"').replace(f'height="{size}px"', 'height="100%"')
    except Exception as e:
        logger.error(f"Erro ao gerar SimilarityMap: {str(e)}")
        return None


def mol_to_png(mol: Optional[Mol], size: int = PNG_SIZE_MOLECULE) -> Optional[bytes]:
    try:
        if not mol:
            return None
        mol_no_h = Chem.RemoveHs(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.prepareMolsBeforeDrawing = True
        options.bondLineWidth = 2.5
        options.minFontSize = 14
        drawer.DrawMolecule(mol_no_h)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        logger.error(f"Erro ao gerar PNG: {str(e)}")
        return None


def generate_similarity_map_png(mol_ref: Optional[Mol], mol_test: Optional[Mol], size: int = PNG_SIZE_SIMILARITY_MAP) -> Optional[bytes]:
    try:
        if not mol_ref or not mol_test:
            return None
        mol_ref_no_h = Chem.RemoveHs(mol_ref)
        mol_test_no_h = Chem.RemoveHs(mol_test)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        drawer.drawOptions().bondLineWidth = 3
        drawer.drawOptions().minFontSize = 14
        SimilarityMaps.GetSimilarityMapForFingerprint(
            mol_ref_no_h,
            mol_test_no_h,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(m, i, radius=MORGAN_RADIUS, useFeatures=True, useChirality=True),
            draw2d=drawer,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        logger.error(f"Erro ao gerar SimilarityMap PNG: {str(e)}")
        return None


def get_3d_molblock(mol: Optional[Mol]) -> Optional[str]:
    try:
        if not mol:
            return None
        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
        return Chem.MolToMolBlock(mol3d)
    except Exception as e:
        logger.error(f"Erro ao gerar 3D: {str(e)}")
        return None


def calc_logd_vs_ph(mol: Optional[Mol], ph_range: Optional[List[int]] = None) -> Optional[List[Dict[str, Any]]]:
    if not mol:
        return None
    if ph_range is None:
        ph_range = list(range(PH_RANGE_START, PH_RANGE_END))
    try:
        logp = Crippen.MolLogP(mol)
        return [
            {"pH": ph, "LogD": round(logp + (NEUTRAL_PH - ph) * LOGD_PH_COEFFICIENT, 3)}
            if ph < NEUTRAL_PH else
            {"pH": ph, "LogD": round(logp - (ph - NEUTRAL_PH) * LOGD_PH_COEFFICIENT, 3)}
            for ph in ph_range
        ]
    except Exception as e:
        logger.error(f"Erro ao calcular LogD vs pH: {str(e)}")
        return None


def compare(smiles_ref: str, smiles_test: str, name_ref: str, name_test: str, fp_type: str, metric: str, show_map: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    mol_ref, err_ref = get_mol(smiles_ref)
    mol_test, err_test = get_mol(smiles_test)

    if err_ref or err_test:
        return None, err_ref or err_test
    if not mol_ref or not mol_test:
        return None, "SMILES inválido"

    props_ref = get_properties(mol_ref)
    props_test = get_properties(mol_test)
    fp_ref = get_fingerprint(mol_ref, fp_type)
    fp_test = get_fingerprint(mol_test, fp_type)

    if not fp_ref or not fp_test:
        return None, f"Erro ao gerar fingerprint {fp_type}"

    similarity = calc_similarity(fp_ref, fp_test, metric)
    classification = classify_similarity(similarity, metric)

    props_data = []
    for key in props_ref.keys():
        ref_val = props_ref[key]
        test_val = props_test[key]
        diff = round(abs(ref_val - test_val), 2) if isinstance(ref_val, (int, float)) else "-"
        props_data.append({"Propriedade": key, "Referência": ref_val, "Teste": test_val, "Diferença": diff})

    similarity_map = generate_similarity_map(mol_ref, mol_test) if show_map else None
    png_ref = mol_to_png(mol_ref)
    png_test = mol_to_png(mol_test)
    similarity_map_png = generate_similarity_map_png(mol_ref, mol_test) if show_map else None
    molblock_ref = get_3d_molblock(mol_ref)
    molblock_test = get_3d_molblock(mol_test)
    logd_ref = calc_logd_vs_ph(mol_ref)
    logd_test = calc_logd_vs_ph(mol_test)

    return {
        "name_ref": name_ref,
        "name_test": name_test,
        "similarity": similarity,
        "classification": classification,
        "properties": props_data,
        "svg_ref": mol_to_svg(mol_ref),
        "svg_test": mol_to_svg(mol_test),
        "png_ref": base64.b64encode(png_ref).decode('utf-8') if png_ref else None,
        "png_test": base64.b64encode(png_test).decode('utf-8') if png_test else None,
        "fingerprint_ref_png": fingerprint_to_png(mol_ref, fp_type),
        "fingerprint_test_png": fingerprint_to_png(mol_test, fp_type),
        "similarity_map": similarity_map,
        "similarity_map_png": base64.b64encode(similarity_map_png).decode('utf-8') if similarity_map_png else None,
        "molblock_ref": molblock_ref,
        "molblock_test": molblock_test,
        "fp_type": fp_type,
        "logd_ref": logd_ref,
        "logd_test": logd_test,
    }, None


def bulk_compare(ref_smiles: str, smiles_list: List[str], names_list: Optional[List[str]] = None, fp_type: str = "Morgan2", metric: str = "Tanimoto") -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    mol_ref, err = get_mol(ref_smiles)
    if err or not mol_ref:
        return None, err or "SMILES de referência inválido"

    fp_ref = get_fingerprint(mol_ref, fp_type)
    if not fp_ref:
        return None, f"Erro ao gerar fingerprint {fp_type}"

    fps = []
    results = []
    for i, smiles in enumerate(smiles_list):
        name = names_list[i] if names_list and i < len(names_list) else f"Mol_{i+1}"
        mol, err_mol = get_mol(smiles)
        if err_mol or not mol:
            results.append({"name": name, "similarity": None, "classification": None, "error": err_mol})
            continue
        fp = get_fingerprint(mol, fp_type)
        if not fp:
            results.append({"name": name, "similarity": None, "classification": None, "error": "Erro fingerprint"})
            continue
        fps.append(fp)
        results.append({"name": name, "smiles": smiles, "error": None})

    if not fps:
        return results, None

    similarities = DataStructs.BulkDiceSimilarity(fp_ref, fps) if metric == "Dice" else DataStructs.BulkTanimotoSimilarity(fp_ref, fps)

    sim_idx = 0
    for res in results:
        if res.get("error") is None and res.get("similarity") is None:
            sim = round(similarities[sim_idx], 4)
            res["similarity"] = sim
            res["classification"] = classify_similarity(sim, metric)
            sim_idx += 1

    return results, None
