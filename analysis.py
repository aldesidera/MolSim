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

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, rdMolDescriptors, Crippen, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem import Mol

# ============================================================================
# CONSTANTES
# ============================================================================

# Fingerprint configuration
FINGERPRINT_BITS = 2048
MORGAN_RADIUS = 2
MACC_KEYS = 166

# LogD vs pH calculation
LOGD_PH_COEFFICIENT = 0.12
NEUTRAL_PH = 7
PH_RANGE_START = 0
PH_RANGE_END = 15

# Drawing options
SVG_SIZE_MOLECULE = 400
SVG_SIZE_SIMILARITY_MAP = 800
PNG_SIZE_MOLECULE = 400
PNG_SIZE_SIMILARITY_MAP = 800

# Similarity classification thresholds
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

VALID_FINGERPRINT_TYPES = {"Morgan2", "Morgan2 FCFP", "RDKit", "MACCS"}
VALID_METRICS = {"Tanimoto", "Dice"}

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_fingerprint_type(fp_type: str) -> bool:
    """
    Valida tipo de fingerprint.
    
    Args:
        fp_type: Tipo de fingerprint a validar
        
    Returns:
        True se válido, False caso contrário
        
    Example:
        >>> validate_fingerprint_type("Morgan2")
        True
        >>> validate_fingerprint_type("Invalid")
        False
    """
    return fp_type in VALID_FINGERPRINT_TYPES


def validate_metric(metric: str) -> bool:
    """
    Valida métrica de similaridade.
    
    Args:
        metric: Métrica a validar ('Tanimoto' ou 'Dice')
        
    Returns:
        True se válida, False caso contrário
    """
    return metric in VALID_METRICS


# ============================================================================
# MOLECULAR FUNCTIONS
# ============================================================================

def get_mol(smiles: str) -> Tuple[Optional[Mol], Optional[str]]:
    """
    Converte SMILES em objeto Mol com validação detalhada.
    
    Args:
        smiles: String SMILES da molécula
        
    Returns:
        Tupla (mol, erro) onde mol é objeto RDKit ou None se erro
        
    Example:
        >>> mol, err = get_mol("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirina
        >>> if not err:
        ...     print(f"Molécula carregada: {Chem.MolToSmiles(mol)}")
    """
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
    """
    Calcula propriedades físico-químicas principais.
    
    Args:
        mol: Objeto RDKit Mol
        
    Returns:
        Dicionário com propriedades calculadas ou None
        
    Example:
        >>> mol, _ = get_mol("CC(=O)OC1=CC=CC=C1C(=O)O")
        >>> props = get_properties(mol)
        >>> print(f"MW: {props['Peso Molecular (g/mol)']}") 
    """
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
        
        logger.debug(f"Propriedades calculadas: MW={mw:.2f}, LogP={logp:.3f}")
        
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
    """
    Gera fingerprint baseado no tipo especificado.
    
    Args:
        mol: Objeto RDKit Mol
        fp_type: Tipo de fingerprint ('Morgan2', 'Morgan2 FCFP', 'RDKit', 'MACCS')
        
    Returns:
        Fingerprint ou None se erro
        
    Example:
        >>> mol, _ = get_mol("CC(=O)OC1=CC=CC=C1C(=O)O")
        >>> fp = get_fingerprint(mol, "Morgan2")
        >>> print(f"Fingerprint size: {len(fp)}")
    """
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
        elif fp_type == "Morgan2 FCFP":
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=MORGAN_RADIUS, nBits=FINGERPRINT_BITS, useFeatures=True
            )
        elif fp_type == "RDKit":
            fp = Chem.RDKFingerprint(mol)
        elif fp_type == "MACCS":
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            return None
        
        logger.debug(f"Fingerprint {fp_type} gerado com sucesso")
        return fp
    except Exception as e:
        logger.error(f"Erro ao gerar fingerprint {fp_type}: {str(e)}")
        return None


def calc_similarity(fp1: Any, fp2: Any, metric: str) -> float:
    """
    Calcula similaridade entre dois fingerprints.
    
    Args:
        fp1: Primeiro fingerprint
        fp2: Segundo fingerprint
        metric: Métrica ('Tanimoto' ou 'Dice')
        
    Returns:
        Valor de similaridade (0.0 a 1.0)
        
    Example:
        >>> # Assuming fp1 and fp2 are fingerprints
        >>> sim = calc_similarity(fp1, fp2, "Tanimoto")
        >>> print(f"Similaridade: {sim:.4f}")
    """
    if not validate_metric(metric):
        logger.error(f"Métrica inválida: {metric}")
        return 0.0
    
    try:
        if metric == "Dice":
            similarity = round(DataStructs.DiceSimilarity(fp1, fp2), 4)
        else:  # Tanimoto
            similarity = round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)
        
        logger.debug(f"Similaridade calculada ({metric}): {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Erro ao calcular similaridade: {str(e)}")
        return 0.0


def classify_similarity(similarity: float, metric: str) -> str:
    """
    Classifica o valor de similaridade.
    
    Args:
        similarity: Valor de similaridade (0.0 a 1.0)
        metric: Métrica usada ('Tanimoto' ou 'Dice')
        
    Returns:
        Classificação ('Muito Alta', 'Alta', 'Moderada', 'Baixa', 'Muito Baixa')
        
    Example:
        >>> classification = classify_similarity(0.85, "Tanimoto")
        >>> print(classification)  # 'Muito Alta'
    """
    thresholds = SIMILARITY_THRESHOLDS.get(metric, SIMILARITY_THRESHOLDS["Tanimoto"])
    
    for classification, threshold in sorted(
        thresholds.items(), key=lambda x: x[1], reverse=True
    ):
        if similarity >= threshold:
            return classification
    
    return "Muito Baixa"


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def mol_to_svg(mol: Optional[Mol], size: int = SVG_SIZE_MOLECULE) -> str:
    """
    Gera SVG da molécula com estética aprimorada.
    
    Args:
        mol: Objeto RDKit Mol
        size: Tamanho em pixels (padrão: 400)
        
    Returns:
        String SVG ou string vazia se erro
    """
    try:
        if not mol:
            logger.warning("Tentativa de gerar SVG com mol=None")
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
        
        svg = svg.replace(f'width="{size}px"', 'width="100%"')
        svg = svg.replace(f'height="{size}px"', 'height="100%"')
        
        logger.debug("SVG molécula gerado com sucesso")
        return svg
    except Exception as e:
        logger.error(f"Erro ao gerar SVG: {str(e)}")
        return ""


def generate_similarity_map(
    mol_ref: Optional[Mol],
    mol_test: Optional[Mol],
    size: int = SVG_SIZE_SIMILARITY_MAP
) -> Optional[str]:
    """
    Gera mapa de similaridade entre duas moléculas.
    
    Args:
        mol_ref: Molécula referência
        mol_test: Molécula teste
        size: Tamanho em pixels (padrão: 800)
        
    Returns:
        String SVG ou None se erro
    """
    try:
        if not mol_ref or not mol_test:
            logger.warning("Tentativa de gerar similarity map com moléculas None")
            return None
        
        mol_ref_no_h = Chem.RemoveHs(mol_ref)
        mol_test_no_h = Chem.RemoveHs(mol_test)
        
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        options = drawer.drawOptions()
        options.bondLineWidth = 3
        options.minFontSize = 14
        
        SimilarityMaps.GetSimilarityMapForFingerprint(
            mol_ref_no_h,
            mol_test_no_h,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(
                m, i, radius=MORGAN_RADIUS, useFeatures=True, useChirality=True
            ),
            draw2d=drawer
        )
        
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace(f'width="{size}px"', 'width="100%"').replace(
            f'height="{size}px"', 'height="100%"'
        )
        
        logger.debug("Similarity map SVG gerado com sucesso")
        return svg
    except Exception as e:
        logger.error(f"Erro ao gerar SimilarityMap: {str(e)}")
        return None


def mol_to_png(mol: Optional[Mol], size: int = PNG_SIZE_MOLECULE) -> Optional[bytes]:
    """
    Gera PNG em memória para o PDF.
    
    Args:
        mol: Objeto RDKit Mol
        size: Tamanho em pixels (padrão: 400)
        
    Returns:
        Bytes PNG ou None se erro
    """
    try:
        if not mol:
            logger.warning("Tentativa de gerar PNG com mol=None")
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
        
        logger.debug("PNG molécula gerado com sucesso")
        return drawer.GetDrawingText()
    except Exception as e:
        logger.error(f"Erro ao gerar PNG: {str(e)}")
        return None


def generate_similarity_map_png(
    mol_ref: Optional[Mol],
    mol_test: Optional[Mol],
    size: int = PNG_SIZE_SIMILARITY_MAP
) -> Optional[bytes]:
    """
    Gera SimilarityMap como PNG para o PDF.
    
    Args:
        mol_ref: Molécula referência
        mol_test: Molécula teste
        size: Tamanho em pixels (padrão: 800)
        
    Returns:
        Bytes PNG ou None se erro
    """
    try:
        if not mol_ref or not mol_test:
            logger.warning("Tentativa de gerar similarity map PNG com moléculas None")
            return None
        
        mol_ref_no_h = Chem.RemoveHs(mol_ref)
        mol_test_no_h = Chem.RemoveHs(mol_test)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        options = drawer.drawOptions()
        options.bondLineWidth = 3
        options.minFontSize = 14
        
        SimilarityMaps.GetSimilarityMapForFingerprint(
            mol_ref_no_h,
            mol_test_no_h,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(
                m, i, radius=MORGAN_RADIUS, useFeatures=True, useChirality=True
            ),
            draw2d=drawer
        )
        
        drawer.FinishDrawing()
        logger.debug("Similarity map PNG gerado com sucesso")
        return drawer.GetDrawingText()
    except Exception as e:
        logger.error(f"Erro ao gerar SimilarityMap PNG: {str(e)}")
        return None


def get_3d_molblock(mol: Optional[Mol]) -> Optional[str]:
    """
    Gera coordenadas 3D reais para visualização interativa (3Dmol.js).
    
    Args:
        mol: Objeto RDKit Mol
        
    Returns:
        String MOL block ou None se erro
    """
    try:
        if not mol:
            logger.warning("Tentativa de gerar 3D com mol=None")
            return None
        
        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
        
        logger.debug("Molécula 3D gerada com sucesso")
        return Chem.MolToMolBlock(mol3d)
    except Exception as e:
        logger.error(f"Erro ao gerar 3D: {str(e)}")
        return None


def calc_logd_vs_ph(
    mol: Optional[Mol],
    ph_range: Optional[List[int]] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Calcula LogD em função do pH (aproximação linear).
    
    Args:
        mol: Objeto RDKit Mol
        ph_range: Lista de valores de pH (padrão: 0-14)
        
    Returns:
        Lista de dicionários {pH, LogD} ou None
        
    Example:
        >>> mol, _ = get_mol("CC(=O)OC1=CC=CC=C1C(=O)O")
        >>> logd_data = calc_logd_vs_ph(mol)
        >>> print(f"LogD at pH 7: {logd_data[7]['LogD']}")
    """
    if not mol:
        logger.warning("Tentativa de calcular LogD com mol=None")
        return None
    
    if ph_range is None:
        ph_range = list(range(PH_RANGE_START, PH_RANGE_END))
    
    try:
        logp = Crippen.MolLogP(mol)
        logd_data = []
        
        for ph in ph_range:
            if ph < NEUTRAL_PH:
                logd = logp + (NEUTRAL_PH - ph) * LOGD_PH_COEFFICIENT
            else:
                logd = logp - (ph - NEUTRAL_PH) * LOGD_PH_COEFFICIENT
            
            logd_data.append({"pH": ph, "LogD": round(logd, 3)})
        
        logger.debug(f"LogD vs pH calculado para {len(ph_range)} pontos")
        return logd_data
    except Exception as e:
        logger.error(f"Erro ao calcular LogD vs pH: {str(e)}")
        return None


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare(
    smiles_ref: str,
    smiles_test: str,
    name_ref: str,
    name_test: str,
    fp_type: str,
    metric: str,
    show_map: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Compara duas moléculas calculando similaridade estrutural.
    
    Args:
        smiles_ref: SMILES da molécula referência
        smiles_test: SMILES da molécula teste
        name_ref: Nome da molécula referência
        name_test: Nome da molécula teste
        fp_type: Tipo de fingerprint
        metric: Métrica de comparação ('Tanimoto' ou 'Dice')
        show_map: Se deve gerar similarity map (padrão: True)
        
    Returns:
        Tupla (resultado_dict, erro) com todos os dados de comparação
        
    Example:
        >>> result, error = compare(
        ...     "CC(=O)OC1=CC=CC=C1C(=O)O",
        ...     "CC(=O)NC1=CC=C(O)C=C1",
        ...     "Aspirina",
        ...     "Paracetamol",
        ...     "Morgan2",
        ...     "Tanimoto"
        ... )
        >>> if not error:
        ...     print(f"Similaridade: {result['similarity']}")
    """
    logger.info(f"Iniciando comparação: {name_ref} vs {name_test}")
    
    mol_ref, err_ref = get_mol(smiles_ref)
    mol_test, err_test = get_mol(smiles_test)
    
    if err_ref or err_test:
        error = err_ref or err_test
        logger.error(f"Erro ao carregar moléculas: {error}")
        return None, error
    
    if not mol_ref or not mol_test:
        logger.error("Moléculas None após parsing")
        return None, "SMILES inválido"
    
    props_ref = get_properties(mol_ref)
    props_test = get_properties(mol_test)
    
    fp_ref = get_fingerprint(mol_ref, fp_type)
    fp_test = get_fingerprint(mol_test, fp_type)
    
    if not fp_ref or not fp_test:
        error = f"Erro ao gerar fingerprint {fp_type}"
        logger.error(error)
        return None, error
    
    similarity = calc_similarity(fp_ref, fp_test, metric)
    classification = classify_similarity(similarity, metric)
    
    # Montar dados de propriedades
    props_data = []
    for key in props_ref.keys():
        ref_val = props_ref[key]
        test_val = props_test[key]
        diff = (
            round(abs(ref_val - test_val), 2)
            if isinstance(ref_val, (int, float))
            else "-"
        )
        props_data.append({
            "Propriedade": key,
            "Referência": ref_val,
            "Teste": test_val,
            "Diferença": diff,
        })
    
    # Gerar visualizações
    similarity_map = generate_similarity_map(mol_ref, mol_test) if show_map else None
    png_ref = mol_to_png(mol_ref)
    png_test = mol_to_png(mol_test)
    similarity_map_png = generate_similarity_map_png(mol_ref, mol_test) if show_map else None
    
    # Gerar 3D MolBlocks
    molblock_ref = get_3d_molblock(mol_ref)
    molblock_test = get_3d_molblock(mol_test)
    
    # Calcular LogD vs pH
    logd_ref = calc_logd_vs_ph(mol_ref)
    logd_test = calc_logd_vs_ph(mol_test)
    
    logger.info(f"Comparação concluída: similaridade={similarity}, classificação={classification}")
    
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
        "similarity_map": similarity_map,
        "similarity_map_png": base64.b64encode(similarity_map_png).decode('utf-8') if similarity_map_png else None,
        "molblock_ref": molblock_ref,
        "molblock_test": molblock_test,
        "fp_type": fp_type,
        "logd_ref": logd_ref,
        "logd_test": logd_test,
    }, None


def bulk_compare(
    ref_smiles: str,
    smiles_list: List[str],
    names_list: Optional[List[str]] = None,
    fp_type: str = "Morgan2",
    metric: str = "Tanimoto"
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Processamento em lote otimizado comparando múltiplas moléculas com uma referência.
    
    Args:
        ref_smiles: SMILES da molécula referência
        smiles_list: Lista de SMILES para comparar
        names_list: Lista de nomes (opcional)
        fp_type: Tipo de fingerprint
        metric: Métrica de comparação
        
    Returns:
        Tupla (lista_resultados, erro)
        
    Example:
        >>> results, error = bulk_compare(
        ...     "CC(=O)OC1=CC=CC=C1C(=O)O",
        ...     ["CC(=O)NC1=CC=C(O)C=C1", "CCCc1ccc(cc1)C(C)C(=O)O"],
        ...     ["Paracetamol", "Ibuprofeno"],
        ...     "Morgan2",
        ...     "Tanimoto"
        ... )
        >>> for result in results:
        ...     print(f"{result['name']}: {result['similarity']}")
    """
    logger.info(f"Iniciando bulk compare com {len(smiles_list)} moléculas")
    
    mol_ref, err = get_mol(ref_smiles)
    if err or not mol_ref:
        error = err or "SMILES de referência inválido"
        logger.error(f"Erro na molécula referência: {error}")
        return None, error
    
    fp_ref = get_fingerprint(mol_ref, fp_type)
    if not fp_ref:
        error = f"Erro ao gerar fingerprint {fp_type}"
        logger.error(error)
        return None, error
    
    fps = []
    results = []
    
    for i, smiles in enumerate(smiles_list):
        name = names_list[i] if names_list and i < len(names_list) else f"Mol_{i+1}"
        mol, err_mol = get_mol(smiles)
        
        if err_mol or not mol:
            results.append({
                "name": name,
                "similarity": None,
                "classification": None,
                "error": err_mol
            })
            continue
        
        fp = get_fingerprint(mol, fp_type)
        if not fp:
            results.append({
                "name": name,
                "similarity": None,
                "classification": None,
                "error": "Erro fingerprint"
            })
            continue
        
        fps.append(fp)
        results.append({"name": name, "smiles": smiles, "error": None})
    
    if not fps:
        logger.warning("Nenhum fingerprint válido gerado para bulk compare")
        return results, None
    
    # Calcular similaridades em lote
    if metric == "Dice":
        similarities = DataStructs.BulkDiceSimilarity(fp_ref, fps)
    else:
        similarities = DataStructs.BulkTanimotoSimilarity(fp_ref, fps)
    
    # Atribuir resultados
    sim_idx = 0
    for res in results:
        if res.get("error") is None and res.get("similarity") is None:
            sim = round(similarities[sim_idx], 4)
            res["similarity"] = sim
            res["classification"] = classify_similarity(sim, metric)
            sim_idx += 1
    
    logger.info(f"Bulk compare concluído: {len([r for r in results if r['error'] is None])} moléculas processadas")
    return results, None
