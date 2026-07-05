"""
MolSim Final v3.2.4 — Análise Molecular Avançada
• Visualização 3D das moléculas
• Layout mais compacto e profissional
• Similarity Map aprimorado
• Suporte a PDF com PNGs
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, rdMolDescriptors, Crippen, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
import io
import base64


def get_mol(smiles):
    """Converte SMILES em objeto Mol com validação detalhada."""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, "SMILES inválido – não foi possível parsear."
        
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        return mol, None
    except Exception as e:
        err_str = str(e).lower()
        if "valence" in err_str:
            return None, "Erro de Valência: átomo com número inválido de ligações."
        elif any(x in err_str for x in ["ring", "kekul", "aromatic"]):
            return None, "Erro de anel ou Kekulização: estrutura aromática inválida."
        else:
            return None, f"Erro ao processar SMILES: {str(e)[:100]}"


def get_properties(mol):
    """Calcula propriedades físico-químicas principais."""
    if not mol:
        return None
    
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


def get_fingerprint(mol, fp_type):
    """Gera fingerprint."""
    try:
        if fp_type == "Morgan2":
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        elif fp_type == "RDKit":
            return Chem.RDKFingerprint(mol)
        elif fp_type == "MACCS":
            return MACCSkeys.GenMACCSKeys(mol)
        else:
            return None
    except Exception as e:
        print(f"Erro ao gerar fingerprint {fp_type}: {str(e)}")
        return None


def calc_similarity(fp1, fp2, metric):
    """Calcula similaridade."""
    try:
        if metric == "Dice":
            return round(DataStructs.DiceSimilarity(fp1, fp2), 4)
        else:
            return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)
    except Exception as e:
        print(f"Erro ao calcular similaridade: {str(e)}")
        return 0.0


def mol_to_svg(mol, size=400):
    """Gera SVG da molécula com estética aprimorada."""
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
        
        svg = svg.replace(f'width="{size}px"', 'width="100%"')
        svg = svg.replace(f'height="{size}px"', 'height="100%"')
        
        return svg
    except Exception as e:
        print(f"Erro ao gerar SVG: {str(e)}")
        return ""


def generate_similarity_map(mol_ref, mol_test, size=800):
    """Mapa de similaridade com useChirality=True."""
    try:
        if not mol_ref or not mol_test:
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
            lambda m, i: SimilarityMaps.GetMorganFingerprint(m, i, radius=2, useFeatures=True, useChirality=True),
            draw2d=drawer
        )
        
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace(f'width="{size}px"', 'width="100%"').replace(f'height="{size}px"', 'height="100%"')
        return svg
    except Exception as e:
        print(f"Erro ao gerar SimilarityMap: {str(e)}")
        return None


def mol_to_png(mol, size=400):
    """Gera PNG em memória para o PDF."""
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
        print(f"Erro ao gerar PNG: {str(e)}")
        return None


def generate_similarity_map_png(mol_ref, mol_test, size=800):
    """Gera SimilarityMap como PNG para o PDF."""
    try:
        if not mol_ref or not mol_test:
            return None
        mol_ref_no_h = Chem.RemoveHs(mol_ref)
        mol_test_no_h = Chem.RemoveHs(mol_test)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        options = drawer.drawOptions()
        options.bondLineWidth = 3
        options.minFontSize = 14
        SimilarityMaps.GetSimilarityMapForFingerprint(
            mol_ref_no_h, mol_test_no_h,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(m, i, radius=2, useFeatures=True, useChirality=True),
            draw2d=drawer
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        print(f"Erro ao gerar SimilarityMap PNG: {str(e)}")
        return None


def get_3d_molblock(mol):
    """Gera coordenadas 3D reais para visualização interativa (3Dmol.js)."""
    try:
        if not mol:
            return None
        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
        return Chem.MolToMolBlock(mol3d)
    except Exception as e:
        print(f"Erro ao gerar 3D: {str(e)}")
        return None


def calc_logd_vs_ph(mol, ph_range=None):
    """Calcula LogD em função do pH (aproximação linear)."""
    if not mol:
        return None
    if ph_range is None:
        ph_range = list(range(0, 15))
    
    logp = Crippen.MolLogP(mol)
    logd_data = []
    for ph in ph_range:
        if ph < 7:
            logd = logp + (7 - ph) * 0.12
        else:
            logd = logp - (ph - 7) * 0.12
        logd_data.append({"pH": ph, "LogD": round(logd, 3)})
    return logd_data


def compare(smiles_ref, smiles_test, name_ref, name_test, fp_type, metric, show_map=True):
    """Compara duas moléculas (versão final v3.2.4)."""
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
    
    # Classificação
    if metric == "Dice":
        if similarity >= 0.90: classification = "Muito Alta"
        elif similarity >= 0.70: classification = "Alta"
        elif similarity >= 0.50: classification = "Moderada"
        elif similarity >= 0.30: classification = "Baixa"
        else: classification = "Muito Baixa"
    else:
        if similarity >= 0.85: classification = "Muito Alta"
        elif similarity >= 0.65: classification = "Alta"
        elif similarity >= 0.40: classification = "Moderada"
        elif similarity >= 0.20: classification = "Baixa"
        else: classification = "Muito Baixa"
    
    props_data = []
    for key in props_ref.keys():
        ref_val = props_ref[key]
        test_val = props_test[key]
        diff = round(abs(ref_val - test_val), 2) if isinstance(ref_val, (int, float)) else "-"
        props_data.append({
            "Propriedade": key,
            "Referência": ref_val,
            "Teste": test_val,
            "Diferença": diff
        })
    
    # SVGs e PNGs
    similarity_map = generate_similarity_map(mol_ref, mol_test) if show_map else None
    png_ref = mol_to_png(mol_ref)
    png_test = mol_to_png(mol_test)
    similarity_map_png = generate_similarity_map_png(mol_ref, mol_test) if show_map else None
    
    # 3D MolBlocks
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
        "similarity_map": similarity_map,
        "similarity_map_png": base64.b64encode(similarity_map_png).decode('utf-8') if similarity_map_png else None,
        "molblock_ref": molblock_ref,
        "molblock_test": molblock_test,
        "fp_type": fp_type,
        "logd_ref": logd_ref,
        "logd_test": logd_test,
    }, None


def bulk_compare(ref_smiles, smiles_list, names_list=None, fp_type="Morgan2", metric="Tanimoto"):
    """Processamento em lote otimizado com Bulk Similarity."""
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
    
    if metric == "Dice":
        similarities = DataStructs.BulkDiceSimilarity(fp_ref, fps)
    else:
        similarities = DataStructs.BulkTanimotoSimilarity(fp_ref, fps)
    
    sim_idx = 0
    for res in results:
        if res.get("error") is None and res.get("similarity") is None:
            sim = round(similarities[sim_idx], 4)
            res["similarity"] = sim
            if metric == "Dice":
                cls = "Muito Alta" if sim >= 0.90 else "Alta" if sim >= 0.70 else "Moderada" if sim >= 0.50 else "Baixa" if sim >= 0.30 else "Muito Baixa"
            else:
                cls = "Muito Alta" if sim >= 0.85 else "Alta" if sim >= 0.65 else "Moderada" if sim >= 0.40 else "Baixa" if sim >= 0.20 else "Muito Baixa"
            res["classification"] = cls
            sim_idx += 1
    
    return results, None