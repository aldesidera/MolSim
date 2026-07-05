"""
Testes unitários para o módulo analysis.py
"""

import pytest
from analysis import (
    get_mol,
    get_properties,
    get_fingerprint,
    calc_similarity,
    classify_similarity,
    validate_fingerprint_type,
    validate_metric,
    compare,
    bulk_compare,
    calc_logd_vs_ph,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def aspirina_smiles():
    """SMILES da Aspirina (ácido acetilsalicílico)."""
    return "CC(=O)OC1=CC=CC=C1C(=O)O"


@pytest.fixture
def paracetamol_smiles():
    """SMILES do Paracetamol."""
    return "CC(=O)NC1=CC=C(O)C=C1"


@pytest.fixture
def ibuprofeno_smiles():
    """SMILES do Ibuprofeno."""
    return "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture
def invalid_smiles():
    """SMILES inválido para testes."""
    return "INVALID_SMILES_XYZ"


@pytest.fixture
def aspirina_mol(aspirina_smiles):
    """Molécula carregada de Aspirina."""
    mol, _ = get_mol(aspirina_smiles)
    return mol


# ============================================================================
# TESTS: VALIDATION FUNCTIONS
# ============================================================================

class TestValidationFunctions:
    """Testes para funções de validação."""
    
    def test_validate_fingerprint_type_valid(self):
        """Testa validação de tipos de fingerprint válidos."""
        assert validate_fingerprint_type("Morgan2") is True
        assert validate_fingerprint_type("Morgan2 FCFP") is True
        assert validate_fingerprint_type("RDKit") is True
        assert validate_fingerprint_type("MACCS") is True
    
    def test_validate_fingerprint_type_invalid(self):
        """Testa validação de tipos de fingerprint inválidos."""
        assert validate_fingerprint_type("Invalid") is False
        assert validate_fingerprint_type("") is False
        assert validate_fingerprint_type("morgan2") is False  # case-sensitive
    
    def test_validate_metric_valid(self):
        """Testa validação de métricas válidas."""
        assert validate_metric("Tanimoto") is True
        assert validate_metric("Dice") is True
    
    def test_validate_metric_invalid(self):
        """Testa validação de métricas inválidas."""
        assert validate_metric("Invalid") is False
        assert validate_metric("") is False
        assert validate_metric("tanimoto") is False  # case-sensitive


# ============================================================================
# TESTS: MOLECULAR FUNCTIONS
# ============================================================================

class TestMolecularFunctions:
    """Testes para funções moleculares."""
    
    def test_get_mol_valid_smiles(self, aspirina_smiles):
        """Testa carregamento de SMILES válido."""
        mol, err = get_mol(aspirina_smiles)
        assert mol is not None
        assert err is None
    
    def test_get_mol_invalid_smiles(self, invalid_smiles):
        """Testa tratamento de SMILES inválido."""
        mol, err = get_mol(invalid_smiles)
        assert mol is None
        assert err is not None
        assert "SMILES inválido" in err or "Erro ao processar SMILES" in err
    
    def test_get_mol_empty_string(self):
        """Testa tratamento de string vazia."""
        mol, err = get_mol("")
        assert mol is None
        assert err is not None
    
    def test_get_mol_whitespace(self):
        """Testa tratamento de SMILES com espaços."""
        mol, err = get_mol("  CC(=O)OC1=CC=CC=C1C(=O)O  ")
        assert mol is not None
        assert err is None
    
    def test_get_properties_valid_mol(self, aspirina_mol):
        """Testa cálculo de propriedades de molécula válida."""
        props = get_properties(aspirina_mol)
        assert props is not None
        assert "Peso Molecular (g/mol)" in props
        assert "Coeficiente de Partição (LogP)" in props
        assert "Área de Superfície Polar (Å²)" in props
        assert "Doadores de H (HBD)" in props
        assert "Receptores de H (HBA)" in props
        assert "Ligações Rotacionais" in props
        
        # Validar tipos e ranges
        assert isinstance(props["Peso Molecular (g/mol)"], float)
        assert props["Peso Molecular (g/mol)"] > 0
        assert isinstance(props["Doadores de H (HBD)"], int)
        assert props["Doadores de H (HBD)"] >= 0
    
    def test_get_properties_none_mol(self):
        """Testa cálculo de propriedades com mol=None."""
        props = get_properties(None)
        assert props is None
    
    def test_get_fingerprint_morgan2(self, aspirina_mol):
        """Testa geração de fingerprint Morgan2."""
        fp = get_fingerprint(aspirina_mol, "Morgan2")
        assert fp is not None
        assert len(fp) == 2048
    
    def test_get_fingerprint_morgan2_fcfp(self, aspirina_mol):
        """Testa geração de fingerprint Morgan2 FCFP."""
        fp = get_fingerprint(aspirina_mol, "Morgan2 FCFP")
        assert fp is not None
        assert len(fp) == 2048
    
    def test_get_fingerprint_rdkit(self, aspirina_mol):
        """Testa geração de fingerprint RDKit."""
        fp = get_fingerprint(aspirina_mol, "RDKit")
        assert fp is not None
    
    def test_get_fingerprint_maccs(self, aspirina_mol):
        """Testa geração de fingerprint MACCS."""
        fp = get_fingerprint(aspirina_mol, "MACCS")
        assert fp is not None
        assert len(fp) == 166
    
    def test_get_fingerprint_invalid_type(self, aspirina_mol):
        """Testa geração de fingerprint com tipo inválido."""
        fp = get_fingerprint(aspirina_mol, "Invalid")
        assert fp is None
    
    def test_get_fingerprint_none_mol(self):
        """Testa geração de fingerprint com mol=None."""
        fp = get_fingerprint(None, "Morgan2")
        assert fp is None


# ============================================================================
# TESTS: SIMILARITY FUNCTIONS
# ============================================================================

class TestSimilarityFunctions:
    """Testes para funções de similaridade."""
    
    def test_calc_similarity_tanimoto(self, aspirina_smiles, paracetamol_smiles):
        """Testa cálculo de similaridade Tanimoto."""
        mol1, _ = get_mol(aspirina_smiles)
        mol2, _ = get_mol(paracetamol_smiles)
        
        fp1 = get_fingerprint(mol1, "Morgan2")
        fp2 = get_fingerprint(mol2, "Morgan2")
        
        sim = calc_similarity(fp1, fp2, "Tanimoto")
        assert 0.0 <= sim <= 1.0
        assert isinstance(sim, float)
    
    def test_calc_similarity_dice(self, aspirina_smiles, paracetamol_smiles):
        """Testa cálculo de similaridade Dice."""
        mol1, _ = get_mol(aspirina_smiles)
        mol2, _ = get_mol(paracetamol_smiles)
        
        fp1 = get_fingerprint(mol1, "Morgan2")
        fp2 = get_fingerprint(mol2, "Morgan2")
        
        sim = calc_similarity(fp1, fp2, "Dice")
        assert 0.0 <= sim <= 1.0
        assert isinstance(sim, float)
    
    def test_calc_similarity_identical(self, aspirina_smiles):
        """Testa similaridade de molécula consigo mesma."""
        mol, _ = get_mol(aspirina_smiles)
        fp = get_fingerprint(mol, "Morgan2")
        
        sim = calc_similarity(fp, fp, "Tanimoto")
        assert sim == 1.0
    
    def test_classify_similarity_tanimoto_muito_alta(self):
        """Testa classificação 'Muito Alta' para Tanimoto."""
        classification = classify_similarity(0.85, "Tanimoto")
        assert classification == "Muito Alta"
    
    def test_classify_similarity_tanimoto_alta(self):
        """Testa classificação 'Alta' para Tanimoto."""
        classification = classify_similarity(0.65, "Tanimoto")
        assert classification == "Alta"
    
    def test_classify_similarity_tanimoto_moderada(self):
        """Testa classificação 'Moderada' para Tanimoto."""
        classification = classify_similarity(0.40, "Tanimoto")
        assert classification == "Moderada"
    
    def test_classify_similarity_tanimoto_baixa(self):
        """Testa classificação 'Baixa' para Tanimoto."""
        classification = classify_similarity(0.20, "Tanimoto")
        assert classification == "Baixa"
    
    def test_classify_similarity_tanimoto_muito_baixa(self):
        """Testa classificação 'Muito Baixa' para Tanimoto."""
        classification = classify_similarity(0.10, "Tanimoto")
        assert classification == "Muito Baixa"
    
    def test_classify_similarity_dice_muito_alta(self):
        """Testa classificação 'Muito Alta' para Dice."""
        classification = classify_similarity(0.90, "Dice")
        assert classification == "Muito Alta"
    
    def test_classify_similarity_dice_muito_baixa(self):
        """Testa classificação 'Muito Baixa' para Dice."""
        classification = classify_similarity(0.10, "Dice")
        assert classification == "Muito Baixa"


# ============================================================================
# TESTS: CALCULATION FUNCTIONS
# ============================================================================

class TestCalculationFunctions:
    """Testes para funções de cálculo."""
    
    def test_calc_logd_vs_ph_valid_mol(self, aspirina_mol):
        """Testa cálculo de LogD vs pH."""
        logd_data = calc_logd_vs_ph(aspirina_mol)
        assert logd_data is not None
        assert len(logd_data) == 15  # pH 0-14
        
        # Validar estrutura
        for data_point in logd_data:
            assert "pH" in data_point
            assert "LogD" in data_point
            assert isinstance(data_point["pH"], int)
            assert isinstance(data_point["LogD"], float)
    
    def test_calc_logd_vs_ph_custom_range(self, aspirina_mol):
        """Testa cálculo de LogD com range customizado."""
        custom_range = [5, 7, 9]
        logd_data = calc_logd_vs_ph(aspirina_mol, ph_range=custom_range)
        assert logd_data is not None
        assert len(logd_data) == 3
    
    def test_calc_logd_vs_ph_none_mol(self):
        """Testa cálculo de LogD com mol=None."""
        logd_data = calc_logd_vs_ph(None)
        assert logd_data is None


# ============================================================================
# TESTS: COMPARISON FUNCTIONS
# ============================================================================

class TestComparisonFunctions:
    """Testes para funções de comparação."""
    
    def test_compare_valid_inputs(self, aspirina_smiles, paracetamol_smiles):
        """Testa comparação com inputs válidos."""
        result, error = compare(
            aspirina_smiles,
            paracetamol_smiles,
            "Aspirina",
            "Paracetamol",
            "Morgan2",
            "Tanimoto"
        )
        assert error is None
        assert result is not None
        assert result["similarity"] is not None
        assert result["classification"] is not None
        assert result["classification"] in ["Muito Alta", "Alta", "Moderada", "Baixa", "Muito Baixa"]
    
    def test_compare_invalid_smiles_ref(self, paracetamol_smiles, invalid_smiles):
        """Testa comparação com SMILES referência inválido."""
        result, error = compare(
            invalid_smiles,
            paracetamol_smiles,
            "Invalid",
            "Paracetamol",
            "Morgan2",
            "Tanimoto"
        )
        assert result is None
        assert error is not None
    
    def test_compare_invalid_smiles_test(self, aspirina_smiles, invalid_smiles):
        """Testa comparação com SMILES teste inválido."""
        result, error = compare(
            aspirina_smiles,
            invalid_smiles,
            "Aspirina",
            "Invalid",
            "Morgan2",
            "Tanimoto"
        )
        assert result is None
        assert error is not None
    
    def test_compare_invalid_fingerprint_type(self, aspirina_smiles, paracetamol_smiles):
        """Testa comparação com tipo de fingerprint inválido."""
        result, error = compare(
            aspirina_smiles,
            paracetamol_smiles,
            "Aspirina",
            "Paracetamol",
            "Invalid",
            "Tanimoto"
        )
        assert result is None
        assert error is not None
    
    def test_compare_all_fingerprint_types(self, aspirina_smiles, paracetamol_smiles):
        """Testa comparação com todos os tipos de fingerprint."""
        for fp_type in ["Morgan2", "Morgan2 FCFP", "RDKit", "MACCS"]:
            result, error = compare(
                aspirina_smiles,
                paracetamol_smiles,
                "Aspirina",
                "Paracetamol",
                fp_type,
                "Tanimoto"
            )
            assert error is None, f"Erro com {fp_type}: {error}"
            assert result is not None
            assert 0.0 <= result["similarity"] <= 1.0
    
    def test_bulk_compare_valid_inputs(self, aspirina_smiles, paracetamol_smiles, ibuprofeno_smiles):
        """Testa bulk compare com inputs válidos."""
        results, error = bulk_compare(
            aspirina_smiles,
            [paracetamol_smiles, ibuprofeno_smiles],
            ["Paracetamol", "Ibuprofeno"],
            "Morgan2",
            "Tanimoto"
        )
        assert error is None
        assert results is not None
        assert len(results) == 2
        
        for result in results:
            assert result["name"] in ["Paracetamol", "Ibuprofeno"]
            assert result["similarity"] is not None
            assert 0.0 <= result["similarity"] <= 1.0
    
    def test_bulk_compare_invalid_reference(self, invalid_smiles, paracetamol_smiles):
        """Testa bulk compare com SMILES referência inválido."""
        results, error = bulk_compare(
            invalid_smiles,
            [paracetamol_smiles],
            ["Paracetamol"],
            "Morgan2",
            "Tanimoto"
        )
        assert error is not None
        assert results is None
    
    def test_bulk_compare_mixed_valid_invalid(self, aspirina_smiles, paracetamol_smiles, invalid_smiles):
        """Testa bulk compare com mistura de SMILES válidos e inválidos."""
        results, error = bulk_compare(
            aspirina_smiles,
            [paracetamol_smiles, invalid_smiles],
            ["Paracetamol", "Invalid"],
            "Morgan2",
            "Tanimoto"
        )
        assert error is None
        assert results is not None
        assert len(results) == 2
        
        # Primeiro resultado válido
        assert results[0]["similarity"] is not None
        # Segundo resultado inválido
        assert results[1]["error"] is not None


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Testes para casos extremos."""
    
    def test_very_small_molecule(self):
        """Testa com molécula muito pequena (metano)."""
        mol, err = get_mol("C")
        assert mol is not None
        assert err is None
        
        props = get_properties(mol)
        assert props is not None
        assert props["Peso Molecular (g/mol)"] < 20
    
    def test_large_molecule(self):
        """Testa com molécula grande."""
        # Testosterona
        large_mol_smiles = "CC(C)CCCC(C)(C)C1CCC2C1(CCCC2=CC(=O)C)C"
        mol, err = get_mol(large_mol_smiles)
        assert mol is not None
        assert err is None
        
        props = get_properties(mol)
        assert props is not None
        assert props["Peso Molecular (g/mol)"] > 200
    
    def test_molecule_with_stereochemistry(self):
        """Testa molécula com estereoquímica."""
        # D-Glucose
        glucose_smiles = "C([C@@H]1[C@H]([C@@H]([C@H](C(=O)O1)O)O)O)O"
        mol, err = get_mol(glucose_smiles)
        assert mol is not None
        assert err is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
