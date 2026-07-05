"""
MolSim Final — Backend Flask (v3.3)
• Input validation
• Improved error handling
• Type hints
"""

import logging
from typing import Tuple, Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from analysis import compare, bulk_compare, validate_fingerprint_type, validate_metric
from fpdf import FPDF
import io
import traceback
import base64
import tempfile

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_compare_request(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Valida requisição de comparação.
    
    Args:
        data: Dados da requisição JSON
        
    Returns:
        Tupla (válido, mensagem_erro)
    """
    required_fields = ['smiles_ref', 'smiles_test', 'metric']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Campo obrigatório ausente: {field}"
    
    fp_type = data.get('fp_type', 'Morgan2')
    if not validate_fingerprint_type(fp_type):
        return False, f"Tipo de fingerprint inválido: {fp_type}"
    
    metric = data.get('metric')
    if metric not in ['Tanimoto', 'Dice']:
        return False, f"Métrica inválida: {metric}"
    
    return True, ""


def validate_bulk_compare_request(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Valida requisição de bulk compare.
    
    Args:
        data: Dados da requisição JSON
        
    Returns:
        Tupla (válido, mensagem_erro)
    """
    required_fields = ['ref_smiles', 'smiles_list', 'metric']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Campo obrigatório ausente: {field}"
    
    if not isinstance(data['smiles_list'], list) or len(data['smiles_list']) == 0:
        return False, "smiles_list deve ser uma lista não vazia"
    
    fp_type = data.get('fp_type', 'Morgan2')
    if not validate_fingerprint_type(fp_type):
        return False, f"Tipo de fingerprint inválido: {fp_type}"
    
    metric = data.get('metric')
    if metric not in ['Tanimoto', 'Dice']:
        return False, f"Métrica inválida: {metric}"
    
    return True, ""


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Página principal."""
    logger.info("Página principal acessada")
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def api_compare():
    """
    Endpoint para comparação de duas moléculas.
    
    Esperado JSON:
    {
        "smiles_ref": "string",
        "smiles_test": "string",
        "name_ref": "string (opcional)",
        "name_test": "string (opcional)",
        "fp_type": "string (Morgan2, Morgan2 FCFP, RDKit, MACCS)",
        "metric": "string (Tanimoto, Dice)",
        "show_logd": "boolean (opcional)"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Requisição /compare sem JSON")
            return jsonify({"error": "Requisição JSON vazia"}), 400
        
        # Validar requisição
        valid, error_msg = validate_compare_request(data)
        if not valid:
            logger.warning(f"Validação falhou: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        # Valores padrão
        fp_type = data.get('fp_type', 'Morgan2')
        name_ref = data.get('name_ref', 'Molécula Referência')
        name_test = data.get('name_test', 'Molécula Teste')
        show_logd = data.get('show_logd', True)
        
        logger.info(f"Comparação iniciada: {fp_type} + {data['metric']}")
        
        result, error = compare(
            data['smiles_ref'],
            data['smiles_test'],
            name_ref,
            name_test,
            fp_type,
            data['metric'],
            show_map=show_logd
        )
        
        if error:
            logger.error(f"Erro na comparação: {error}")
            return jsonify({"error": error}), 400
        
        logger.info(f"Comparação concluída com sucesso")
        return jsonify(result), 200
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro não tratado em /compare: {error_trace}")
        return jsonify({"error": f"Erro interno do servidor: {str(e)[:100]}"}), 500


@app.route('/bulk-compare', methods=['POST'])
def api_bulk_compare():
    """
    Endpoint para comparação em lote.
    
    Esperado JSON:
    {
        "ref_smiles": "string",
        "smiles_list": ["string", ...],
        "names_list": ["string", ...] (opcional),
        "fp_type": "string (Morgan2, Morgan2 FCFP, RDKit, MACCS)",
        "metric": "string (Tanimoto, Dice)"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Requisição /bulk-compare sem JSON")
            return jsonify({"error": "Requisição JSON vazia"}), 400
        
        # Validar requisição
        valid, error_msg = validate_bulk_compare_request(data)
        if not valid:
            logger.warning(f"Validação falhou: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        fp_type = data.get('fp_type', 'Morgan2')
        
        logger.info(f"Bulk compare iniciado: {len(data['smiles_list'])} moléculas, {fp_type} + {data['metric']}")
        
        results, error = bulk_compare(
            data['ref_smiles'],
            data['smiles_list'],
            data.get('names_list'),
            fp_type,
            data['metric']
        )
        
        if error:
            logger.error(f"Erro no bulk compare: {error}")
            return jsonify({"error": error}), 400
        
        logger.info(f"Bulk compare concluído com sucesso")
        return jsonify({"results": results}), 200
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro não tratado em /bulk-compare: {error_trace}")
        return jsonify({"error": f"Erro interno do servidor: {str(e)[:100]}"}), 500


@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    """
    Endpoint para exportação em PDF.
    
    Esperado JSON com dados de resultado da comparação.
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Requisição /export-pdf sem JSON")
            return jsonify({"error": "Requisição JSON vazia"}), 400
        
        logger.info(f"Exportação PDF iniciada para {data.get('name_ref')} vs {data.get('name_test')}")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(190, 10, "Análise de Similaridade Molecular", ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(190, 8, f"Metodo: {data['fp_type']} | Metrica: {data['metric']}", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(190, 10, f"{data['name_ref']} vs {data['name_test']}", ln=True)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(190, 8, f"Similaridade: {data['similarity']:.4f} ({data['classification']})", ln=True)
        pdf.ln(8)
        
        # === Moléculas (PNG) ===
        y = pdf.get_y()
        if data.get('png_ref'):
            try:
                png_bytes = base64.b64decode(data['png_ref'])
                with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                    tmp.write(png_bytes)
                    tmp.flush()
                    pdf.image(tmp.name, x=10, y=y, w=85)
            except Exception as e:
                logger.warning(f"Erro ao adicionar PNG ref no PDF: {str(e)}")
        
        if data.get('png_test'):
            try:
                png_bytes = base64.b64decode(data['png_test'])
                with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                    tmp.write(png_bytes)
                    tmp.flush()
                    pdf.image(tmp.name, x=105, y=y, w=85)
            except Exception as e:
                logger.warning(f"Erro ao adicionar PNG test no PDF: {str(e)}")
        
        pdf.ln(90)
        
        # === Similarity Map (PNG) ===
        if data.get('similarity_map_png'):
            y = pdf.get_y()
            try:
                png_bytes = base64.b64decode(data['similarity_map_png'])
                with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                    tmp.write(png_bytes)
                    tmp.flush()
                    pdf.image(tmp.name, x=10, y=y, w=180)
                pdf.ln(195)
            except Exception as e:
                logger.warning(f"Erro ao adicionar similarity map PNG no PDF: {str(e)}")
        
        # Tabela de propriedades
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 8, "Propriedade", border=1, align='C')
        pdf.cell(40, 8, "Referência", border=1, align='C')
        pdf.cell(40, 8, "Teste", border=1, align='C')
        pdf.cell(40, 8, "Diferença", border=1, ln=True, align='C')
        
        pdf.set_font("Helvetica", "", 9)
        for prop in data.get('properties', []):
            pdf.cell(60, 8, str(prop['Propriedade'])[:20], border=1)
            pdf.cell(40, 8, str(prop['Referência']), border=1, align='C')
            pdf.cell(40, 8, str(prop['Teste']), border=1, align='C')
            pdf.cell(40, 8, str(prop['Diferença']), border=1, ln=True, align='C')
        
        pdf_bytes = pdf.output()
        
        logger.info("PDF gerado com sucesso")
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='molsim_resultado.pdf'
        )
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro ao gerar PDF: {error_trace}")
        return jsonify({"error": f"Erro ao gerar PDF: {str(e)[:100]}"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handler para rotas não encontradas."""
    logger.warning(f"Rota não encontrada: {request.path}")
    return jsonify({"error": "Rota não encontrada"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler para erros internos do servidor."""
    logger.error(f"Erro interno do servidor: {str(error)}")
    return jsonify({"error": "Erro interno do servidor"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("🚀 MolSim v3.3 iniciando em http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
