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
from analysis import compare, bulk_compare, validate_fingerprint_type
from fpdf import FPDF
import io
import traceback
import base64
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def validate_compare_request(data: Dict[str, Any]) -> Tuple[bool, str]:
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def api_compare():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requisição JSON vazia"}), 400

        valid, error_msg = validate_compare_request(data)
        if not valid:
            return jsonify({"error": error_msg}), 400

        fp_type = data.get('fp_type', 'Morgan2')
        result, error = compare(
            data['smiles_ref'],
            data['smiles_test'],
            data.get('name_ref', 'Molécula Referência'),
            data.get('name_test', 'Molécula Teste'),
            fp_type,
            data['metric'],
            show_map=data.get('show_logd', True)
        )

        if error:
            return jsonify({"error": error}), 400

        return jsonify(result), 200
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro não tratado em /compare: {error_trace}")
        return jsonify({"error": f"Erro interno do servidor: {str(e)[:100]}"}), 500


@app.route('/bulk-compare', methods=['POST'])
def api_bulk_compare():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requisição JSON vazia"}), 400

        valid, error_msg = validate_bulk_compare_request(data)
        if not valid:
            return jsonify({"error": error_msg}), 400

        results, error = bulk_compare(
            data['ref_smiles'],
            data['smiles_list'],
            data.get('names_list'),
            data.get('fp_type', 'Morgan2'),
            data['metric']
        )

        if error:
            return jsonify({"error": error}), 400

        return jsonify({"results": results}), 200
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro não tratado em /bulk-compare: {error_trace}")
        return jsonify({"error": f"Erro interno do servidor: {str(e)[:100]}"}), 500


@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requisição JSON vazia"}), 400

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
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

        # Página/Bloco 1: estruturas
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

        pdf.ln(95)

        # Página/Bloco 2: fingerprints em página separada para evitar corte
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(190, 8, "Estruturas e Fingerprints", ln=True)
        pdf.ln(5)

        if data.get('fingerprint_ref_png') or data.get('fingerprint_test_png'):
            if data.get('fingerprint_ref_png'):
                try:
                    fp_bytes = base64.b64decode(data['fingerprint_ref_png'])
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                        tmp.write(fp_bytes)
                        tmp.flush()
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.cell(90, 8, f"Fingerprint: {data['name_ref']}", ln=0)
                        pdf.ln(10)
                        pdf.image(tmp.name, x=15, y=pdf.get_y(), w=75)
                except Exception as e:
                    logger.warning(f"Erro ao adicionar fingerprint ref no PDF: {str(e)}")

            if data.get('fingerprint_test_png'):
                try:
                    fp_bytes = base64.b64decode(data['fingerprint_test_png'])
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                        tmp.write(fp_bytes)
                        tmp.flush()
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.cell(90, 8, f"Fingerprint: {data['name_test']}", ln=0)
                        pdf.ln(10)
                        pdf.image(tmp.name, x=110, y=pdf.get_y(), w=75)
                except Exception as e:
                    logger.warning(f"Erro ao adicionar fingerprint test no PDF: {str(e)}")

        # Página/Bloco 3: propriedades em página própria para não cortar
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(190, 8, "Propriedades Físico-Químicas", ln=True)
        pdf.ln(4)

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

        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(190, 6, f"Gerado em {data.get('generated_at', 'data não informada')}", ln=True, align='C')

        pdf_bytes = pdf.output()
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
    return jsonify({"error": "Rota não encontrada"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erro interno do servidor"}), 500


if __name__ == '__main__':
    logger.info("🚀 MolSim v3.3 iniciando em http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
