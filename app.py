"""
MolSim Final — Backend Flask (v3.2)
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from analysis import compare, bulk_compare
from fpdf import FPDF
import io
import traceback
import base64
import tempfile

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def api_compare():
    """Compara duas moléculas."""
    try:
        data = request.get_json()
        
        valid_fp_types = ['Morgan2', 'Morgan2 FCFP', 'RDKit', 'MACCS']
        fp_type = data.get('fp_type', 'Morgan2')
        if fp_type not in valid_fp_types:
            fp_type = 'Morgan2'
        
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
        print(f"Erro: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/bulk-compare', methods=['POST'])
def api_bulk_compare():
    """Comparação em lote (Bulk Similarity)."""
    try:
        data = request.get_json()
        valid_fp_types = ['Morgan2', 'Morgan2 FCFP', 'RDKit', 'MACCS']
        fp_type = data.get('fp_type', 'Morgan2')
        if fp_type not in valid_fp_types:
            fp_type = 'Morgan2'
        
        results, error = bulk_compare(
            data['ref_smiles'],
            data['smiles_list'],
            data.get('names_list'),
            fp_type,
            data['metric']
        )
        if error:
            return jsonify({"error": error}), 400
        return jsonify({"results": results}), 200
    except Exception as e:
        print(f"Erro bulk: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    """Exporta resultado em PDF com imagens PNG em memória."""
    try:
        data = request.get_json()
        
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
            png_bytes = base64.b64decode(data['png_ref'])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                tmp.write(png_bytes)
                tmp.flush()
                pdf.image(tmp.name, x=10, y=y, w=85)
        if data.get('png_test'):
            png_bytes = base64.b64decode(data['png_test'])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                tmp.write(png_bytes)
                tmp.flush()
                pdf.image(tmp.name, x=105, y=y, w=85)
        pdf.ln(90)
        
        # === Similarity Map (PNG) ===
        if data.get('similarity_map_png'):
            y = pdf.get_y()
            png_bytes = base64.b64decode(data['similarity_map_png'])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                tmp.write(png_bytes)
                tmp.flush()
                pdf.image(tmp.name, x=10, y=y, w=180)
            pdf.ln(195)
        
        # Tabela de propriedades
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 8, "Propriedade", border=1, align='C')
        pdf.cell(40, 8, "Referência", border=1, align='C')
        pdf.cell(40, 8, "Teste", border=1, align='C')
        pdf.cell(40, 8, "Diferença", border=1, ln=True, align='C')
        
        pdf.set_font("Helvetica", "", 9)
        for prop in data['properties']:
            pdf.cell(60, 8, str(prop['Propriedade'])[:20], border=1)
            pdf.cell(40, 8, str(prop['Referência']), border=1, align='C')
            pdf.cell(40, 8, str(prop['Teste']), border=1, align='C')
            pdf.cell(40, 8, str(prop['Diferença']), border=1, ln=True, align='C')
        
        pdf_bytes = pdf.output()
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='molsim_resultado.pdf'
        )
    
    except Exception as e:
        print(f"Erro ao gerar PDF: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 MolSim v3.2 iniciando em http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)