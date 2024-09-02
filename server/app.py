import os
from flask import Flask, request, jsonify, render_template
from models.cohere_model import generate_answer_with_cohere
from models.bert_model import get_answer as get_bert_answer
from models.t5_model import generate_answer as generate_t5_answer
from faiss_index.faiss_manager import FAISSManager
from models.dialogpt_model import generate_response

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))

# Inicializa el administrador de FAISS
faiss_manager = FAISSManager(vector_dim=384,  # Ajusta esto al tamaño del vector de tu modelo
                             index_path='path_to_index_file',
                             data_path='data/Base_conocimiento_pre.xlsx')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_response():
    user_message = request.form.get('msg', '')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # 1. Verifica la base de conocimiento
        base_answer = faiss_manager.get_answer(user_message)

        if base_answer:
            # 2. Si la pregunta está en la base de conocimiento, usa BERT, T5 y Cohere para generar la respuesta
            answer = get_bert_answer(user_message) or generate_t5_answer(user_message) or generate_answer_with_cohere(user_message)
        else:
            # 3. Si la pregunta no está en la base de conocimiento, usa DialoGPT para generar la respuesta
            answer = generate_response(user_message)

        return jsonify({"answer": answer}), 200
    except Exception as e:
        app.logger.error(f"Error: {e}")  # Imprimir el error para depuración
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
