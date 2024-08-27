import os
from flask import Flask, request, jsonify, render_template
from models.cohere_model import generate_answer_with_cohere
from models.bert_model import get_answer as get_bert_answer
from models.t5_model import generate_answer as generate_t5_answer
from faiss_index.faiss_manager import FAISSManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Inicializa el modelo DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Configura el padding a la izquierda
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))

# Inicializa el administrador de FAISS
faiss_manager = FAISSManager(vector_dim=384,  # Ajusta esto al tamaño del vector de tu modelo
                             index_path='path_to_index_file',
                             data_path='data/Base_conocimiento_pre.xlsx')

# Variable para almacenar el historial de chat
chat_history_ids = torch.tensor([], dtype=torch.long)  # Inicializar como tensor vacío

def is_valid_response(response, context):
    """
    Verifica si la respuesta generada es válida en función del contexto.
    """
    context_keywords = context.lower().split()
    response_keywords = response.lower().split()
    return any(word in response_keywords for word in context_keywords)

def is_university_related(question):
    """
    Verifica si la pregunta está relacionada con la universidad.
    """
    university_keywords = [
        "universidad", "biblioteca", "curso", "profesor", "alumno", 
        "matrícula", "examen", "calificación", "estudiante", "aula", 
        "campus", "carrera", "inscripción", "clase", "asignatura"
    ]
    return any(keyword.lower() in question.lower() for keyword in university_keywords)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_response():
    global chat_history_ids
    
    user_message = request.form.get('msg', '')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Verifica la base de conocimiento primero
        base_answer = faiss_manager.get_answer(user_message)

        if not base_answer or base_answer.strip() == "":
            base_answer = get_bert_answer(user_message)
        
        if not base_answer or base_answer.strip() == "":
            base_answer = generate_t5_answer(user_message)
        
        if not base_answer or base_answer.strip() == "":
            base_answer = generate_answer_with_cohere(user_message)

        # Si no se encontró respuesta en la base de conocimiento, usar DialoGPT
        if not base_answer or base_answer.strip() == "":
            new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
            
            if chat_history_ids.numel() > 0:
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids
            
            chat_history_ids = model.generate(
                bot_input_ids,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1000
            )
            
            base_answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            # Verificar la validez de la respuesta generada por DialoGPT
            if not is_valid_response(base_answer, user_message):
                base_answer = "Lo siento, no tengo suficiente información para responder a tu pregunta. Por favor, proporciona más contexto o intenta con otra consulta."
            else:
                # Añadir al historial si tiene sentido y contexto
                bot_input_ids = tokenizer.encode(base_answer + tokenizer.eos_token, return_tensors='pt')
                chat_history_ids = torch.cat([chat_history_ids, bot_input_ids], dim=-1) if chat_history_ids.numel() > 0 else bot_input_ids
        else:
            # Añadir al historial solo si es una respuesta válida
            if "Lo siento, no tengo suficiente información" not in base_answer:
                bot_input_ids = tokenizer.encode(base_answer + tokenizer.eos_token, return_tensors='pt')
                chat_history_ids = torch.cat([chat_history_ids, bot_input_ids], dim=-1) if chat_history_ids.numel() > 0 else bot_input_ids

        # Si no hay respuesta adecuada, devuelve el mensaje de no información
        if not base_answer or "Lo siento, no tengo suficiente información" in base_answer:
            base_answer = "Lo siento, no tengo suficiente información para responder a tu pregunta. Por favor, proporciona más contexto o intenta con otra consulta."

        return jsonify({"answer": base_answer}), 200
    except Exception as e:
        app.logger.error(f"Error: {e}")  # Imprimir el error para depuración
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
