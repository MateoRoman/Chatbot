import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Inicializa el modelo DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Configura el padding a la izquierda
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

def clean_response(response, user_message):
    # Función para limpiar la respuesta eliminando fragmentos de la pregunta
    response_lower = response.lower()
    user_message_lower = user_message.lower()
    
    # Eliminamos cualquier coincidencia con la pregunta del usuario
    if response_lower.startswith(user_message_lower):
        response = response[len(user_message):].strip()
    
    return response.strip()

def generate_answer_with_dialogpt(user_message):
    # Tokeniza el mensaje del usuario
    new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
    
    # Genera la respuesta con DialoGPT
    chat_history_ids = model.generate(
        new_user_input_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=200,   # Limita la longitud de la respuesta
        temperature=0.7,     # Ajusta la creatividad de las respuestas
        top_p=0.9,           # Ajusta la diversidad de las respuestas
        top_k=50             # Limita el número de palabras a considerar
    )
    
    # Decodifica la respuesta generada
    generated_message = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    
    # Limpiar la respuesta para eliminar repeticiones de la pregunta
    cleaned_message = clean_response(generated_message, user_message)
    
    # Si la respuesta está vacía o es muy similar a la pregunta, devuelves una respuesta predeterminada
    if cleaned_message == "":
        cleaned_message = "Lo siento, no tengo una respuesta adecuada para eso."
    
    return cleaned_message

def generate_response(user_message):
    # Utiliza el modelo DialoGPT para generar una respuesta coherente sin usar historial
    return generate_answer_with_dialogpt(user_message)
