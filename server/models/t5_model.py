import sys
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_excel('data/Base_conocimiento_pre.xlsx')

questions_df = df['Pregunta'].tolist()
answers_df = df['Respuesta'].tolist()

# Inicializar el modelo T5 y el tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')  # Cambiado a 't5-base' para más capacidad
model = T5ForConditionalGeneration.from_pretrained('t5-base')
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
question_embeddings = embedding_model.encode(questions_df, convert_to_tensor=True)

# Definir un conjunto de palabras clave relacionadas con la universidad
university_keywords = [
    "universidad", "biblioteca", "curso", "profesor", "alumno", 
    "matrícula", "examen", "calificación", "estudiante", "aula", 
    "campus", "carrera", "inscripción", "clase", "asignatura"
]

def is_university_related(question):
    # Comprobar si alguna palabra clave está en la pregunta
    for keyword in university_keywords:
        if keyword.lower() in question.lower():
            return True

    # Si no se encuentran palabras clave, usar un modelo de embeddings para comparar la pregunta con conceptos generales de la universidad
    general_university_questions = [
        "¿Qué es la universidad?",
        "¿Cómo funciona la universidad?",
        "¿Qué servicios ofrece la universidad?",
        "¿Cuáles son las normas en la universidad?",
        "¿Cómo puedo matricularme en la universidad?",
        "¿Cómo puedo acceder a la biblioteca?",
        "¿Cuáles son las reglas en el campus?",
        "¿Qué carreras se pueden estudiar en la universidad?"
    ]
    
    general_embeddings = embedding_model.encode(general_university_questions, convert_to_tensor=True)
    question_embedding = embedding_model.encode([question], convert_to_tensor=True)
    
    # Calcular la similitud
    similarities = util.pytorch_cos_sim(question_embedding, general_embeddings)[0]

    # Imprimir similitudes para depuración
    print(f"Similitudes: {similarities.tolist()}")

    # Si alguna similitud es suficientemente alta, consideramos la pregunta relacionada
    if max(similarities) > 0.65:  # Ajustar el umbral según sea necesario
        return True
    
    return False

def generate_answer(question):
    if not is_university_related(question):
        return "Lo siento, solo respondo preguntas sobre la universidad."
    
    input_text = "question: " + question
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=500,  # Aumentado a 500 para respuestas más largas
        num_beams=5, 
        early_stopping=True,
        no_repeat_ngram_size=3  # Aumentar para evitar repetición
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def find_closest_answer(question):
    if not is_university_related(question):
        return "Lo siento, solo respondo preguntas sobre la universidad."

    question_embedding = embedding_model.encode([question], convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]
    most_similar_index = similarities.argmax().item()
    return answers_df[most_similar_index]

def enrich_answer(question, base_answer):
    if not is_university_related(question):
        return "Lo siento, solo respondo preguntas sobre la universidad."
    
    input_text = f"Context: {question}\n\nRespuesta: {base_answer}\n\n"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=500,  # Asegura suficiente longitud
        num_beams=5,
        early_stopping=True,
        temperature=0.7,
        no_repeat_ngram_size=3  # Evitar repetición de n-gramas
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
