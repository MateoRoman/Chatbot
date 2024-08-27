import sys
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Añadir el directorio del servidor al path para resolver las importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Cargar el archivo Excel
df = pd.read_excel('data/Base_conocimiento_pre.xlsx')

# Extraer preguntas y respuestas
questions_df = df['Pregunta'].tolist()
answers_df = df['Respuesta'].tolist()

# Cargar el modelo de embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Obtener embeddings para las preguntas en el CSV
question_embeddings = model.encode(questions_df, convert_to_tensor=True)

def get_answer(question):
    # Obtener embedding para la pregunta ingresada
    question_embedding = model.encode([question], convert_to_tensor=True)

    # Calcular similitud con todas las preguntas en el CSV
    similarities = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]

    # Encontrar el índice de la pregunta más similar
    most_similar_index = similarities.argmax().item()

    # Devolver la respuesta correspondiente
    answer = answers_df[most_similar_index]
    
    # Asegúrate de que la respuesta completa se devuelva
    return answer.strip()