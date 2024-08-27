from sentence_transformers import SentenceTransformer, util

# Cargar un modelo preentrenado de embeddings, como BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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
    
    general_embeddings = model.encode(general_university_questions, convert_to_tensor=True)
    question_embedding = model.encode([question], convert_to_tensor=True)
    
    # Calcular la similitud
    similarities = util.pytorch_cos_sim(question_embedding, general_embeddings)[0]

    # Imprimir similitudes para depuración
    print(f"Similitudes: {similarities.tolist()}")

    # Si alguna similitud es suficientemente alta, consideramos la pregunta relacionada
    if max(similarities) > 0.65:  # Ajustar el umbral según sea necesario
        return True
    
    return False
