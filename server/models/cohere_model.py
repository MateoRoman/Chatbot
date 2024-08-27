import cohere

cohere_api_key = 'd9Q0jGmd5FO8wSYJ6Omb0R5dgUtmg8uX06wxIVQD'
cohere_client = cohere.Client(cohere_api_key)

def generate_answer_with_cohere(question, context=""):
    try:
        prompt = f"Provide a detailed and natural-sounding response to the question: '{question}'. Context: {context}"
        response = cohere_client.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=500,  # Aumentado a 500 para asegurar respuestas completas
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"Error al generar respuesta con Cohere: {e}")
        return "No se pudo generar una respuesta."
