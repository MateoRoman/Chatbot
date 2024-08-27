import requests

def main():
    server_url = "http://localhost:5000/ask"

    print("Cliente listo para hacer preguntas. Escribe 'exit' para salir.")
    while True:
        question = input("Pregunta: ")
        if question.lower() == 'exit':
            break

        response = requests.post(server_url, json={"question": question})
        if response.status_code == 200:
            print(f"Respuesta: {response.json()['answer']}")
        else:
            print("Error al comunicarse con el servidor.")

if __name__ == "__main__":
    main()
