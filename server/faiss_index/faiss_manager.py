import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

class FAISSManager:
    def __init__(self, vector_dim, index_path, data_path):
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.data_path = data_path
        self.index = faiss.IndexFlatL2(vector_dim)
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.load_data()

    def load_data(self):
        # Cargar datos de la base de conocimiento
        df = pd.read_excel(self.data_path)
        questions_df = df['Pregunta'].tolist()
        self.answers_df = df['Respuesta'].tolist()

        # Obtener embeddings para las preguntas en el CSV
        self.question_embeddings = self.embedding_model.encode(questions_df, convert_to_tensor=True)
        self.index.add(np.array(self.question_embeddings.tolist()))

    def search(self, query, k=5):
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        distances, indices = self.index.search(np.array(query_embedding.tolist()), k)
        return distances, indices

    def get_answer(self, query, k=5, threshold=0.6):
        distances, indices = self.search(query, k)
        most_similar_index = indices[0][0]  # Obtener el índice más cercano

        # Verificar si la similitud más alta es mayor que el umbral
        if distances[0][0] < threshold:
            return None
        
        # Asegúrate de devolver la respuesta completa
        return self.answers_df[most_similar_index].strip()
