# Proyecto Chatbot

Este proyecto consiste en el desarrollo de un chatbot que utiliza un sistema de Recuperación y Generación (RAG) para proporcionar respuestas coherentes basadas en una base de conocimiento específica y un modelo de lenguaje preentrenado.

## Integrantes

- **David Cantuña**
- **Matías Padrón**
- **Mateo Román**

## Descripción

El chatbot está diseñado para responder preguntas utilizando una combinación de técnicas de recuperación de información y generación de texto. La arquitectura del chatbot integra modelos como BERT, T5, y Cohere para manejar preguntas dentro de la base de conocimiento y DialoGPT para generar respuestas cuando la información no está disponible en la base.

## Características Principales

- **Recuperación de información:** Responde preguntas simples utilizando una base de conocimiento predefinida.
- **Generación de texto:** Utiliza DialoGPT para generar respuestas coherentes a preguntas no cubiertas en la base de conocimiento.
- **Arquitectura modular:** Fácil de adaptar y expandir con nuevos modelos o bases de conocimiento.

## Requisitos

- Python 3.x
- Bibliotecas necesarias listadas en `requirements.txt`

## Instrucciones de Uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/proyecto-chatbot.git
