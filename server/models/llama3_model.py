from transformers import AutoTokenizer, AutoModelForCausalLM

# Autenticación en Hugging Face (asegúrate de haberlo hecho antes en otro lugar del código)
# from huggingface_hub import login
# login(token="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")

model_id = 'meta-llama/Meta-Llama-3.0'  # Cambia esto al modelo disponible
llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token="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")
llama_model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token="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")

def enrich_answer_with_llama(question, base_answer):
    input_text = (
        f"Respuesta base: {base_answer}\n\n"
        "Proporciona una respuesta detallada y coherente basada en la respuesta base anterior."
    )
    inputs = llama_tokenizer.encode(input_text, return_tensors="pt")
    outputs = llama_model.generate(
        inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True,
        temperature=0.7  # Ajusta esto si es necesario
    )
    generated_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return generated_response
