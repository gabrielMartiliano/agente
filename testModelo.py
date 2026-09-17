import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

print("=== TESTANDO O AGENTE (EXCLUSIVO PARA CPU) ===")

# ==========================================
# 1. AJUSTE DE CAMINHOS RELATIVOS
# ==========================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_script) in ["cpu", "amd", "nvidia"]:
    caminho_modelo_final = os.path.join(diretorio_script, "..", "models", "lora_model_saude")
else:
    caminho_modelo_final = os.path.join(diretorio_script, "models", "lora_model_saude")

modelo_base_id = "meta-llama/Llama-3.2-3B"

# ==========================================
# 2. CARREGAMENTO DOS MODELOS E OTIMIZAÇÃO
# ==========================================
print(f"Carregando tokenizer do {modelo_base_id}...")
tokenizer = AutoTokenizer.from_pretrained(modelo_base_id)

print("\nCarregando modelo base na memória RAM...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_base_id,
    dtype=torch.bfloat16, 
    device_map="cpu", # Força o uso estrito da CPU
    low_cpu_mem_usage=True
)

print(f"\nInjetando os pesos treinados (LoRA) de: {caminho_modelo_final}...")
model = PeftModel.from_pretrained(model, caminho_modelo_final)
print("Modelo pronto para uso!\n")

perguntas = [
    "### Pergunta: O que é saúde pública?\n### Resposta:",
    "### Pergunta: Qual o principal objetivo da etapa de triagem em um hospital?\n### Resposta:",
]

for i, prompt in enumerate(perguntas, 1):
    print(f"--- TESTE {i} ---")
   
    
    inputs = tokenizer(prompt, return_tensors="pt") # Mantém os tensores na RAM

    outputs = model.generate(
        **inputs,
        max_new_tokens=150, 
        do_sample=False,
        use_cache=True, 
        pad_token_id=tokenizer.eos_token_id
    )

    resposta_texto = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"\n{resposta_texto}\n")
    print("=" * 50 + "\n")