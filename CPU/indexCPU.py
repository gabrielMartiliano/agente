# ==========================================
# Treinamento de LLaMA-3.2-3B com LoRA (Modo CPU / Windows)
# ========================================== 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

# ==========================================
# 0. AJUSTE DE CAMINHOS RELATIVOS (Para subpastas)
# ==========================================

diretorio_script = os.path.dirname(os.path.abspath(__file__))

caminho_dataset = os.path.join(diretorio_script, "..", "data", "dataset_mestre.jsonl")
caminho_checkpoints = os.path.join(diretorio_script, "..", "training", "checkpoints_salvos")
caminho_modelo_final = os.path.join(diretorio_script, "..", "models", "lora_model_saude")

print("=== INICIANDO AMBIENTE LOCAL (CPU) ===")
print(f"Buscando dataset em: {caminho_dataset}")

# ==========================================
# 1. CARREGAMENTO DO MODELO E TOKENIZER
# ==========================================
modelo_id = "meta-llama/Llama-3.2-3B"

print(f"\nCarregando tokenizer do {modelo_id}...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nCarregando modelo base na memória (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# ==========================================
# 2. CONFIGURAÇÃO DO LORA
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ==========================================
# 3. DATASET E TREINAMENTO
# ==========================================
dataset = load_dataset("json", data_files=caminho_dataset, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=10, 
        learning_rate=2e-4,
        logging_steps=1,
        output_dir=caminho_checkpoints,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        report_to="none" 
    ),
)

print("\nIniciando o treinamento na CPU...")
trainer.train()

# ==========================================
# 4. SALVANDO OS PESOS FINAIS
# ==========================================
trainer.model.save_pretrained(caminho_modelo_final)
print(f"\nTreinamento concluído e salvo com sucesso em: {caminho_modelo_final}")