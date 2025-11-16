from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-V3.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # For efficiency
    device_map="auto",  # Auto-GPU/CPU
    load_in_8bit=True  # Quantized to fit ~16GB RAM
)

def generate_report(anomaly, risk, forecast):
    prompt = f"""
    You are a Military Medical AI. Generate a structured JSON report:
    Brain Anomaly: {anomaly}%
    Risk Level: {risk}
    Forecast: {forecast}

    Output only valid JSON:
    {{"patient_id": "TBI-001", "conclusion": "Urgent evacuation recommended"}}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Output only valid JSON:")[-1].strip()  # Extract JSON