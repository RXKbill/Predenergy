import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DeepSeekDistillInference:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()  # 设置为评估模式

    def generate_pred_logits(self, prompt, max_length=100, temperature=1.0):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_pred_logits = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_pred_logits

if __name__ == "__main__":
    model_path = "models/deepseek_distill_r1_1.5B"

    inference_tool = DeepSeekDistillInference(model_path)
    
    # 输入提示
    prompt = "Once upon a time in a land far, far away,"
    
    # 生成文本
    generated_pred_logits = inference_tool.generate_pred_logits(prompt, max_length=200, temperature=0.7)
    
    # 打印生成的文本
    print("Generated Text:")
    print(generated_pred_logits)