from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained("./models/bitnet")
tokenizer.save_pretrained("./models/bitnet")
