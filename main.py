from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("addy88/programming-lang-identifier")
model = AutoModelForSequenceClassification.from_pretrained("addy88/programming-lang-identifier")


code_to_identify = input("Enter the code snippet to identify the programming language: ")

input_ids = tokenizer.encode(code_to_identify, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = model(input_ids)[0]

language_idx = logits.argmax().item()

label_name = model.config.id2label[language_idx]

print("Predicted Language:", label_name)
