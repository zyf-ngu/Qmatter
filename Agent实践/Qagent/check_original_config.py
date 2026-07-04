
import json

original_config_path = r"E:\llmmodels\allMiniLML6v2\sentence_bert_config.json"
with open(original_config_path, "r", encoding="utf-8") as f:
    original_config = json.load(f)
print("Original sentence_bert_config.json:", original_config)
