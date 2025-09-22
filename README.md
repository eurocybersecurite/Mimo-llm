# Mimo Language Model

Mimo est un modèle de langage fine-tuné sur DeepSeek-R1-Distill-Qwen-1.5B pour les tâches de code et de conversation, utilisant la technique LoRA.

![Mimo](assets/mimo.png)

## Installation des dépendances

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

Assurez-vous d'avoir `git-lfs` installé pour le téléchargement des modèles.

## Fine-tuning du modèle

Le script `fine_tune_mimo.py` permet de fine-tuner le modèle. Avant de l'exécuter, assurez-vous de définir votre token Hugging Face comme variable d'environnement :

```bash
export HF_TOKEN="votre_token_hugging_face"
```

Ensuite, lancez le script de fine-tuning :

```bash
python fine_tune_mimo.py
```

Ce script chargera le dataset `mohamed.jsonl` ainsi qu'un sous-ensemble du dataset public `mosaicml/instruct-v3` pour le fine-tuning. Le modèle fine-tuné et le tokenizer seront sauvegardés dans le répertoire `./Mimo`.

## Exemples de génération

### Génération de code

```python
# Exemple d'utilisation du modèle fine-tuné pour générer du code
# (Le code réel d'inférence dépendra de la manière dont le modèle est chargé après fine-tuning)

# Supposons que vous ayez chargé le modèle fine-tuné dans 'mimo_model' et 'mimo_tokenizer'
# from transformers import AutoModelForCausalLM, AutoTokenizer
# mimo_model = AutoModelForCausalLM.from_pretrained("./Mimo")
# mimo_tokenizer = AutoTokenizer.from_pretrained("./Mimo")

# prompt = "Écris une fonction Python pour trier une liste."
# inputs = mimo_tokenizer(prompt, return_tensors="pt")
# outputs = mimo_model.generate(**inputs, max_new_tokens=100)
# print(mimo_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Génération de conversation

```python
# Exemple d'utilisation du modèle fine-tuné pour une conversation
# (Le code réel d'inférence dépendra de la manière dont le modèle est chargé après fine-tuning)

# prompt = "Quelle est la meilleure façon d'apprendre une nouvelle langue ?"
# inputs = mimo_tokenizer(prompt, return_tensors="pt")
# outputs = mimo_model.generate(**inputs, max_new_tokens=150)
# print(mimo_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Structure du dépôt

```
Mimo/
├── README.md
├── assets/mimo.png
├── mohamed.jsonl
├── fine_tune_mimo.py
├── requirements.txt
└── .gitignore
