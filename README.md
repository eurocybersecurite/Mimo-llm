# Mimo Language Model

Mimo est un modèle de langage autonome et puissant, conçu pour exceller dans les tâches de code et de conversation, surpassant les modèles conventionnels grâce à son architecture avancée et son fine-tuning spécialisé.

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

```

## Intégration et Utilisation de Mimo

### Sur Mac et PC (via VSCode ou terminal)

Pour utiliser Mimo, vous pouvez soit exécuter le script de fine-tuning, soit utiliser le modèle fine-tuné pour l'inférence.

#### 1. Exécution du script de fine-tuning (`fine_tune_mimo.py`)

Assurez-vous d'avoir installé les dépendances (`pip install -r requirements.txt`) et configuré votre `HF_TOKEN`. Ensuite, exécutez le script :

```bash
python fine_tune_mimo.py
```

Ce script entraînera le modèle et sauvegardera les résultats dans le dossier `./Mimo`.

#### 2. Utilisation du modèle fine-tuné pour l'inférence

Une fois le fine-tuning terminé (ou si vous utilisez un modèle pré-entraîné fine-tuné), vous pouvez charger le modèle et le tokenizer pour générer du texte.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel # Nécessaire si vous avez sauvegardé des adaptateurs LoRA séparément

# Chemin vers le modèle fine-tuné (ou le répertoire contenant les adaptateurs LoRA)
model_dir = "./Mimo" 

# Configuration de la quantification pour charger le modèle efficacement (si utilisé lors du fine-tuning)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Charger le modèle de base (si vous avez fine-tuné avec LoRA et sauvegardé les adaptateurs)
# Si vous avez sauvegardé le modèle complet (avec adaptateurs fusionnés), chargez-le directement.
# Pour cet exemple, nous supposons que vous avez sauvegardé le modèle complet dans model_dir.
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config, # Utilisez la config de quantification si le modèle a été fine-tuné avec
    device_map="auto",
    token=os.environ.get("HF_TOKEN") # Si nécessaire pour charger certains modèles
)

# Exemple d'inférence pour la génération de code
prompt_code = "Écris une fonction Python pour calculer la somme des éléments d'une liste."
inputs_code = tokenizer(prompt_code, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs_code = model.generate(
        **inputs_code,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id # Assurez-vous que pad_token_id est défini
    )
generated_code = tokenizer.decode(outputs_code[0], skip_special_tokens=True)
print("--- Génération de Code ---")
print(generated_code)

# Exemple d'inférence pour la conversation
prompt_conversation = "Quelle est la capitale de l'Australie ?"
inputs_conversation = tokenizer(prompt_conversation, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs_conversation = model.generate(
        **inputs_conversation,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
generated_conversation = tokenizer.decode(outputs_conversation[0], skip_special_tokens=True)
print("\n--- Génération de Conversation ---")
print(generated_conversation)

```

### Intégration dans VSCode

1.  **Cloner le dépôt** : Ouvrez votre terminal VSCode (`Ctrl+` ou `Cmd+`) et clonez le dépôt :
    ```bash
    git clone https://github.com/eurocybersecurite/mimo-llm.git
    cd mimo-llm
    ```
2.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
3.  **Exécuter le script de fine-tuning ou d'inférence** : Vous pouvez exécuter `fine_tune_mimo.py` ou créer un nouveau script Python pour l'inférence, en utilisant le code d'exemple ci-dessus.
4.  **Utiliser des extensions** : Pour une intégration plus poussée, vous pourriez explorer des extensions VSCode qui permettent d'exécuter du code Python ou d'interagir avec des modèles de langage locaux.

**Note sur la visibilité du dépôt :**
Je ne peux pas modifier les paramètres de visibilité du dépôt GitHub. Vous devrez vous rendre sur GitHub et rendre le dépôt public manuellement si vous le souhaitez.

---
Auteur : ABDESSEMED Mohamed
Entreprise : Eurocybersecurite
Contact : mohamed.abdessemed@eurocybersecurite.fr
