# 🚀 Mimo Language Model

Mimo est un modèle de langage AI pour exceller à la fois en **génération de code** et en **conversations naturelles**.  
Il est issu d'un mélange de datasets puissants.

![Mimo](assets/mimo.png)

---

## ✨ Points forts de Mimo

- 🔧 **Optimisé pour le code** : génération fiable de scripts Python, JS, etc.  
- 💬 **Excellente conversation** : réponses naturelles et contextualisées.  
- ⚡ **Compatibilité multiplateforme** : fonctionne sur Mac, PC et VSCode.  
- 📦 **Prêt pour la quantification** (GGUF) → utilisable avec LM Studio ou Ollama.  

---

## 📦 Installation

Clonez le dépôt et installez les dépendances :

```bash
git clone https://github.com/votre-utilisateur/mimo-llm.git
cd mimo-llm
pip install -r requirements.txt
```

⚠️ Assurez-vous d’avoir `git-lfs` installé pour gérer les poids du modèle.

---

## 🔑 Configuration

Avant toute utilisation, configurez votre **Hugging Face Token** :

```bash
export HF_TOKEN="votre_token_hugging_face"
```

---

## 🏋️ Fine-tuning

Lancez le fine-tuning avec :

```bash
python fine_tune_mimo.py
```

**IMPORTANT :** Remplacez `example.jsonl` par votre propre fichier de dataset avant d'exécuter ce script. Le fichier `example.jsonl` contient quelques exemples fictifs à des fins de démonstration.

- Utilise vos données perso (`example.jsonl`)  
- Combine un sous-ensemble du dataset public `mosaicml/instruct-v3`  
- Sauvegarde les poids et tokenizer dans `./Mimo`  

⚠️ **Note de sécurité** : ne publiez jamais vos données privées ou sensibles dans le dépôt public.

---

## 🧑‍💻 Exemples d’utilisation

### Génération de code

```python
prompt = "Écris une fonction Python pour trier une liste."
inputs = mimo_tokenizer(prompt, return_tensors="pt")
outputs = mimo_model.generate(**inputs, max_new_tokens=100)
print(mimo_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Conversation

```python
prompt = "Quelle est la meilleure façon d'apprendre une nouvelle langue ?"
inputs = mimo_tokenizer(prompt, return_tensors="pt")
outputs = mimo_model.generate(**inputs, max_new_tokens=150)
print(mimo_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Performances comparatives

| Modèle                          | Code (Python) | Conversation | Mémoire requise |
|---------------------------------|---------------|--------------|-----------------|
| GPT-Neo 1.3B                    | ⭐⭐            | ⭐⭐           | ~12 Go          |
| DeepSeek-Qwen-1.5B (base)       | ⭐⭐⭐           | ⭐⭐⭐          | ~10 Go          |
| **Mimo-1.5B (fine-tuned)**      | ⭐⭐⭐⭐          | ⭐⭐⭐⭐         | ~8 Go (quantisé) |

➡️ **Mimo surpasse la version de base** sur les benchmarks internes (code + QA).

![Mimo Performance](assets/mimo_conv_code.png)

---

## 📂 Structure du dépôt

```
Mimo/
├── README.md
├── assets/mimo.png
├── assets/mimo_conv_code.png
├── example.jsonl        # Jeu de données fictif
├── fine_tune_mimo.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🛠️ Intégration dans VSCode

1. Clonez le dépôt :  
   ```bash
   git clone https://github.com/votre-utilisateur/mimo-llm.git
   cd mimo-llm
   ```
2. Installez les dépendances :  
   ```bash
   pip install -r requirements.txt
   ```
3. Exécutez soit :  
   - `fine_tune_mimo.py` → pour l’entraînement  
   - un script d’inférence personnalisé  

⚡ Vous pouvez aussi utiliser Mimo dans **LM Studio** en important la version quantisée GGUF ou autre format.

---

## 📜 Licence

Ce projet est sous licence **Apache 2.0**.  
Voir le fichier [LICENSE](LICENSE) pour les détails.

⚠️ **Note importante** : le fichier `example.jsonl` est fourni uniquement comme exemple.  
N’incluez jamais vos données sensibles ou privées dans le dépôt public.

---

## 📧 Auteur

- **Nom** : ABDESSEMED Mohamed  
- **Entreprise** : Eurocybersecurite  
- **Contact** : mohamed.abdessemed@eurocybersecurite.fr

