# 🚀 Mimo Language Model

Mimo est un modèle de langage AI pour exceller à la fois en **génération de code** et en **conversations naturelles**.  
Il est issu d'un mélange de datasets puissants.

![Mimo](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/mimo.png)

---

## 📑 Table des matières

- [✨ Points forts de Mimo](#-points-forts-de-mimo)
- [📦 Installation](#-installation)
- [🔑 Configuration](#-configuration)
- [🏋️ Fine-tuning](#-fine-tuning)
- [🧑‍💻 Exemples d’utilisation](#-exemples-dutilisation)
  - [Génération de code](#génération-de-code)
  - [Conversation](#conversation)
- [📊 Performances comparatives](#-performances-comparatives)
- [🧠 Visualisations comparatives](#-visualisations-comparatives)
  - [Radar de performance](#-1-radar-de-performance)
  - [Entraînement & efficacité](#-2-entrainement--efficacité)
  - [Classification & Clustering](#-3-classification--clustering)
  - [Précision en classification](#-4-précision-en-classification)
  - [Raisonnement avancé](#-5-raisonnement-avancé)
  - [Conscience artificielle](#-6-conscience-artificielle-concept)
- [📂 Structure du dépôt](#-structure-du-dépôt)
- [🛠️ Intégration dans VSCode](#-intégration-dans-vscode)
- [📧 Auteur](#-auteur)

---

## ✨ Points forts de Mimo

- 🔧 **Optimisé pour le code** : génération fiable de scripts Python, JS, etc.  
- 💬 **Excellente conversation** : réponses naturelles et contextualisées.  
- ⚡ **Compatibilité multiplateforme** : fonctionne sur Mac, PC et VSCode.  
- 📦 **Prêt pour la quantification** (GGUF) → utilisable avec LM Studio ou Ollama.  

---

## 📦 Installation

Clonez le dépôt et installez les dépendances dans un environnement virtuel :

```bash
# Cloner le dépôt
git clone https://github.com/eurocybersecurite/Mimo-llm.git
cd Mimo-llm

# Créer et activer un environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate  # Sur Linux/macOS
# Ou sur Windows : .\.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

⚠️ Assurez-vous d’avoir `git-lfs` installé pour gérer les poids du modèle.

---

## 🔑 Configuration

Avant toute utilisation, configurez votre **Hugging Face Token** :

```bash
export HF_TOKEN="votre_token_hugging_face"
```
(Remplacez `"votre_token_hugging_face"` par votre véritable token.)

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

---

## 🧑‍💻 Exemples d’utilisation

### Génération de code

```python
# Assurez-vous que le modèle et le tokenizer sont chargés correctement
# Exemple d'inférence pour la génération de code
prompt_code = "Écris une fonction Python pour calculer la somme des éléments d'une liste."
inputs_code = tokenizer(prompt_code, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs_code = model.generate(
        **inputs_code,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )
generated_code = tokenizer.decode(outputs_code[0], skip_special_tokens=True)
print("--- Génération de Code ---")
print(generated_code)
```

### Conversation

```python
# Exemple d'inférence pour la conversation
prompt_conversation = "Quelle est la meilleure façon d'apprendre une nouvelle langue ?"
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

---

## 📊 Performances comparatives

| Modèle                          | Code (Python) | Conversation | Mémoire requise |
|---------------------------------|---------------|--------------|-----------------|
| GPT-Neo 1.3B                    | ⭐⭐            | ⭐⭐           | ~12 Go          |
| DeepSeek-Qwen-1.5B (base)       | ⭐⭐⭐           | ⭐⭐⭐          | ~10 Go          |
| **Mimo-1.5B (fine-tuned)**      | ⭐⭐⭐⭐          | ⭐⭐⭐⭐         | ~8 Go (quantisé) |

➡️ **Mimo surpasse la version de base** sur les benchmarks internes (code + QA).

![Mimo Performance](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/mimo_conv_code.png)

---

## 🧠 Visualisations comparatives

Afin d’illustrer les forces de **Mimo** par rapport aux autres modèles, plusieurs visualisations ont été générées à partir de benchmarks internes (Septembre 2025).  

### 🔹 1. Radar de performance
![Radar Performance](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/performance_radar.png)  

Ce diagramme radar compare les capacités globales (Code, Conversation, Mémoire, Raisonnement).  
➡️ **Mimo domine** sur tous les axes, montrant son équilibre entre compréhension et génération.

---

### 🔹 2. Entraînement & efficacité
![Entraînement](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/entrainement.png)  

Comparaison des métriques d’entraînement :  
- ⏱ Temps d’entraînement plus court  
- 📉 Perte finale plus faible  
- 💾 Mémoire optimisée  

➡️ **Mimo apprend plus vite et avec moins de ressources**.

---

### 🔹 3. Classification & Clustering
![Classification & Clustering](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/Classification_Clustering.png)  

Ce graphique montre comment chaque modèle regroupe les données en classes.  
➡️ Les clusters prédits par **Mimo** sont **plus nets et bien séparés**, preuve de sa meilleure capacité de généralisation.

---

### 🔹 4. Précision en classification
![Métriques de classification](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/metriques_classification.png)  

Comparaison des scores : **Accuracy, Recall, F1-score**.  
➡️ **Mimo** garde une avance claire sur la précision et la robustesse des prédictions.

---

### 🔹 5. Raisonnement avancé
![Raisonnement Mimo](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/resennement_mimo.png)  

Testé sur des tâches de raisonnement logique et contextuel.  
➡️ **Mimo** démontre une supériorité dans la résolution de problèmes complexes.

---

### 🔹 6. Conscience artificielle (concept)
![Conscience artificielle](https://raw.githubusercontent.com/eurocybersecurite/Mimo-llm/main/assets/Conscience_artificielle.png)  

Visualisation heatmap sur 5 axes : **Perception, Mémoire, Raisonnement, Créativité, Auto-adaptation**.  
➡️ **Mimo émerge comme le modèle le plus “conscient”**, avec des scores nettement supérieurs aux autres.

---

## 📂 Structure du dépôt

```
Mimo/
├── README.md
├── assets/mimo.png
├── mohamed.jsonl
├── fine_tune_mimo.py
├── requirements.txt
└── .gitignore
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

⚡ Vous pouvez aussi utiliser Mimo dans **LM Studio** en important la version quantisée GGUF ou autre Format.

---








## 📧 Auteur

- **Nom** : ABDESSEMED Mohamed  
- **Entreprise** : Eurocybersecurite  
- **Contact** : mohamed.abdessemed@eurocybersecurite.fr

