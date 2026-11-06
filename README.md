# TP — RAG multimodal et agentique

**Réalisé par :** Aliaa CHIBEL — **Encadrant :** Pr. Mohamed YOUSSFI

Agent capable d'extraire, d'indexer et d'interroger de façon autonome le texte et les images de l'article
*Attention Is All You Need*. Tout le travail se trouve dans le notebook
[`rag_multimodal_agentique_chibel.ipynb`](rag_multimodal_agentique_chibel.ipynb).

## Architecture

```mermaid
graph TD
    PDF[PDF] --> T[Texte et fragments]
    PDF --> I[Images]
    T --> C[CLIP ViT-B-32]
    I --> C
    C --> DB[(ChromaDB)]

    Q([Question : texte et/ou image]) --> M
    M["mind : LLM (Reasoning)"] -->|appel d'outil| B["body : retriever_tool (Action)"]
    B <--> DB
    B --> O["observation : ajout des figures"]
    O --> M
    M -->|pas d'appel d'outil| R([Réponse citée])
```

## Correspondance avec l'énoncé

| Étape | Réalisation |
|---|---|
| 1. Extraction | PyMuPDF : texte par page, fragments d'environ 60 tokens, images sauvegardées |
| 2. Embeddings | CLIP `ViT-B-32` (`laion2b_s34b_b79k`) via `open_clip`, vecteurs normalisés |
| 3. Indexation | ChromaDB, identifiant SHA-256, métadonnées (type, page, chemin) |
| 4. Recherche | Similarité cosinus, requête texte ou image, affichage des `k` meilleurs résultats |
| 5. Outil | `retriever_tool`, avec description et schéma d'arguments pour guider le LLM |
| 6. Agent | `StateGraph` LangGraph : `mind`, `body`, `observation` (cycle ReAct) |
| 7. Tests | 8 scénarios, trace de chaque décision et bilan automatique |

## Résultats

Exécution enregistrée dans le notebook (le LLM n'étant pas déterministe, les réponses peuvent varier) :

- 8 décisions d'appel d'outil correctes sur 8, et au plus 3 tours de recherche par question.
- Réponses justes sur le fond dans les 8 scénarios, dont un cas d'information absente et une requête par image.
- Contrôle automatique des faits : 7/8. Le test 5 ne mentionne pas « sinus », terme que la question ne demandait pas.
- Principale limite : certaines citations de page sont imprécises (test 8). L'analyse complète se trouve à la fin du notebook.

## Exécution

Testé sous Windows avec Python 3.12, sur CPU.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
echo GEMINI_API_KEY=votre_cle> .env
jupyter notebook rag_multimodal_agentique_chibel.ipynb
```

Le PDF est téléchargé automatiquement au premier lancement. Les poids de CLIP (environ 600 Mo) le sont aussi.
Le LLM est `gemini-3.5-flash-lite`, appelé via `langchain-google-genai`. Une clé s'obtient sur
[Google AI Studio](https://aistudio.google.com/apikey). Le modèle se change avec la variable `MODELE_LLM` : il doit
accepter les images et les appels d'outils.
