# AfloChat

## Description

Aflokkat_AI est un projet de chatbot personnalisé utilisant le Retrieval-Augmented Generation (RAG). Le projet est hébergé sur un ordinateur équipé de 2 RTX A4000 avec 16Go de RAM chacune.

## Structure du projet

- `src/` : Contient le code source du projet.
- `data/` : Contient les données utilisées pour entraîner et tester le modèle.
- `models/` : Contient les modèles entraînés.
- `notebooks/` : Contient les notebooks Jupyter pour les POC et l'expérimentation.
- `scripts/` : Contient les scripts utilitaires pour la gestion du projet.
- `docs/` : Contient la documentation du projet.
- `tests/` : Contient les tests unitaires et d'intégration.
- `releases/` : Contient les versions publiées du projet.

## Phase de Sourcing

La phase de sourcing consiste à collecter et préparer les données nécessaires pour entraîner le modèle de chatbot. Voici les étapes à suivre :

1. Collecte des données : Rassemblez les données pertinentes pour votre chatbot.
2. Prétraitement des données : Nettoyez et formatez les données pour les rendre utilisables.
3. Stockage des données : Enregistrez les données prétraitées dans le répertoire `data/`.

## Choix des Modèles

Pour le choix des modèles, nous recommandons d'utiliser des modèles récents et performants adaptés à votre cas d'utilisation. Voici quelques suggestions :

- **Mistral** : Pour des performances élevées en génération de texte et compréhension du langage naturel : https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407

Vous pouvez également explorer d'autres modèles disponibles sur des plateformes comme Hugging Face.

## LangChain

LangChain est une bibliothèque puissante pour la construction de chaînes de traitement de langage naturel. Elle permet de combiner plusieurs modèles et techniques de traitement de texte pour créer des pipelines sophistiqués. LangChain facilite l'intégration de différentes étapes de traitement, telles que la tokenisation, l'analyse syntaxique, la génération de texte, et bien plus encore. Pour plus d'informations, consultez le dépôt GitHub de LangChain : https://github.com/langchain/langchain.

## Base de données vectorielle

Pour stocker et rechercher efficacement des vecteurs, vous pouvez utiliser des bases de données vectorielles. Voici quelques options populaires :

- **Faiss** : Une bibliothèque de Facebook AI Research pour une recherche de similarité rapide et précise : https://github.com/facebookresearch/faiss
- **Chroma** : Une base de données vectorielle rapide et évolutive pour les applications de machine learning : https://www.trychroma.com/

Ces bases de données vous permettront de gérer efficacement les vecteurs générés par votre modèle de chatbot.

## Installation

Instructions pour installer les dépendances et configurer l'environnement.

## Configuration

Instructions pour configurer le projet, y compris les variables d'environnement et les fichiers de configuration nécessaires.

## Utilisation

Instructions pour utiliser le chatbot.

## Contribuer

Instructions pour contribuer au projet.

## Licence

Informations sur la licence du projet.
