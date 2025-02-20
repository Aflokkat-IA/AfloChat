# 🤖 AfloChat

## 📝 Description

AfloChat is a customized chatbot project utilizing **Retrieval-Augmented Generation (RAG)**. It is designed to assist **students** and **staff** in a school setting by improving understanding of **courses, administrative documentation, and regulations**. 🏫📚 The chatbot tailors its responses based on the user’s profile (_student or staff_), the courses they have taken, or the department they belong to.

The project is hosted on a computer equipped with **two RTX A4000 GPUs**, each with **16GB of RAM**. 🚀💻

## 📂 Project Structure

- 📜 **`src/`** : Contains the project's source code.
- 📁 **`data/`** : Stores the data used to train and test the model.
- 🎯 **`models/`** : Holds the trained models.
- 📓 **`notebooks/`** : Includes Jupyter notebooks for POCs and experimentation.
- ⚙️ **`scripts/`** : Contains utility scripts for project management.
- 📑 **`docs/`** : Provides project documentation.
- 🧪 **`tests/`** : Includes unit and integration tests.
- 🚀 **`releases/`** : Stores published versions of the project.

## 🔍 Data Sourcing Phase

The data sourcing phase consists of collecting and preparing the necessary information for training the chatbot. The steps involved are:

1. 📥 **Data Collection**: Gather relevant information, including **course materials, administrative documents, and regulations**.
2. 🛠 **Data Preprocessing**: Clean and format the data to ensure usability.
3. 💾 **Data Storage**: Save the preprocessed data in the `data/` directory.

## 🏆 Model Selection

For optimal performance, we recommend using recent and efficient language models suited to this use case. Some suggested models include:

- 🚀 **Mistral**: A high-performance model for text generation and natural language understanding: [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)

Other models can be explored on platforms like **Hugging Face**.

## 🔗 LangChain Integration

[LangChain](https://github.com/langchain/langchain) is a powerful library for building **natural language processing pipelines**. It enables the combination of multiple models and text-processing techniques to create advanced workflows. LangChain facilitates tasks such as **tokenization, syntactic analysis, and text generation**. ⚡🤖

## 🗄️ Vector Database

To efficiently store and search for **vector embeddings**, the following vector databases are recommended:

- 🔍 **[Faiss](https://github.com/facebookresearch/faiss)**: A library developed by **Facebook AI Research** for fast and accurate similarity search.
- ⚡ **[Chroma](https://www.trychroma.com/)**: A scalable and high-speed vector database for machine learning applications.

These databases enable efficient handling of the chatbot's **vectorized data**.

## ⚙️ Installation

📌 Instructions on how to **install dependencies** and **set up the environment**.

## 🔧 Configuration

📌 Guidelines for configuring the project, including **setting environment variables** and **required configuration files**.

## 🖥️ Usage

📌 Instructions on how to **interact with the chatbot**.

## 🤝 Contributing

📌 Guidelines for **contributing to the project**.

## 📜 License

📌 Information regarding the project's **license**. 📄
