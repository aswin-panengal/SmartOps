# SmartOps 

A production-grade, dual-engine AI operations platform that allows users to query raw data and internal company documents using natural language. 

## Architecture

SmartOps is built on a dual-engine architecture to handle both structured and unstructured data efficiently without relying on massive, slow, and expensive LLM context windows.

* **Engine 1: Analytical (CSV)** — Instead of feeding entire datasets to an LLM, this engine extracts a lightweight Pandas "blueprint" (columns, types, shape, sample rows) and uses Gemini to generate and execute native Python code on the fly to answer user queries.
* **Engine 2: Semantic (PDF)** — A complete Retrieval-Augmented Generation (RAG) pipeline. It chunks internal documents, embeds them via Google's embedding models, stores the vectors locally in Qdrant, and retrieves exact context for Q&A.

**Tech Stack:** FastAPI, Qdrant (Docker), LangChain, Google Gemini 2.5 Flash, Pandas.

## How to Run Locally

### Prerequisites
* Python 3.10+
* Docker Desktop (Must be running for the Qdrant database)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aswin-panengal/smartops-platform.git
cd smartops-platform
