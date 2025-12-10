# src/rag_qa.py
import os
from .retriever import Retriever
from .embedder import embed_texts
import textwrap
import json
from openai import OpenAI

# will be configured at runtime
LLM_BACKEND = {"use_local": False, "local_model": None, "openai_key": None, "hf_model": None}

def initialize_llm():
    use_local = os.getenv("USE_LOCAL_LLM", "0") == "1"
    LLM_BACKEND["use_local"] = use_local
    if use_local:
        LLM_BACKEND["local_model"] = os.getenv("LOCAL_LLM_MODEL", "local-model-path")
    else:
        LLM_BACKEND["openai_key"] = os.getenv("OPENAI_API_KEY")

def build_prompt(query, hits, max_chars=4000):
    ctx_parts = []
    for i, h in enumerate(hits):
        meta = h["meta"]
        page = meta.get("page")
        text = meta.get("text").replace("\n", " ")
        ctx_parts.append(f"[source: page {page} | id: {meta.get('id')}]\n{text[:1200]}")
    context = "\n\n---\n\n".join(ctx_parts)
    prompt = textwrap.dedent(f"""
    You are a document-grounded assistant. Use ONLY the context below to answer.
    If the answer can't be found, respond: "Not found in document".

    Context:
    {context}

    Question: {query}

    Answer (concise, include inline citations like (page X)):
    """)
    return prompt

# OpenAI path
def call_openai(prompt, max_tokens=400):
    client = OpenAI(api_key="your-open-api-key-here")


    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

# Local HF generate (simple wrapper)
def call_local_model(prompt, max_length=512):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    model_name = os.getenv("LOCAL_LLM_MODEL")
    if model_name is None:
        raise ValueError("LOCAL_LLM_MODEL not set in env.")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_length=max_length)
    out = gen(prompt, do_sample=False)[0]['generated_text']
    return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()

def answer_query(query, retriever: Retriever, k=5):
    hits = retriever.retrieve(query, k=k)
    prompt = build_prompt(query, hits)
    if LLM_BACKEND["use_local"]:
        answer = call_local_model(prompt)
    else:
        answer = call_openai(prompt)
    return answer, hits

# convenience init
def initialize_llm_wrapper():
    initialize_llm()
