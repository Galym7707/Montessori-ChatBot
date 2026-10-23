import os
import torch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from unsloth import FastLanguageModel

# 1. Загрузка текстовых файлов
docs = []
file_paths = ["book1.txt", "book2.txt"]

for path in file_paths:
    if os.path.exists(path):
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

# 2. Нарезка на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 3. Эмбеддинги (Hugging Face)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={'device': 'cpu'}
)

# 4. Инициализация ChromaDB
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

# 5. Загрузка Qwen 2.5 3B (4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 6. Функция Strict RAG
def ask_rag(query: str) -> str:
    docs = retriever.invoke(query)
    
    if not docs:
        return "В представленной книге нет ответа на этот вопрос."
    
    context_str = "\n\n".join([f"• {d.page_content}" for d in docs])
    
    prompt = f"""<|im_start|>system
Ты — строгий текстовый ассистент.
Отвечай ИСКЛЮЧИТЕЛЬНО на основе КОНТЕКСТА.
ТЕБЕ КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ СОБСТВЕННЫЕ ИЛИ ВНЕШНИЕ ЗНАНИЯ.
Если в контексте прямо нет ответа на вопрос, ответь ровно одной фразой: "В представленной книге нет ответа на этот вопрос."

КОНТЕКСТ ИЗ КНИГИ:
{context_str}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.01, 
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    print("Ответ 1:", ask_rag("Каковы требования к организации спального места?"))
    print("Ответ 2:", ask_rag("Как выбрать автокресло для ребенка?"))
