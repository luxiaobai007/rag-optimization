import json
import time

import numpy as np
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.config import Keys
from langchain_community.embeddings import DashScopeEmbeddings

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=Keys.DASHSCOPE_API_KEY
)



embedding = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=Keys.DASHSCOPE_API_KEY
)




def extract_text_from_pdf(pdf_path):
    """从一个pdf文件中提取文本

    Args:
        pdf_path: pdf文件路径

    Return:
        str: extracted text from pdf
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def chunk_text(text, chunk_size, overlap):
    """
    将文本分割为具有重叠部分的文本块

    :param text: str, 原始文本
    :param chunk_size: int, 每个文本块的最大长度
    :param overlap: int, 相邻文本块之间的重叠长度
    :return: List[str], 分割后的文本块列表
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须在 [0, chunk_size) 范围内")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 更新下一块的起始位置，考虑 overlap
        start += chunk_size - overlap

    return chunks

# 将分割的文本块进行向量化
def create_embeddings(texts,batch_size=10):
    embeddings = []
    for i in range(0,len(texts),batch_size):
        batch = texts[i:i+batch_size]
        response = embedding.embed_documents(batch)
        embeddings.extend(response)
        time.sleep(0.2)
    return  embeddings


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query,text_chunks,embeddings,k=5):
    query_embeddings = embedding.embed_query(query)
    similarity_scores = []
    for i, chunk_embeddings in enumerate(embeddings):
        similarity_score =cosine_similarity(np.array(query_embeddings),np.array(chunk_embeddings))
        similarity_scores.append((i,similarity_score))
    similarity_scores.sort(key=lambda x:x[1],reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[i] for i in top_indices]


# llm基于相似度检索的结果进行响应
def generate_response(system_prompt,user_message):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    llm_chain = prompt | llm
    return llm_chain.invoke({"question": user_message})



pdf_path = "../data/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
#重复的文本块在每段文本块的10%~20%之间
chunks = chunk_text(extracted_text, chunk_size=1000, overlap=200)
# print("文本块数:", len(chunks))
# print("\n第一块：")
# print(chunks[0])
# print("\n第二块：")
# print(chunks[1])

embeddings = create_embeddings(chunks)

with open("../data/val.json") as f:
    data = json.load(f)

query = data[0]['question']
print("question:",query)

top_chunks = semantic_search(query,chunks,embeddings,k=2)

# for i, chunk in enumerate(top_chunks):
#     print(f"Context {i + 1}:\n{chunk}\n=====================================")


system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

ai_response = generate_response(system_prompt,user_prompt)
print("AI响应：：",ai_response.content)


# 评估AI的结果
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print("评估响应:",evaluation_response.content)






