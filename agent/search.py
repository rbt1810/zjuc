import ollama
from langchain.chains import RetrievalQA

# 初始化ollama客户端（确保ollama服务运行）
client = ollama.Client(host='http://127.0.0.1:11434')

# 定义RAG智能体
def course_agent(query):
    # 检索相关文档片段
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    context = retriever.get_relevant_documents(query)
    
    # 构建提示（集成课程上下文）
    prompt = f"""
    You are an AI assistant for MA333 Big Data Analysis Course. Answer based ONLY on the course context below.
    Context: {context}
    Query: {query}
    """
    
    # 调用qwen3:32b模型
    response = client.chat(model='qwen3:32b', messages=[
        {'role': 'system', 'content': 'You are a professional course assistant for MA333.'},
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

# 示例查询
query = "Explain data discretization methods mentioned in the course."
answer = course_agent(query)
print(f"Answer: {answer}")