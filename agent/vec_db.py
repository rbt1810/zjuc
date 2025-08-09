import json
import numpy as np
import os
import faiss
from typing import List, Dict, Any
from pathlib import Path

class VectorDatabase:
    """向量数据库类，用于高效存储和检索向量化内容"""
    
    def __init__(self, index_path: str = "vector_index", metadata_path: str = "metadata.json"):
        """
        初始化向量数据库
        
        参数:
            index_path: 保存FAISS索引的路径
            metadata_path: 保存元数据的JSON路径
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.vector_dim = None
        
        # 创建目录（如果不存在）
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
    
    def build_from_vectors_file(self, vectors_file: str) -> None:
        """
        从向量化的JSON文件构建数据库
        
        参数:
            vectors_file: 包含向量化数据的JSON文件路径
        """
        # 加载数据
        with open(vectors_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vector_dim = 768
        self.metadata = []
        
        # 提取向量和元数据
        all_vectors = []
        for i, block in enumerate(data['blocks']):
            vector = block['vector']
            self.metadata.append({
                "block_id": block['id'],
                "text": block['text'][:500],  # 保存前500个字符以便展示
                "full_text": block['text'],
                "page": block['page'],
                # "section": block['section'],
                # "doc_title": block['doc_metadata']['document_title'],
                # "doc_author": block['doc_metadata']['author'],
                "index": i  # 保存原始索引位置
            })
            all_vectors.append(vector)
        
        # 转换为NumPy数组
        vectors = np.array(all_vectors, dtype=np.float32)
        
        # 创建并填充FAISS索引
        self.index = faiss.IndexFlatIP(self.vector_dim)  # 使用内积（余弦相似度）
        self.index.add(vectors)
        print(f"✅ 成功构建向量数据库！包含 {len(vectors)} 个向量")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        在数据库中搜索相似项
        
        参数:
            query_vector: 查询向量（必须是二维数组，shape=(1, vector_dim)）
            k: 返回最相似的k个结果
            
        返回:
            包含匹配结果元数据的字典列表
        """
        if self.index is None:
            raise RuntimeError("数据库尚未构建，请先调用build_from_vectors_file()")
        
        # 确保查询向量格式正确（二维数组）
        if isinstance(query_vector, list):
            query_vector = np.array([query_vector], dtype=np.float32)
        elif query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)  # 修正：将一维转为二维
        
        # 执行搜索
        distances, indices = self.index.search(query_vector, k)
        
        # 获取结果元数据
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            results.append({
                "similarity": float(distances[0][i]),
                **self.metadata[idx]
            })
        
        return results
    
    def text_search(self, query: str, model, k: int = 5) -> List[Dict[str, Any]]:
        """
        通过文本查询进行搜索（需要传入向量化模型）
        
        参数:
            query: 查询文本
            model: 向量化模型实例
            k: 返回最相似的k个结果
            
        返回:
            包含匹配结果元数据的字典列表
        """
        # 向量化查询文本
        query_vector = model.vectorize([query])[0]
        return self.search(query_vector, k)
    
    def save(self) -> None:
        """保存数据库到磁盘"""
        if self.index is None:
            raise RuntimeError("数据库尚未构建，无法保存")
        
        # 保存FAISS索引
        faiss.write_index(self.index, self.index_path)
        
        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": self.metadata,
                "vector_dim": self.vector_dim
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 数据库已保存到: {self.index_path} 和 {self.metadata_path}")
    
    def load(self) -> None:
        """从磁盘加载数据库"""
        # 加载FAISS索引
        self.index = faiss.read_index(self.index_path)
        
        # 加载元数据
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = data["metadata"]
            self.vector_dim = data["vector_dim"]
        
        print(f"🔍 数据库已加载! 包含 {len(self.metadata)} 个向量")

# 主函数：构建和使用向量数据库
if __name__ == "__main__":
    # 配置文件路径
    VECTORS_FILE = "./lectures/introduction/introduction_Course_Syllabus_vectorized.json"  # 之前生成的向量化文件
    MODEL_PATH = "models/sentence-transformers_all-mpnet-base-v2"  # 向量化模型路径
    DB_PATH = "./lectures/introduction/introduction_Course_Syllabus_db"  # 数据库保存路径
    
    # 1. 创建向量数据库实例
    print("=" * 60)
    print("📚 构建向量数据库...")
    db = VectorDatabase(
        index_path=f"{DB_PATH}/index.faiss", 
        metadata_path=f"{DB_PATH}/metadata.json"
    )
    
    # 2. 从向量化文件构建数据库
    db.build_from_vectors_file(VECTORS_FILE)
    
    # 3. 保存数据库以便后续使用
    os.makedirs(DB_PATH, exist_ok=True)
    db.save()
    
    # 4. 测试查询 (需先加载模型)
    print("\n🔍 测试查询功能...")
    
    # 加载向量化模型 (使用之前的实现)
    print(f"加载模型: {MODEL_PATH}")
    # 注意：这里需要导入您之前的向量化模型实现
    # from minimal_vectorizer import MinimalVectorizer  # 取消注释并使用您的实现
    # vectorizer = MinimalVectorizer(MODEL_PATH)
    
    # 由于导入可能复杂，这里简化测试
    print("（在实际使用中，需要加载向量化模型来处理文本查询）")
    
    # 5. 示例搜索 - 使用预先加载的数据库
    print("\n🔄 重新加载数据库进行测试...")
    reloaded_db = VectorDatabase(
        index_path=f"{DB_PATH}/index.faiss", 
        metadata_path=f"{DB_PATH}/metadata.json"
    )
    reloaded_db.load()
    
    # 6. 模拟查询 - 修正向量形状
    print("\n示例搜索结果:")
    # 创建正确形状的查询向量 (1 x vector_dim)
    example_query_vector = np.random.randn(1, reloaded_db.vector_dim).astype(np.float32)
    
    # 执行搜索
    results = reloaded_db.search(example_query_vector, k=3)
    
    # 显示结果 - 使用实际存在的元数据字段
    for i, res in enumerate(results):
        print(f"\n结果 #{i+1}")
        print(f"  相似度: {res['similarity']:.4f}")
        # 使用实际保存的字段
        print(f"  块ID: {res['block_id']}")
        print(f"  内容: {res['text']}...")
        print(f"  页码: {res['page']}")
        print(f"  原始索引: {res['index']}")
    
    print("\n✨ 向量数据库构建完成，可以开始使用！")
