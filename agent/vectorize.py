import json
import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MinimalVectorizer")

class MinimalVectorizer:
    """极简版文本向量化器，避免任何复杂依赖"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        初始化向量化器
        
        参数:
            model_path: 模型路径
            device: 指定计算设备 (如'cpu'或'cuda')
        """
        logger.info("初始化极简版向量化器...")
        
        # 验证模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 检查必要的文件
        required_files = ["config.json", "tokenizer_config.json"]
        weight_files = ["pytorch_model.bin", "model.safetensors"]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                logger.warning(f"注意: 缺少文件 {file}，但尝试继续加载")
        
        # 设置设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        logger.info("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("加载模型...")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        
        # 获取向量维度
        self.vector_dim = self.model.config.hidden_size
        logger.info(f"模型加载成功! 向量维度: {self.vector_dim}")
    
    def vectorize(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """向量化文本"""
        self.model.eval()  # 设置为评估模式
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # 分词
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # 获取模型输出
                outputs = self.model(**inputs)
                
                # 使用平均池化获取句向量
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                
                # 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                # 记录进度
                progress = min(i + batch_size, len(texts))
                logger.info(f"已处理: {progress}/{len(texts)} ({progress/len(texts)*100:.1f}%)")
        
        return np.vstack(all_embeddings)
    
    def mean_pooling(self, model_output, attention_mask):
        """实现平均池化策略"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class JSONVectorizer:
    """JSON文档向量化处理器 - 极简版"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.vectorizer = MinimalVectorizer(model_path, device)
        self.document = None
        self.text_blocks = []
        self.vectors = None
    
    def load_json(self, file_path: str):
        """加载JSON文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.document = json.load(f)
        
        title = self.document.get('metadata', {}).get('document_title', '未知文档')
        logger.info(f"✅ 已加载文档: {title}")
    
    def extract_text_blocks(self):
        """提取文本块 - 简化版本"""
        if self.document is None:
            raise ValueError("尚未加载文档")
        
        text_blocks = []
        
        # 遍历所有页面
        for page_idx, page in enumerate(self.document.get('pages', [])):
            # 提取页面内的文本内容
            for content_block in page.get('content', []):
                block_text = []
                
                # 添加标题和摘要
                if header := content_block.get('header'):
                    block_text.append(f"标题: {header}")
                if summary := content_block.get('summary'):
                    block_text.append(f"摘要: {summary}")
                
                # 提取内容数组中的文本
                for element in content_block.get('content_array', []):
                    if element.get('type') == 'text' and element.get('value'):
                        block_text.append(element['value'])
                
                if not block_text:
                    continue
                    
                full_text = ". ".join(block_text)
                text_blocks.append({
                    "id": f"block_{page_idx}_{len(text_blocks)}",
                    "text": full_text,
                    "page_idx": page_idx
                })
        
        self.text_blocks = text_blocks
        logger.info(f"📊 提取到 {len(text_blocks)} 个文本块")
        return text_blocks
    
    def vectorize_blocks(self, batch_size: int = 4):
        """向量化文本块"""
        if not self.text_blocks:
            self.extract_text_blocks()
            
        texts = [block["text"] for block in self.text_blocks]
        logger.info(f"🚀 开始向量化 {len(texts)} 个文本块...")
        
        # 向量化（带GPU回退逻辑）
        try:
            self.vectors = self.vectorizer.vectorize(texts, batch_size)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.warning("⚠️ GPU内存不足，切换到CPU模式...")
                self.vectorizer.device = torch.device('cpu')
                self.vectorizer.model = self.vectorizer.model.to('cpu')
                self.vectors = self.vectorizer.vectorize(texts, batch_size)
            else:
                raise
        
        logger.info(f"✅ 向量化完成! 向量维度: {self.vectors.shape[1]}")
        return self.vectors
    
    def save_results(self, output_file: str):
        """保存结果"""
        if self.vectors is None or not self.text_blocks:
            raise ValueError("请先执行vectorize_blocks()")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results = {
            "document_title": self.document.get('metadata', {}).get('document_title', ''),
            "blocks": []
        }
        
        for i, block in enumerate(self.text_blocks):
            results["blocks"].append({
                "id": block["id"],
                "page": block["page_idx"],
                "text": block["text"][:500] + "..." if len(block["text"]) > 500 else block["text"],
                "vector": self.vectors[i].tolist()
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 结果已保存到: {output_file}")
    
    def process_document(self, json_path: str, output_path: str):
        """完整处理流程"""
        self.load_json(json_path)
        self.extract_text_blocks()
        self.vectorize_blocks()
        self.save_results(output_path)
        return self

def main():
    """主函数"""
    # 获取当前脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置路径
    model_path = os.path.join(script_dir, "models", "sentence-transformers_all-mpnet-base-v2")
    input_json = os.path.join(script_dir, "lectures", "introduction", "introduction_Mathematical_Representation.json")
    output_json = os.path.join(script_dir, "lectures", "introduction", "introduction_Mathematical_Representation_vectorized.json")
    
    print("=" * 50)
    print("极简版JSON文档向量化系统")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print(f"输入文件: {input_json}")
    print(f"输出文件: {output_json}")
    print()
    
    try:
        processor = JSONVectorizer(model_path)
        processor.process_document(input_json, output_json)
        
        print("\n处理摘要:")
        print(f"- 文档标题: {processor.document['metadata']['document_title']}")
        print(f"- 文本块数量: {len(processor.text_blocks)}")
        print(f"- 向量维度: {processor.vectors.shape[1]}")
        
        print("\n✨ 处理完成! ✨")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        logger.exception("处理失败")

if __name__ == "__main__":
    main()
