import os
import sys
import json
import time
import numpy as np
import torch
from flask import Flask, request, Response, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from flask_cors import CORS

# 配置常量
QWEN_MODEL_PATH = "/home/wangzy/zjuc/models/Qwen3-32B"  # 本地Qwen3-32B模型路径
VECTOR_DB_INDEX_PATH = "/home/wangzy/zjuc/agent/lectures/introduction/introduction_Mathematical_Representation_db/index.faiss"  # FAISS索引文件路径
VECTOR_DB_METADATA_PATH = "/home/wangzy/zjuc/agent/lectures/introduction/introduction_Mathematical_Representation_db/metadata.json"  # 元数据文件路径
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/home/wangzy/zjuc/agent/models'
VECTOR_MODEL_PATH = "./models/sentence-transformers_all-mpnet-base-v2"  # 本地文本向量化模型路径
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# 初始化Flask应用
app = Flask(__name__, template_folder='/home/wangzy/zjuc/agent/templates/')
CORS(app)
app.config['JSON_AS_ASCII'] = False  # 禁用ASCII编码，修复中文显示问题
start_time = time.time()

# 修复中文Unicode编码问题的响应函数
def json_response(data, status=200):
    """返回正确编码的JSON响应（解决中文显示问题）"""
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype='application/json; charset=utf-8'
    )

# 日志配置
def log(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# GPU工具函数
def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        log("GPU显存已清理")

def get_gpu_status():
    """获取GPU状态信息"""
    status = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            status.append({
                "设备": f"cuda:{i}",
                "名称": torch.cuda.get_device_name(i),
                "内存(GB)": {
                    "总量": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "已分配": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                    "已预留": round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                }
            })
    return status

# 文本向量化封装
class TextVectorizer:
    """本地文本向量化模型封装"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载本地向量化模型"""
        log(f"⏳ 正在加载文本向量化模型: {self.model_path}")
        try:
            self.model = SentenceTransformer(self.model_path, local_files_only=True, device="cuda")
            log(f"✅ 文本向量化模型加载成功! 设备: {self.model.device}")
        except Exception as e:
            log(f"❌ 文本向量化模型加载失败: {str(e)}", "ERROR")
            raise
    
    def vectorize(self, text: str) -> np.ndarray:
        """将文本转换为向量"""
        log(f"向量化文本: {text[:50]}...")
        if not self.model:
            raise RuntimeError("向量化模型未初始化")
        
        # 确保输入是字符串
        if not isinstance(text, str):
            text = str(text)
        
        # 向量化
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.reshape(1, -1)  # 返回(1, dim)形状

# 向量数据库封装
class VectorDatabase:
    """本地向量数据库封装"""
    
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self.向量维度 = None
        
        self.加载数据库()
    
    def 加载数据库(self):
        """加载向量数据库"""
        log(f"⏳ 正在加载向量数据库: {self.index_path}")
        
        # 加载FAISS索引
        try:
            self.index = faiss.read_index(self.index_path)
            self.向量维度 = self.index.d
            log(f"✅ FAISS索引加载成功! 维度: {self.向量维度}")
        except Exception as e:
            log(f"❌ FAISS索引加载失败: {str(e)}", "ERROR")
            raise
        
        # 加载元数据
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            log(f"✅ 元数据加载成功! 数量: {len(self.metadata)}")
        except Exception as e:
            log(f"❌ 元数据加载失败: {str(e)}", "ERROR")
            raise
    
    def 搜索(self, 查询向量: np.ndarray, k: int = 5) -> list:
        """在数据库中搜索相似项"""
        log(f"执行搜索: k={k}")
        # 输入验证
        if 查询向量.shape[1] != self.向量维度:
            raise ValueError(f"查询向量维度错误。数据库维度: {self.向量维度}, 输入维度: {查询向量.shape[1]}")
        
        # 执行搜索
        try:
            距离, 索引 = self.index.search(查询向量, k)
        except Exception as e:
            log(f"❌ 向量搜索失败: {str(e)}", "ERROR")
            return []
        
        # 处理结果
        结果 = []
        if 索引.size == 0:
            log("⚠️ 搜索未返回任何结果", "WARNING")
            return 结果
            
        for i in range(len(索引[0])):
            索引值 = 索引[0][i]
            if 索引值 < 0 or 索引值 >= len(self.metadata):
                continue
                
            元数据 = self.metadata[索引值]
            
            # 构建结果对象
            结果项 = {
                "相似度": float(距离[0][i]),
                "文档标题": 元数据.get('doc_title', '无标题'),
                "章节": 元数据.get("section", "未知章节"),
                "内容": 元数据.get("text", ""),
                "源索引": int(索引值)
            }
            结果.append(结果项)
        
        return 结果
    
    def 健康检查(self) -> dict:
        """检查数据库健康状况"""
        return {
            "索引已加载": self.index is not None,
            "元数据已加载": bool(self.metadata),
            "元数据数量": len(self.metadata) if self.metadata else 0,
            "向量维度": self.向量维度
        }

# Qwen模型封装（支持多GPU）
class QwenModel:
    """本地Qwen3-32B模型封装（支持双GPU）"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device_map = None
        self.device_count = torch.cuda.device_count()
        
        self.加载模型()
    
    def 加载模型(self):
        """加载Qwen模型和分词器（支持多GPU）"""
        log(f"⏳ 正在加载Qwen模型 ({self.device_count} GPUs)...")
        start_time = time.time()
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 创建基础模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map=None  # 先不分配设备
            )
            
            # 多GPU设备映射
            if self.device_count > 1:
                log(f"🔀 分配模型到 {self.device_count} 个GPU...")
                
                # 计算设备内存分布
                max_memory = get_balanced_memory(
                    model,
                    no_split_module_classes=model._no_split_modules,
                    dtype=model.dtype
                )
                
                # 创建设备映射
                self.device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=model._no_split_modules
                )
                
                # 将模型分发到多个设备
                self.model = dispatch_model(
                    model,
                    device_map=self.device_map,
                    main_device=0  # 主设备为第一个GPU
                )
                
                log(f"✅ 模型已分配到设备: {self.device_map}")
            else:
                # 单GPU情况
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.model = model.to(device)
                log(f"✅ 模型已加载到设备: {device}")
            
            load_time = time.time() - start_time
            log(f"加载耗时: {load_time:.1f}秒")
        except Exception as e:
            log(f"❌ 模型加载失败: {str(e)}", "ERROR")
            raise
    
    def 生成(self, prompt: str, max_length: int = 1024, temperature: float = 0.7) -> str:
        """使用Qwen模型生成文本"""
        log(f"生成响应: max_length={max_length}")
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未正确初始化")
        
        # 编码输入并移动到主设备
        device = self.主设备
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    @property
    def 主设备(self):
        """获取主设备"""
        if self.device_count > 0:
            return "cuda:0"
        return "cpu"
    
    def 健康检查(self) -> dict:
        """检查模型健康状况"""
        return {
            "模型已加载": self.model is not None,
            "分词器已加载": self.tokenizer is not None,
            "设备数量": self.device_count,
            "设备映射": self.device_map,
            "主设备": self.主设备
        }

# AI助教智能体
class AI助教:
    """AI助教智能体核心类"""
    
    def __init__(self, 向量数据库: VectorDatabase, 向量化器: TextVectorizer, qwen模型: QwenModel):
        self.向量数据库 = 向量数据库
        self.向量化器 = 向量化器
        self.qwen模型 = qwen模型
    
    def 检索相关内容(self, 查询: str, top_k: int = 5) -> list:
        """从知识库检索相关内容"""
        # 向量化查询文本
        try:
            查询向量 = self.向量化器.vectorize(查询)
            log(f"🔢 查询向量化成功, 维度: {查询向量.shape}")
        except Exception as e:
            log(f"❌ 向量化失败: {str(e)}", "ERROR")
            return []
        
        # 执行搜索
        结果 = self.向量数据库.搜索(查询向量, top_k)
        
        # 检查结果是否有效
        if not 结果:
            log("⚠️ 搜索未返回任何结果", "WARNING")
            return []
            
        log(f"🔍 找到 {len(结果)} 个相关片段")
        return 结果
    
    def 生成响应(self, 查询: str, 上下文: list, max_length: int = 1024) -> str:
        """基于上下文生成响应"""
        # 构建提示
        提示 = self.构建提示(查询, 上下文)
        
        # 生成响应
        log(f"🧠 开始生成响应: {查询} (max_length={max_length})")
        try:
            log("⚡ 使用Qwen生成响应...")
            start_time = time.time()
            响应 = self.qwen模型.生成(提示, max_length, temperature=0.7)
            gen_time = time.time() - start_time
            log(f"✅ 响应生成完成 (耗时: {gen_time:.2f}秒)")
            return 响应
        except Exception as e:
            log(f"❌ 响应生成失败: {str(e)}", "ERROR")
            return "抱歉，生成响应时出错"
    
    def 构建提示(self, 查询: str, 上下文: list) -> str:
        """构建提示模板"""
        提示 = "你是一个专业的AI助教，请基于以下知识库内容回答学生的问题。\n\n"
        提示 += f"学生问题: {查询}\n\n"
        提示 += "相关参考资料:\n"
        
        for i, 项 in enumerate(上下文):
            提示 += f"[参考资料 {i+1}]\n"
            提示 += f"标题: {项['文档标题']}\n"
            提示 += f"章节: {项['章节']}\n"
            提示 += f"内容: {项['内容'][:300]}...\n\n"
        
        提示 += "\n请根据以上资料，用专业、简洁的语言回答学生问题。"
        return 提示

    def respond_to_query(self, query: str, max_length=1000, top_k=5) -> dict:
        """处理用户查询并生成响应"""
        # 1. 检索相关内容
        relevant_content = self.检索相关内容(query, top_k)
        
        # 2. 生成AI助教回答
        ai_response = self.生成响应(query, relevant_content, max_length)
        
        # 3. 构建响应结构
        return {
            "query": query,
            "response": ai_response,
            "sources": [
                {
                    "doc_title": item.get("文档标题", "无标题"),
                    "section": item.get("章节", "未知章节"),
                    "page": item.get("page_idx", 0),
                    "similarity": item.get("相似度", 0.0),
                    "text_excerpt": item.get("内容", "")[:300]
                }
                for item in relevant_content
            ]
        }

# 全局智能体实例
助教 = None

@app.route('/')
def home():
    """提供前端界面"""
    return render_template('agent.html')

# 在提问端点添加参数支持

@app.route('/ask', methods=['POST'])
def handle_ask():
    try:
        # 获取JSON数据
        data = request.get_json()
        if not data or 'query' not in data:
            return json_response({"error": "缺少查询参数"}, 400)
        
        # 提取参数
        question = data['query']
        max_length = data.get('max_length', 1000)
        top_k = data.get('top_k', 5)
        
        # 确保助教实例已初始化
        global 助教
        if not 助教:
            return json_response({"error": "AI助教未初始化"}, 500)
        
        # 使用AI助教生成响应
        response_data = 助教.respond_to_query(question, max_length, top_k)
        
        return json_response(response_data)
    except Exception as e:
        log(f"处理提问时出错: {str(e)}\n{traceback.format_exc()}", "ERROR")
        return json_response({"error": "内部服务器错误"}, 500)

# def _build_cors_preflight_response():
#     response = Response()
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     response.headers.add("Access-Control-Allow-Headers", "Content-Type")
#     response.headers.add("Access-Control-Allow-Methods", "POST")
#     return response

# 在健康检查端点添加更多状态信息
@app.route('/health')
def health_check():
    try:
        # 获取模型信息
        model_info = {
            "model_loaded": assistant.qwen is not None,
            "model_device": assistant.qwen.device if assistant.qwen else None,
            "model_params": str(assistant.qwen.model.config) if assistant.qwen else None
        }
        
        # 获取向量数据库信息
        vector_db_info = {
            "vector_db_loaded": assistant.vector_db is not None,
            "vector_count": len(assistant.vector_db.metadata) if assistant.vector_db else 0
        }
        
        return json_response({
            "status": "healthy",
            "version": "1.0.0",
            "uptime": f"{time.time() - start_time:.2f} seconds",
            "model": model_info,
            "vector_db": vector_db_info
        })
    except Exception as e:
        return json_response({
            "status": "error",
            "message": str(e)
        }, 500)

@app.route('/gpu-status', methods=['GET'])
def gpu状态():
    """GPU状态端点"""
    return json_response(get_gpu_status())

# 初始化函数
def 初始化助教():
    """初始化AI助教智能体"""
    global 助教
    
    try:
        # 清理显存
        clear_gpu_memory()
        
        # 初始化组件
        log("🚀 正在初始化文本向量化模型...")
        向量化器 = TextVectorizer(VECTOR_MODEL_PATH)
        
        log("🚀 正在初始化向量数据库...")
        向量数据库 = VectorDatabase(VECTOR_DB_INDEX_PATH, VECTOR_DB_METADATA_PATH)
        
        log("🚀 正在初始化Qwen3-32B模型...")
        qwen模型 = QwenModel(QWEN_MODEL_PATH)
        
        # 创建智能体
        助教 = AI助教(向量数据库, 向量化器, qwen模型)
        log("🎉 AI助教智能体初始化完成!")
        
        # 返回成功消息
        return True
    except Exception as e:
        log(f"❌ AI助教初始化失败: {str(e)}", "ERROR")
        return False

# 主函数
if __name__ == '__main__':
    try:
        # 初始化智能体
        if not 初始化助教():
            log("❌ AI助教初始化失败，无法启动服务", "ERROR")
            sys.exit(1)
        
        # 启动服务器
        app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)
    except Exception as e:
        log(f"❌ 系统启动失败: {str(e)}", "ERROR")
        sys.exit(1)