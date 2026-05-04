# -rag-enterprise-knowledgebase-
`企业级RAG知识库问答系统，基于LangChain+通义千问，求职演示项目`
## 📌 项目定位
面向企业场景的RAG（检索增强生成）知识库问答系统，适配阿里云通义千问生态，可作为LLM应用开发能力的演示项目，核心体现「数据处理→向量检索→大模型调用→前端交互」全链路开发能力。

## 🚀 核心技术亮点（面试重点）
### 1. 自定义核心模块（手写实现，体现底层理解）
- 自定义通义千问Embedding类（适配阿里云DashScope API）
- 自定义余弦相似度检索（替代原生检索，体现算法基础）
- 增强型文本切分（适配中文语义，处理乱码/特殊格式）
- 强约束Prompt模板（保证回答规范性，适配企业场景）

### 2. 工程化能力
- 多格式文件解析（TXT/PDF/DOCX），自动编码检测（解决乱码痛点）
- 前端交互优化（进度条/分Tab/样式定制），提升用户体验
- 异常处理+友好提示，适配生产级场景的鲁棒性

### 3. 技术栈
- 框架：LangChain（RAG架构）、Streamlit（前端）
- 向量库：FAISS（轻量高效）
- 大模型：通义千问（qwen-turbo）
- 辅助：chardet（编码检测）、PyPDF2/python-docx（文件解析）

## 🔧 快速启动（面试演示步骤）
### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/你的用户名/rag-enterprise-knowledgebase.git
cd rag-enterprise-knowledgebase

# 安装依赖
pip install -r requirements.txt
