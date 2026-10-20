# 英入眼帘 · English Learning Agent

一个英语学习智能体 Web 应用。输入新学的英文单词，自动生成覆盖全部词汇的英文短文、中文翻译、朗读音频、主题插画与阅读理解题。

---

## ✨ 核心功能

1. **词汇清洗**：去空白、去重、支持英文大小写，检出非法字符
2. **双文生成**：根据词汇语义生成 2 篇连贯英文短文，覆盖全部目标词汇
3. **覆盖检测**：word boundary 匹配并支持常见词形变化（run/running、achieve/achieved 均算覆盖；`art` 不会被 `article` 错误匹配）
4. **自动重试**：覆盖 < 100% 时自动重新生成，最多 3 次
5. **诚实报告**：显示缺失单词，绝不伪造 100%
6. **中文翻译**：两篇文章翻译为自然中文，残留英文自动重译
7. **语音朗读**：优先 Edge TTS，失败时浏览器 SpeechSynthesis 兜底
8. **主题插画**：每篇文章自动生成一张与主题相符的简约编辑插画风图片（智谱 CogView），标注图中出现的输入词，抽象概念词给出提示
9. **阅读理解**：每篇文章 3 道雅思风格选择题（主旨/细节/推断），提交后显示答案与解析
10. **学习记录**：SQLite 保存全部学习数据，答题记录独立保存可回溯
11. **历史查看 & 重新生成 & 分享 & 报告导出**

---

## 🏗 工作流程

`EnglishLearningAgent` 作为核心协调器，编排以下模块（纯 Python 类实现）：

```
VocabularyAnalyzer → ArticleGenerator → ContentEvaluator
                   → Translator → QuestionGenerator → TTS → ImageGenerator → LearningRecord
```

---

## 📦 安装

> 环境要求：Python 3.8+

```bash
cd ai-english-agent
pip install -r requirements.txt
```

> 如果系统目录无写入权限，可安装到项目本地：
> ```bash
> pip install --target=_vendor -r requirements.txt
> ```
> （`app.py` 已自动将 `_vendor` 加入 `sys.path`）

---

## ⚙️ 配置

复制 `.env.example` 为 `.env` 并填写 API Key：

```bash
cp .env.example .env
```

编辑 `.env`（默认使用智谱 AI，支持任何 OpenAI 兼容格式的 API）：

```ini
# 智谱 AI（文本 + 图片生成）
#   注册：https://open.bigmodel.cn
QWEN_API_KEY=your_api_key_here
QWEN_API_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
QWEN_MODEL=glm-4

# 智谱图片生成（CogView）
ZHIPU_IMAGE_URL=https://open.bigmodel.cn/api/paas/v4/images/generations
ZHIPU_IMAGE_MODEL=cogview-3

# 服务端口（默认 5000，可修改）
PORT=5000
```

**安全说明**：API Key 仅在服务端读取，不会暴露到前端。`.env` 已被 `.gitignore` 排除，请勿提交到仓库。

---

## 🚀 启动

```bash
python app.py
```

启动后访问：

- 首页：http://127.0.0.1:5000/
- 学习页：http://127.0.0.1:5000/learning
- 历史记录：http://127.0.0.1:5000/history
- 关于：http://127.0.0.1:5000/about

---

## 📋 使用流程

1. 在首页输入 5 个以上英文单词（空格/逗号/换行分隔），确认拼写后点击「开始生成」
2. 系统根据词汇语义生成文章并自动检测覆盖率，不足 100% 自动重试（最多 3 次）
3. 在学习页查看：
   - 两篇英文文章（目标词汇高亮）+ 中文翻译
   - 点击播放按钮收听英文朗读，可调节语速
   - 完成 6 道阅读理解选择题，提交后查看答案与解析
4. 可重新生成或导出学习报告
5. 所有记录自动保存到历史记录页面，可查看详情、分享或删除

---

## 🗄 数据存储

使用 SQLite（`data/learning.db`），每条记录包含：

| 字段 | 说明 |
|------|------|
| created_at | 生成时间 |
| words | 目标单词 |
| articles | 两篇英文文章 |
| translations | 两篇中文翻译 |
| questions | 6 道阅读理解题及答案 |
| coverage | 词汇覆盖率 |
| matched_words | 已覆盖单词 |
| missing_words | 缺失单词 |
| audio_paths | 音频文件路径 |
| demo_mode | 是否演示模式 |

---

## 🛠 技术栈

- **后端**：Python + Flask
- **大语言模型**：智谱 AI GLM（支持任意 OpenAI 兼容格式）
- **图片生成**：智谱 CogView-3
- **语音朗读**：edge-tts（主）/ 浏览器 SpeechSynthesis（兜底）
- **数据库**：SQLite
- **前端**：HTML / CSS / JavaScript（原生，无框架）

---

## 📁 项目结构

```
ai-english-agent/
├── app.py              # Flask 应用，路由与 API
├── agent.py            # EnglishLearningAgent 核心协调器（含图片生成）
├── db.py               # SQLite 数据库
├── report.py           # 学习报告导出（Markdown/PDF）
├── requirements.txt
├── .gitignore
├── .env / .env.example
├── templates/          # HTML 模板
│   ├── base.html
│   ├── index.html      # 首页
│   ├── learning.html   # 学习页
│   ├── history.html    # 历史记录
│   ├── about.html      # 关于
│   └── share.html      # 分享页
├── static/
│   ├── css/style.css
│   └── js/app.js
├── audio/              # 生成的语音音频（gitignore）
├── images/             # 生成的主题插画（gitignore）
└── data/               # SQLite 数据库（gitignore）
```
