"""AI English Learning Agent — 核心协调器。

流水线：
    VocabularyAnalyzer -> ArticleGenerator -> ContentEvaluator
    -> Translator -> QuestionGenerator -> TTS -> LearningRecord

不依赖 LangChain/LangGraph，纯 Python 类实现。
"""
import os
import re
import json
import asyncio
import logging
import time
from typing import List, Dict, Tuple, Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

logger = logging.getLogger(__name__)

# ---------- 配置 ----------
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "glm-4")

# 智谱图片生成（CogView）
ZHIPU_IMAGE_URL = os.getenv(
    "ZHIPU_IMAGE_URL",
    "https://open.bigmodel.cn/api/paas/v4/images/generations",
)
ZHIPU_IMAGE_MODEL = os.getenv("ZHIPU_IMAGE_MODEL", "cogview-3")

AUDIO_SPEED = float(os.getenv("AUDIO_SPEED", "1.0"))

SEP = "===ARTICLE_SEPARATOR==="
MAX_RETRY = 3
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# 1. Vocabulary Analyzer
# ============================================================
class VocabularyAnalyzer:
    """清洗、去重、校验输入单词。"""

    @staticmethod
    def clean(words: List[str]) -> Tuple[List[str], List[str]]:
        cleaned, invalid, seen = [], [], set()
        for w in words:
            w = (w or "").strip()
            if not w:
                continue
            if re.match(r"^[a-zA-Z]+$", w):
                low = w.lower()
                if low not in seen:
                    seen.add(low)
                    cleaned.append(w)  # 保留原始大小写
            else:
                invalid.append(w)
        return cleaned, invalid


# ============================================================
# 2. Article Generator
# ============================================================
class ArticleGenerator:
    """调用大语言模型生成 2 篇覆盖全部目标词汇的英文短文。"""

    @staticmethod
    def _call(prompt: str) -> str:
        if not QWEN_API_KEY:
            raise RuntimeError("QWEN_API_KEY not configured")
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }
        # 判断是否为 OpenAI 兼容格式（chat/completions）
        # 支持：DashScope 兼容模式、SiliconFlow、DeepSeek、OpenRouter、Moonshot、智谱等
        is_openai_compatible = any(
            kw in QWEN_API_URL for kw in ["chat/completions", "compatible-mode", "dashscope"]
        )
        if is_openai_compatible:
            payload = {
                "model": QWEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=90, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            raise RuntimeError(f"API status {resp.status_code}: {resp.text[:300]}")
        else:
            # 阿里云原始格式（input.messages）
            payload = {
                "model": QWEN_MODEL,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {"temperature": 0.7},
            }
            resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=90, verify=False)
            if resp.status_code == 200:
                return resp.json()["output"]["text"]
            raise RuntimeError(f"API status {resp.status_code}: {resp.text[:300]}")

    @staticmethod
    def generate(words: List[str]) -> List[str]:
        prompt = (
            "You are an experienced English teacher and writer. "
            "Write TWO coherent, natural English articles for intermediate English learners.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. The two articles TOGETHER must contain ALL target words: "
            f"{', '.join(words)}.\n"
            "2. The two articles MUST be clearly DIFFERENT in theme, setting and genre — "
            "e.g. one personal narrative / story, the other an expository or opinion piece. "
            "Different places, different protagonists, different opening scenes. "
            "They must NOT read like two versions of the same story.\n"
            "3. Distribute the target words across the two articles by semantic fit — "
            "each word goes into the article where it fits most naturally — but do NOT "
            "split them mechanically half-and-half, and do NOT force all words into one article.\n"
            "4. Each article must be a REAL article with a clear topic, logical flow, "
            "and varied sentence structures — NOT a list of disconnected sentences.\n"
            "5. Use each target word naturally in context, as a native speaker would. "
            "Do NOT put words in quotes or repeat the same sentence template.\n"
            "6. Use the EXACT word forms given (do not change tense/plural unless the word itself is plural).\n"
            "7. Each article should be 120-200 words, engaging and easy to understand.\n"
            "8. Return ONLY the two articles, separated by exactly one line containing: "
            f"{SEP}\n"
            "No extra explanation, no numbering, no markdown code fences."
        )
        try:
            text = ArticleGenerator._call(prompt)
            articles = [a.strip() for a in text.split(SEP) if a.strip()]
            if len(articles) < 2:
                articles = [a.strip() for a in re.split(r"\n\s*\n", text) if a.strip()][:2]
            if len(articles) < 2:
                articles = ArticleGenerator._fallback(words)
            return articles[:2]
        except Exception as e:
            logger.warning("Article generation failed: %s", e)
            return ArticleGenerator._fallback(words)

    @staticmethod
    def _fallback(words: List[str]) -> List[str]:
        """无 API 时的本地文章生成器：基于单词语义构建两篇有逻辑、句式多样的短文。"""
        # 句式模板库 —— 避免重复，让文章有自然的语流
        templates = [
            "When we talk about {w}, it is impossible to ignore its role in everyday life.",
            "Many people are surprised to learn how deeply {w} affects our daily routines.",
            "The concept of {w} has evolved significantly over the past few decades.",
            "If you have ever wondered about {w}, you are not alone in this curiosity.",
            "One of the most fascinating aspects of {w} is how it connects to other ideas.",
            "Experts in the field often emphasize the importance of {w} in modern society.",
            "Understanding {w} can open doors to new perspectives and opportunities.",
            "It is worth noting that {w} plays a surprisingly large part in how we communicate.",
        ]
        connectors = [
            "Furthermore,", "In addition,", "Moreover,", "Beyond this,",
            "What makes it even more interesting is that", "Interestingly,",
            "As a matter of fact,", "From another angle,",
        ]
        closings = [
            "In conclusion, these ideas show how interconnected our world truly is.",
            "All in all, exploring these topics reveals the richness of human experience.",
            "Ultimately, gaining a deeper understanding of these concepts enriches our lives.",
            "Taken together, these reflections remind us of the complexity of language and thought.",
        ]

        def build_article(word_list: List[str]) -> str:
            import random
            rng = random.Random(hash(tuple(word_list)))
            sentences = []
            used_tpl = set()
            connector_starts = ("when ", "if ", "many ", "the concept", "one of ",
                                "experts ", "understanding ", "it is worth", "what makes",
                                "interestingly", "as a matter", "from another")
            for i, w in enumerate(word_list):
                available = [t for t in templates if t not in used_tpl]
                if not available:
                    available = templates
                    used_tpl.clear()
                tpl = rng.choice(available)
                used_tpl.add(tpl)
                sent = tpl.format(w=w)
                if i > 0 and not sent.lower().startswith(connector_starts):
                    sent = rng.choice(connectors) + " " + sent[0].lower() + sent[1:]
                sentences.append(sent)
            sentences.append(rng.choice(closings))
            return " ".join(sentences)

        # 按语义分组：简单地交错分配，使两篇都有词汇覆盖面
        half = max(1, (len(words) + 1) // 2)
        w1 = [words[i] for i in range(len(words)) if i % 2 == 0][:half] or words[:half]
        w2 = [w for w in words if w not in w1] or words[half:]
        if not w2:
            w2 = words[half:] or words
        return [build_article(w1), build_article(w2)]


# ============================================================
# 3. Content Evaluator
# ============================================================
class ContentEvaluator:
    """Vocabulary Coverage 检测（word boundary 匹配，支持常见词形变化）。

    文章中出现目标词的合理变形（复数、过去式、-ing、比较级等）即视为覆盖，
    避免 "run" 写成 "running" 时被误判为缺失。
    """

    @staticmethod
    def variants(w: str) -> List[str]:
        """生成单词的常见变形（长词在前，优先匹配）。"""
        w = w.lower()
        v = {w}
        n = len(w)
        if n >= 2:
            is_vowel = lambda c: c in "aeiou"
            # 复数 / 第三人称单数
            if w.endswith(("s", "x", "z", "ch", "sh")):
                v.add(w + "es")
            elif w.endswith("y") and not is_vowel(w[-2]):
                v.add(w[:-1] + "ies")
            else:
                v.add(w + "s")
            # 过去式
            if w.endswith("e"):
                v.add(w + "d")
            elif w.endswith("y") and not is_vowel(w[-2]):
                v.add(w[:-1] + "ied")
            else:
                v.add(w + "ed")
                # CVC 双写尾字母（stop→stopped）
                if n >= 3 and not is_vowel(w[-1]) and w[-1] not in "ywx" \
                        and is_vowel(w[-2]) and not is_vowel(w[-3]):
                    v.add(w + w[-1] + "ed")
            # -ing
            if w.endswith("ie"):
                v.add(w[:-2] + "ying")
            elif w.endswith("e") and not w.endswith("ee"):
                v.add(w[:-1] + "ing")
            else:
                v.add(w + "ing")
                if n >= 3 and not is_vowel(w[-1]) and w[-1] not in "ywx" \
                        and is_vowel(w[-2]) and not is_vowel(w[-3]):
                    v.add(w + w[-1] + "ing")  # run→running
            # 比较级 / 最高级
            if w.endswith("e"):
                v.add(w + "r")
                v.add(w + "st")
            elif w.endswith("y") and not is_vowel(w[-2]):
                v.add(w[:-1] + "ier")
                v.add(w[:-1] + "iest")
            else:
                v.add(w + "er")
                v.add(w + "est")
            # 副词 -ly（形容词目标词常以副词形式出现：happy→happily）
            if w.endswith("y") and not is_vowel(w[-2]):
                v.add(w[:-1] + "ily")
            elif w.endswith("le"):
                v.add(w[:-1] + "y")
            else:
                v.add(w + "ly")
        return sorted(v, key=len, reverse=True)

    @staticmethod
    def _pattern(w: str) -> str:
        alts = "|".join(re.escape(x) for x in ContentEvaluator.variants(w))
        return r"\b(?:" + alts + r")\b"

    @staticmethod
    def evaluate(words: List[str], articles: List[str]) -> Dict:
        all_text = " ".join(articles).lower()
        matched, missing, stats = [], [], []
        for w in words:
            cnt = len(re.findall(ContentEvaluator._pattern(w), all_text))
            stats.append({"word": w, "found": cnt > 0, "count": cnt})
            if cnt > 0:
                matched.append(w)
            else:
                missing.append(w)
        coverage = round(len(matched) / len(words) * 100, 2) if words else 0.0
        return {
            "coverage": coverage,
            "matched_words": matched,
            "missing_words": missing,
            "word_stats": stats,
            "total": len(words),
        }


# ============================================================
# 4. Translator
# ============================================================
class Translator:
    """将两篇英文文章翻译为自然中文（逐篇独立翻译，单篇失败互不影响）。"""

    FALLBACK = "（翻译暂时不可用，请稍后重试）"

    @staticmethod
    def translate(articles: List[str]) -> List[str]:
        out = [Translator._translate_one(a) for a in articles[:2]]
        while len(out) < 2:
            out.append(Translator.FALLBACK)
        return out[:2]

    @staticmethod
    def _has_english(text: str) -> bool:
        # 检测译文中是否残留连续 2 个及以上英文字母的单词（允许单个字母如 A）
        return bool(re.search(r"[A-Za-z]{2,}", text))

    @staticmethod
    def _translate_one(article: str) -> str:
        prompt = (
            "把下面这篇英文文章完整翻译成自然、流畅的中文。\n"
            "严格要求：\n"
            "1. 译文中不得残留任何英文单词或词组，每一个英文词都必须翻译成中文"
            "（约定俗成的专有缩写如 UNESCO、DNA 除外，需在其后加括号中文注释）；\n"
            "2. 逐句完整翻译，不得漏译、跳译或省略任何句子；\n"
            "3. 保留原文的标题与段落结构，每段对应翻译；\n"
            "4. 只输出中文译文，不要输出任何解释或英文原文。\n\n"
            f"{article}"
        )
        try:
            text = ArticleGenerator._call(prompt).strip()
            if not text:
                return Translator.FALLBACK
            # 若译文中残留英文单词，带纠错要求自动重试一次
            if Translator._has_english(text):
                logger.info("Translation contains English words, retrying once")
                retry_prompt = prompt + (
                    "\n\n注意：你上一次的译文中残留了未翻译的英文单词，"
                    "请重新翻译，确保译文中除 UNESCO、DNA 等公认缩写及其括号注释外，"
                    "不出现任何英文字母组成的单词。"
                )
                text2 = ArticleGenerator._call(retry_prompt).strip()
                if text2:
                    text = text2
            return text
        except Exception as e:
            logger.warning("Translation failed: %s", e)
            return Translator.FALLBACK


# ============================================================
# 5. Question Generator
# ============================================================
class QuestionGenerator:
    """基于两篇文章生成阅读理解选择题（每篇 3 题，共 6 题，雅思风格）。"""

    @staticmethod
    def generate(articles: List[str], words: List[str]) -> List[Dict]:
        questions = []
        for ai, art in enumerate(articles[:2], start=1):
            prompt = (
                "You are an IELTS reading test designer. Based on the following English article, "
                f"generate EXACTLY 3 multiple-choice reading comprehension questions for Article {ai}.\n"
                "IELTS-style requirements:\n"
                "1. Question 1: Main idea / gist question (what is the article mainly about).\n"
                "2. Question 2: Specific detail question (find a fact stated in the text).\n"
                "3. Question 3: Inference / implication question (what can be inferred).\n"
                "- Each question has 4 options, exactly ONE correct answer.\n"
                "- Options must be plain text WITHOUT any letter prefix (no A. B. C. D. at the start).\n"
                "- Distractors must be plausible but NOT supported by the text.\n"
                "- Questions must be answerable ONLY from the article content.\n"
                "- Provide a brief explanation in Chinese (中文) citing the relevant part of the text.\n"
                "Return ONLY a JSON array with this exact structure, no markdown fences:\n"
                '[{"question":"...","options":["option text","option text","option text","option text"],'
                '"answer":0,"explanation":"中文解析"}]\n\n'
                f"Article {ai}:\n{art}"
            )
            try:
                text = ArticleGenerator._call(prompt)
                text = re.sub(r"^```(?:json)?\s*", "", text.strip())
                text = re.sub(r"\s*```$", "", text)
                qs = json.loads(text)
                for q in qs[:3]:
                    # 清理选项：去除可能的字母前缀（A. B. C. D.）
                    raw_opts = [str(o).strip() for o in q.get("options", [])][:4]
                    clean_opts = [re.sub(r"^[A-D][\.\)、]\s*", "", o) for o in raw_opts]
                    questions.append({
                        "question": str(q.get("question", "")).strip(),
                        "options": clean_opts,
                        "answer": int(q.get("answer", 0)),
                        "explanation": str(q.get("explanation", "")).strip(),
                        "article_index": ai,
                    })
            except Exception as e:
                logger.warning("Question generation for article %d failed: %s", ai, e)
                questions.extend(QuestionGenerator._fallback_for_article(art, ai, words))
        # 确保正好 6 题
        if len(questions) < 6:
            for ai in range(1, 3):
                if len([q for q in questions if q.get("article_index") == ai]) < 3:
                    questions.extend(QuestionGenerator._fallback_for_article(articles[ai - 1], ai, words))
        return questions[:6]

    @staticmethod
    def _fallback_for_article(article: str, article_index: int, words: List[str]) -> List[Dict]:
        """为单篇文章生成 3 道基于内容的雅思风格题目（fallback）。"""
        # 提取文章中的关键信息用于出题
        sents = re.split(r"(?<=[.!?])\s+", article.strip())
        sents = [s for s in sents if len(s) > 10]
        first_sent = sents[0] if sents else article
        # 提取文章中的实词作为干扰项来源
        content_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", article.lower()))
        target_in_article = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", article.lower())]

        # 第 1 题：主旨题
        main_topic = target_in_article[0] if target_in_article else (content_words.pop() if content_words else "the topic")
        q1 = {
            "question": f"What is the main idea of Article {article_index}?",
            "options": [
                f"The article mainly discusses {main_topic} and its significance.",
                f"The article argues that {main_topic} should be banned completely.",
                f"The article provides a historical timeline of {main_topic}.",
                f"The article compares {main_topic} with ancient civilizations.",
            ],
            "answer": 0,
            "explanation": f"文章围绕 {main_topic} 展开，说明其作用和重要性，而非主张禁止、历史年代考证或与古代文明比较。",
            "article_index": article_index,
        }

        # 第 2 题：细节题 —— 基于第一句
        q2 = {
            "question": f"According to Article {article_index}, which statement is true?",
            "options": [
                first_sent[:80] + ("..." if len(first_sent) > 80 else ""),
                f"The article claims that {main_topic} has no connection to daily life.",
                f"The author states that {main_topic} was discovered in the 15th century.",
                f"The text says {main_topic} is irrelevant to modern society.",
            ],
            "answer": 0,
            "explanation": "这一事实在文章中直接陈述。其他选项与文章内容矛盾或文中未提及。",
            "article_index": article_index,
        }

        # 第 3 题：推断题
        q3 = {
            "question": f"What can be inferred from Article {article_index}?",
            "options": [
                f"The author believes understanding {main_topic} brings practical benefits.",
                f"The author thinks {main_topic} is a passing fashion that will disappear.",
                f"The author suggests everyone should memorize the dictionary.",
                f"The author implies that {main_topic} is harmful to health.",
            ],
            "answer": 0,
            "explanation": "文章的积极说明性语调暗示作者认为理解这一主题是有价值的，而非将其视为有害或无关紧要。",
            "article_index": article_index,
        }
        return [q1, q2, q3]

    @staticmethod
    def _fallback(articles: List[str], words: List[str]) -> List[Dict]:
        """兼容旧接口：为两篇文章各生成 3 题。"""
        questions = []
        for ai, art in enumerate(articles[:2], start=1):
            questions.extend(QuestionGenerator._fallback_for_article(art, ai, words))
        return questions[:6]


# ============================================================
# 6. TTS
# ============================================================
class TTS:
    """优先 Edge TTS，失败返回 None 由前端使用浏览器 SpeechSynthesis fallback。"""

    @staticmethod
    def _rate_to_edge(rate: float) -> str:
        """将 0.5~2.0 的语速转换为 Edge TTS 的 rate 字符串。"""
        pct = int(round((rate - 1.0) * 100))
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct}%"

    @staticmethod
    async def _edge(text: str, path: str, rate: float = 1.0) -> bool:
        try:
            communicate = edge_tts.Communicate(
                text, "en-US-JennyNeural", rate=TTS._rate_to_edge(rate)
            )
            await communicate.save(path)
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception as e:
            logger.warning("Edge TTS failed: %s", e)
            return False

    @staticmethod
    def generate(text: str, output_path: str, rate: float = 1.0) -> bool:
        if not text or not text.strip():
            return False
        if HAS_EDGE_TTS:
            try:
                return asyncio.run(TTS._edge(text, output_path, rate))
            except RuntimeError:
                # 已有事件循环（比如在异步环境），新建线程跑
                import threading
                result = {}

                def _run():
                    loop = asyncio.new_event_loop()
                    try:
                        result["ok"] = loop.run_until_complete(TTS._edge(text, output_path, rate))
                    finally:
                        loop.close()

                t = threading.Thread(target=_run)
                t.start()
                t.join(timeout=60)
                return result.get("ok", False)
            except Exception as e:
                logger.warning("TTS error: %s", e)
        return False


# ============================================================
# 7. Image Generator（智谱 CogView）
# ============================================================
class ImageGenerator:
    """为文章生成主题插画。

    策略（按用户要求调整）：
    - 每篇文章只生成一张与【短文主题】相符的高质量插画
    - 如果画面中能自然融入输入词对应的具象事物则更好（非强制）
    - 抽象概念词不入画，仅在前端给出文字提示
    """

    # 提升审美的核心风格描述：高级编辑插画风
    STYLE = (
        "award-winning editorial flat illustration, modern minimalist vector art, "
        "sophisticated muted color palette with soft warm gradients, "
        "subtle paper grain texture, elegant clean geometric shapes, "
        "generous negative space, balanced harmonious composition, "
        "gentle ambient lighting, high-end magazine cover quality, masterpiece"
    )

    NEGATIVE = (
        "no text, no letters, no words, no numbers, no watermark, no logo, "
        "no frame, no border, no clutter, no photorealism, no 3d render, "
        "no blurry details, no distorted faces"
    )

    # 抽象词/概念词列表（无法直接图示的）
    ABSTRACT_WORDS = {
        "sustenance", "concept", "idea", "knowledge", "wisdom", "truth",
        "freedom", "liberty", "justice", "equality", "love", "hate",
        "happiness", "sadness", "anger", "fear", "courage", "bravery",
        "poverty", "wealth", "success", "failure", "challenge", "difficulty",
        "importance", "significance", "meaning", "purpose", "identity",
        "culture", "tradition", "history", "progress", "change", "future",
        "past", "present", "relationship", "connection", "influence",
        "impact", "effect", "result", "consequence", "cause", "reason",
        "belief", "opinion", "view", "perspective", "attitude", "emotion",
        "feeling", "experience", "memory", "imagination", "creativity",
        "innovation", "technology", "science", "art", "music", "literature",
        "development", "growth", "improvement", "enhancement", "advancement",
    }

    @staticmethod
    def _extract_keywords(article: str, target_words: List[str]) -> Tuple[List[str], List[str]]:
        """从【本篇】文章中提取可图示的目标词与抽象词。

        只保留确实出现在本篇文章里的目标词（含词形变化），
        不再从正文抓取额外名词，避免噪音词污染画面。
        返回 (visual_keywords, abstract_words)
        """
        low_text = article.lower()
        visual, abstract = [], []
        for w in target_words:
            pattern = ContentEvaluator._pattern(w)
            if not re.search(pattern, low_text):
                continue  # 该词不在本篇文章中，跳过
            low = w.lower()
            if low in ImageGenerator.ABSTRACT_WORDS:
                abstract.append(w)
            else:
                visual.append(w)
        return visual[:3], abstract

    @staticmethod
    def _build_prompt(article: str, visual_words: List[str]) -> str:
        """构建图片生成 prompt：以文章主题场景为主，输入词元素为辅。"""
        lines = [l.strip() for l in article.strip().split("\n") if l.strip()]
        # 识别标题（首行较短且无句末标点）
        title = ""
        body_start = 0
        if lines and len(lines[0]) < 100 and not re.search(r"[.!?。！？]$", lines[0]):
            title = lines[0]
            body_start = 1
        body = " ".join(lines[body_start:])
        sents = re.split(r"(?<=[.!?])\s+", body)
        scene_desc = " ".join(sents[:2])[:220]

        prompt = f"{ImageGenerator.STYLE}. "
        if title:
            prompt += f"Illustration theme: \"{title}\". "
        prompt += f"Scene inspired by: {scene_desc}. "
        if visual_words:
            prompt += (
                f"If they fit the scene naturally, subtly weave in these elements: "
                f"{', '.join(visual_words)}. "
            )
        prompt += (
            "One clear focal point, cohesive single scene, not a collage. "
            f"{ImageGenerator.NEGATIVE}."
        )
        return prompt

    @staticmethod
    def _call_cogview(prompt: str) -> Optional[str]:
        """调用智谱 CogView API，返回图片 URL 或 None。
        
        图片生成是附加功能，失败不阻塞主流程。
        """
        if not QWEN_API_KEY:
            return None
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": ZHIPU_IMAGE_MODEL,
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1,
        }
        try:
            # 30 秒超时：CogView 实际耗时较长，过短会导致图片静默失败
            resp = requests.post(ZHIPU_IMAGE_URL, headers=headers, json=payload, timeout=30, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                urls = data.get("data", [])
                if urls and len(urls) > 0:
                    return urls[0].get("url") or urls[0].get("b64_json")
            logger.warning("CogView API status %s: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            # 图片生成失败不应阻塞文章生成，只记录日志
            logger.warning("CogView call failed (non-blocking): %s", str(e)[:100])
        return None

    @staticmethod
    def _download_image(url_or_b64: str, article_idx: int) -> Optional[str]:
        """下载图片到本地，返回相对路径。"""
        try:
            ts = int(time.time())
            filename = f"article_{article_idx}_{ts}.png"
            filepath = os.path.join(IMAGE_DIR, filename)

            # 判断是 URL 还是 base64
            if url_or_b64.startswith("http"):
                resp = requests.get(url_or_b64, timeout=30, verify=False)
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    return f"images/{filename}"
            else:
                # data URI 或裸 base64
                import base64
                b64_data = url_or_b64.split(",", 1)[1] if url_or_b64.startswith("data:") else url_or_b64
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                return f"images/{filename}"
        except Exception as e:
            logger.warning("Image download failed: %s", e)
        return None

    @staticmethod
    def generate(article: str, target_words: List[str], article_idx: int) -> Dict:
        """为单篇文章生成图片 + 标注信息。
        
        返回：
        {
            "image_path": "images/xxx.png" 或 None,
            "visual_words": [...],
            "abstract_words": [...],
            "note": "提示信息（如有）"
        }
        """
        visual_words, abstract_words = ImageGenerator._extract_keywords(article, target_words)
        prompt = ImageGenerator._build_prompt(article, visual_words)

        url = ImageGenerator._call_cogview(prompt)
        image_path = ImageGenerator._download_image(url, article_idx) if url else None

        note = None
        if abstract_words:
            abs_str = "、".join(abstract_words)
            note = f"「{abs_str}」为抽象概念，建议结合语境记忆"

        return {
            "image_path": image_path,
            "visual_words": visual_words,
            "abstract_words": abstract_words,
            "note": note,
        }


# ============================================================
# EnglishLearningAgent — 核心协调器
# ============================================================
class EnglishLearningAgent:
    """协调完整学习流程：词汇 -> 文章 -> 评测 -> 翻译 -> 题目 -> TTS。"""

    def __init__(self, audio_dir: str = "audio"):
        self.audio_dir = audio_dir
        os.makedirs(audio_dir, exist_ok=True)

    def run(self, words: List[str]) -> Dict:
        result: Dict = {
            "words": words,
            "articles": [],
            "translations": [],
            "questions": [],
            "images": [],           # 图片生成结果
            "coverage": 0.0,
            "matched_words": [],
            "missing_words": [],
            "word_stats": [],
            "attempts": [],
            "demo_mode": not bool(QWEN_API_KEY),
            "used_fallback": False,
            "translation_ok": True,
            "audio_status": [],
            "audio_paths": [],
        }

        # Demo 模式
        if not QWEN_API_KEY:
            return self._demo(words)

        # 1. 生成 + 评测 + 重试
        articles = []
        eval_result = {}
        for attempt in range(1, MAX_RETRY + 1):
            articles = ArticleGenerator.generate(words)
            eval_result = ContentEvaluator.evaluate(words, articles)
            result["attempts"].append({
                "attempt": attempt,
                "coverage": eval_result["coverage"],
                "missing_words": eval_result["missing_words"],
            })
            if eval_result["coverage"] >= 100.0:
                break

        # 判断是否用了 fallback（fallback 文章包含固定句式标记）
        fallback_markers = ["it is impossible to ignore its role", "affects our daily routines",
                            "has evolved significantly", "are not alone in this curiosity"]
        if articles and any(m in articles[0].lower() for m in fallback_markers):
            result["used_fallback"] = True

        result["articles"] = articles
        result["coverage"] = eval_result.get("coverage", 0.0)
        result["matched_words"] = eval_result.get("matched_words", [])
        result["missing_words"] = eval_result.get("missing_words", [])
        result["word_stats"] = eval_result.get("word_stats", [])

        # 2. 翻译
        translations = Translator.translate(articles)
        result["translations"] = translations
        if translations and translations[0].startswith("（翻译暂时不可用"):
            result["translation_ok"] = False

        # 3. 阅读理解题
        result["questions"] = QuestionGenerator.generate(articles, words)

        # 4. TTS（文件名加时间戳，避免并发/重复生成互相覆盖）
        audio_status, audio_paths = [], []
        ts = int(time.time() * 1000)
        for idx, art in enumerate(articles[:2], start=1):
            path = os.path.join(self.audio_dir, f"article_{idx}_{ts}.mp3")
            ok = TTS.generate(art, path, rate=AUDIO_SPEED)
            audio_status.append({"article": idx, "generated": ok, "method": "edge-tts" if ok else "browser-fallback"})
            audio_paths.append(path if ok else None)
        result["audio_status"] = audio_status
        result["audio_paths"] = audio_paths

        # 5. 图片生成（简约插图风）
        images = []
        for idx, art in enumerate(articles[:2], start=1):
            try:
                img_result = ImageGenerator.generate(art, words, idx)
                images.append(img_result)
            except Exception as e:
                logger.warning("Image generation for article %d failed: %s", idx, e)
                images.append({
                    "image_path": None,
                    "visual_words": [],
                    "abstract_words": [],
                    "note": None,
                })
        result["images"] = images

        return result

    def _demo(self, words: List[str]) -> Dict:
        """无 API Key 时的 Demo 数据，明确标注 Demo Mode，不伪装真实生成。"""
        article1 = (
            "Every morning, I eat an apple and a banana for breakfast. "
            "Sometimes I add an orange to make the meal more colorful. "
            "A grape is my favorite snack, and a mango is a special treat in summer. "
            "I also enjoy drinking water after eating these healthy foods."
        )
        article2 = (
            "At the market, you can find a fresh apple from local farms and a sweet banana. "
            "Try a ripe orange for vitamin C, one grape for a quick snack, "
            "and a juicy mango to make your day sweeter. "
            "Eating fruit every day is a good habit for your health."
        )
        articles = [article1, article2]
        ev = ContentEvaluator.evaluate(words, articles)
        return {
            "words": words,
            "articles": articles,
            "translations": [
                "每天早上，我吃一个苹果和一根香蕉作为早餐。"
                "有时我会加一个橙子，让这顿饭更有色彩。"
                "葡萄是我最喜欢的零食，而芒果是夏天特别的享受。"
                "吃完这些健康食物后，我也喜欢喝水。",
                "在市场里，你能找到本地农场种的新鲜苹果和香甜的香蕉。"
                "试试成熟的橙子补充维生素C，来些葡萄当作快速零食，"
                "再选一个多汁的芒果，让你的日子更甜美。"
                "每天吃水果对你的健康是个好习惯。",
            ],
            "questions": QuestionGenerator._fallback(articles, words),
            "coverage": ev["coverage"],
            "matched_words": ev["matched_words"],
            "missing_words": ev["missing_words"],
            "word_stats": ev["word_stats"],
            "attempts": [{"attempt": 1, "coverage": ev["coverage"], "missing_words": ev["missing_words"], "demo": True}],
            "demo_mode": True,
            "used_fallback": False,
            "translation_ok": True,
            "audio_status": [
                {"article": 1, "generated": False, "method": "browser-fallback"},
                {"article": 2, "generated": False, "method": "browser-fallback"},
            ],
            "audio_paths": [None, None],
            "images": [],
        }
