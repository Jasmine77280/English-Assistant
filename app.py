"""AI English Learning Agent — Flask Web 应用。

API Key 仅在服务端读取（.env），不暴露到前端。
无 API Key 时进入 Demo Mode。
"""
import os
import sys
import logging

# 将项目内 vendor 目录加入 sys.path（Flask 等依赖安装于此）
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, abort, Response,
    send_file,
)

import agent
import db
import report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

db.init_db()
learning_agent = agent.EnglishLearningAgent(audio_dir=AUDIO_DIR)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.globals["chr"] = chr


# ---------- 图片静态文件路由 ----------
@app.route("/images/<path:filename>")
def serve_image(filename):
    """提供 images/ 目录下生成的插图。"""
    return send_from_directory(IMAGE_DIR, filename, mimetype="image/png")


# ---------- 页面路由 ----------
@app.route("/")
def home():
    return render_template("index.html", demo_mode=not bool(agent.QWEN_API_KEY))


@app.route("/learning")
def learning_page():
    return render_template("learning.html", demo_mode=not bool(agent.QWEN_API_KEY))


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/share/<token>")
def share_page(token):
    rec = db.get_record_by_share_token(token)
    if not rec:
        return render_template("share.html", record=None, error="分享链接无效或已过期"), 404
    return render_template("share.html", record=rec)


@app.route("/report/<int:record_id>")
def report_page(record_id):
    rec = db.get_record(record_id)
    if not rec:
        abort(404)
    return Response(report.generate_report_html(rec), mimetype="text/html")


# ---------- API ----------
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "demo_mode": not bool(agent.QWEN_API_KEY),
        "api_key_configured": bool(agent.QWEN_API_KEY),
        "edge_tts_available": agent.HAS_EDGE_TTS,
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    words = data.get("words", [])
    if not isinstance(words, list) or len(words) == 0:
        return jsonify({"error": "请输入至少 5 个英文单词"}), 400

    cleaned, invalid = agent.VocabularyAnalyzer.clean(words)
    if invalid:
        return jsonify({
            "error": "部分单词包含非法字符（仅允许 a-z/A-Z）",
            "invalid_words": invalid,
        }), 400
    if len(cleaned) < 5:
        return jsonify({
            "error": f"至少需要 5 个有效英文单词，当前 {len(cleaned)} 个",
            "word_count": len(cleaned),
        }), 400

    try:
        result = learning_agent.run(cleaned)
    except Exception as e:
        logger.exception("Pipeline failed")
        return jsonify({"error": f"生成失败：{str(e)[:200]}"}), 500

    # 保存记录
    try:
        rid = db.save_record(result)
        result["record_id"] = rid
    except Exception as e:
        logger.warning("DB save failed: %s", e)
        result["record_id"] = None

    # 不暴露服务端绝对路径给前端
    result.pop("audio_paths", None)
    return jsonify(result)


@app.route("/api/regenerate", methods=["POST"])
def regenerate():
    return generate()


@app.route("/api/audio/<int:record_id>/<int:article_idx>")
def serve_audio(record_id, article_idx):
    rec = db.get_record(record_id)
    if not rec:
        abort(404)
    paths = rec.get("audio_paths") or []
    if article_idx < 1 or article_idx > len(paths):
        abort(404)
    path = paths[article_idx - 1]
    if not path or not os.path.exists(path):
        return jsonify({"generated": False, "fallback": "browser-speech-synthesis"}), 200
    return send_from_directory(os.path.dirname(path), os.path.basename(path), mimetype="audio/mpeg")


@app.route("/api/tts", methods=["POST"])
def tts():
    """为给定文本生成 TTS 音频（支持语速调节）。"""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    rate = float(data.get("rate", 1.0))
    rate = max(0.5, min(2.0, rate))
    if not text:
        return jsonify({"error": "text is required"}), 400
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".mp3", dir=AUDIO_DIR)
    os.close(fd)
    ok = agent.TTS.generate(text, path, rate=rate)
    if not ok:
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({"generated": False, "fallback": "browser-speech-synthesis"})
    return send_from_directory(os.path.dirname(path), os.path.basename(path), mimetype="audio/mpeg")


# ---------- 分享链接 ----------
@app.route("/api/share/<int:record_id>", methods=["POST"])
def create_share(record_id):
    rec = db.get_record(record_id)
    if not rec:
        return jsonify({"error": "记录不存在"}), 404
    token = db.create_share_token(record_id)
    share_url = f"{request.host_url}share/{token}"
    return jsonify({"token": token, "share_url": share_url})


# ---------- 报告导出 ----------
@app.route("/api/report/<int:record_id>.md")
def report_md(record_id):
    rec = db.get_record(record_id)
    if not rec:
        abort(404)
    md = report.generate_markdown(rec)
    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{record_id}.md"},
    )


@app.route("/api/report/<int:record_id>.pdf")
def report_pdf(record_id):
    rec = db.get_record(record_id)
    if not rec:
        abort(404)
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".pdf", dir=AUDIO_DIR)
    os.close(fd)
    ok = report.generate_pdf(rec, path)
    if not ok:
        # PDF 生成失败，回退到 HTML 报告（浏览器可打印为 PDF）
        try:
            os.remove(path)
        except OSError:
            pass
        return Response(
            report.generate_report_html(rec),
            mimetype="text/html",
        )
    return send_file(path, mimetype="application/pdf", as_attachment=True, download_name=f"report_{record_id}.pdf")


@app.route("/api/history")
def list_history():
    return jsonify(db.list_records())


@app.route("/api/history/<int:record_id>")
def get_history(record_id):
    rec = db.get_record(record_id)
    if not rec:
        return jsonify({"error": "not found"}), 404
    return jsonify(rec)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    ok = db.delete_record(record_id)
    return jsonify({"deleted": ok})


@app.route("/api/quiz/save", methods=["POST"])
def save_quiz():
    data = request.get_json(force=True)
    record_id = data.get("record_id")
    answers = data.get("answers", [])
    correct_count = data.get("correct_count", 0)
    total_count = data.get("total_count", 0)
    if not record_id:
        return jsonify({"error": "record_id required"}), 400
    attempt_id = db.save_quiz_attempt(record_id, answers, correct_count, total_count)
    return jsonify({"attempt_id": attempt_id, "saved": True})


@app.route("/api/quiz/latest/<int:record_id>")
def latest_quiz(record_id):
    attempt = db.get_latest_quiz_attempt(record_id)
    if not attempt:
        return jsonify({"found": False})
    return jsonify({"found": True, **attempt})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
