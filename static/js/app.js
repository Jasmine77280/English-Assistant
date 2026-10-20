// AI English Learning Agent - 共享前端工具
window.Agent = {
  escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s ?? "";
    return d.innerHTML;
  },
  // 浏览器语音合成 fallback（Edge TTS 失败时使用）
  speak(text, lang = "en-US") {
    if (!("speechSynthesis" in window)) return false;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    u.rate = 0.95;
    window.speechSynthesis.speak(u);
    return true;
  },
};
