"""
capstone/modules/m15_langchain_agent.py
DocumentAssistantAgent — LangChain ReAct agent; barcha kapstone modullarini
asbob (tool) sifatida birlashtiradi. KAPSTONE YAKUNI (himoya demo'si).
Shartnoma: capstone/contracts.py :: DocumentAssistantAgent

Tools:
    sentiment_classify  -> m13 FineTunedClassifier.predict
    retrieve_docs       -> m14 RAGEngine.answer
    summarize_text      -> m12 TransformerSummarizer.summarize
    spell_correct       -> m04 SpellLSHRetriever.correct
    extract_entities    -> m10 NERTagger.entities

langchain / LLM API SHART EMAS:
  - Kaggle yo'li (USE_LANGCHAIN=True): create_react_agent + AgentExecutor + LLM.
  - Offline yo'l: RULE-BASED keyword router (ReAct "Thought" ni o'zbekcha kalit
    so'zlar bilan modellaydi) -> to'g'ri toolni tanlaydi, chaqiradi, javob yig'adi.
Vositalar dependency-injection orqali beriladi (tools dict): m15 orkestratsiya qiladi.
"""
from __future__ import annotations

USE_LANGCHAIN = False    # Kaggle (langchain + LLM) da True; mahalliy offline False


class DocumentAssistantAgent:
    """ReAct agent: 5 kapstone vositasini orkestrlaydi (kapstone yakuni).

    consumed_by: [] (yakuniy o'quv moduli; himoya demo'sida bevosita ishlatiladi).
    """

    # o'zbekcha kalit so'z -> tool (ReAct "Thought" ni offline modellaydi)
    _KEYWORDS = {
        "sentiment_classify": ["yaxshi", "yomon", "yoqdi", "yoqmadi", "sharh",
                               "fikr", "hissiyot", "baho", "qoniqdi"],
        "summarize_text": ["qisqa", "xulosa", "bayon", "umumlashtir", "mazmun", "qisqacha"],
        "extract_entities": ["shahar", "nom", "joy", "tashkilot", "shaxs", "kim", "qayer"],
        "spell_correct": ["imlo", "xato", "to'g'rila", "tuzat", "noto'g'ri"],
        "retrieve_docs": ["qidir", "top", "qanday", "nima", "qonun", "hujjat",
                          "haqida", "qaysi", "tushuntir"],
    }

    def __init__(self, tools: dict | None = None) -> None:
        """tools: {tool_nomi: chaqiriladigan funksiya}. None bo'lsa bo'sh."""
        self._tools = tools or {}
        self.last_trace: list[dict] = []

    # ─── ReAct "Thought": tool tanlash (offline, path-independent) ───────────────
    def route(self, message: str) -> dict:
        """So'rovga mos vositani tanlaydi: {"tool","confidence","args"}."""
        msg = message.lower()
        scores = {tool: sum(1 for kw in kws if kw in msg)
                  for tool, kws in self._KEYWORDS.items()}
        best = max(scores, key=lambda t: scores[t])
        n = scores[best]
        if n == 0:
            return {"tool": "retrieve_docs", "confidence": 0.5, "args": message}
        return {"tool": best, "confidence": min(0.95, 0.7 + 0.1 * n), "args": message}

    def _call_tool(self, tool: str, arg):
        fn = self._tools.get(tool)
        if fn is None:
            return "(vosita ulanmagan)"
        try:
            return fn(arg)
        except Exception as e:                      # demo'da agent buzilmasin
            return "(vosita xatosi: %s)" % e

    # ─── ReAct sikli (offline: 1 qadam router; Kaggle: to'liq LangChain) ─────────
    def run(self, user_message: str) -> str:
        """So'rovni ReAct agentiga yuboradi va o'zbekcha javob qaytaradi."""
        if USE_LANGCHAIN:
            return self._run_langchain(user_message)
        return self._run_offline(user_message)

    def _run_offline(self, message: str) -> str:
        r = self.route(message)
        tool, arg = r["tool"], r["args"]
        thought = "So'rov mazmuni '%s' vositasini talab qiladi." % tool
        observation = self._call_tool(tool, arg)
        self.last_trace = [
            {"thought": thought, "action": tool, "args": arg,
             "confidence": r["confidence"]},
            {"observation": observation},
        ]
        return "Vosita '%s' natijasi asosida javob: %s" % (tool, observation)

    def _run_langchain(self, message: str) -> str:
        # Kaggle yo'li (langchain + LLM): mahalliyda bajarilmaydi.
        from langchain.agents import AgentExecutor, create_react_agent, Tool
        from langchain import hub
        lc_tools = [Tool(name=n, func=f, description=n) for n, f in self._tools.items()]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self._llm, lc_tools, prompt)
        executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=False)
        return executor.invoke({"input": message})["output"]
