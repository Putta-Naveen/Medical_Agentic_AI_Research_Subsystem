import os
import asyncio
import httpx
import logging
import threading
from queue import Queue
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict 
from langgraph.graph import StateGraph, END
import warnings, time, random, json
from dotenv import load_dotenv 
load_dotenv()

# ------------------------ CONFIG ------------------------
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_RAG_URL = os.getenv("GENAI_RAG_URL")
GENAI_RAG_TOKEN = os.getenv("GENAI_RAG_TOKEN")          
MCP_SEARCH_URL   = os.getenv("MCP_SEARCH_URL")
END_USER_ID      = os.getenv("END_USER_ID")

MIN_OVERALL = float(os.getenv("MIN_OVERALL"))
MAX_LOOPS   = int(os.getenv("MAX_LOOPS"))
SUBQ_SEARCH_COUNT = int(os.getenv("SUBQ_SEARCH_COUNT"))
MAX_SOURCES_FOR_CITATIONS = int(os.getenv("MAX_SOURCES_FOR_CITATIONS"))
MAX_EVIDENCE_SNIPPETS     = int(os.getenv("MAX_EVIDENCE_SNIPPETS"))

DEFAULT_ALLOWED = {
    "nih.gov", "ncbi.nlm.nih.gov", "cdc.gov", "who.int",
    "aafp.org", "hopkinsarthritis.org", "uptodate.com",
    "columbia-lyme.org", "racgp.org.au", "hss.edu", "pmc.ncbi.nlm.nih.gov",
}
ENV_ALLOWED = {d.strip().lower() for d in os.getenv("ALLOWED_DOMAINS", "").split(",") if d.strip()}
ALLOWED_DOMAINS = ENV_ALLOWED if ENV_ALLOWED else DEFAULT_ALLOWED

UA_POOL = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LangGraphServer")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. Set it in env before running.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"temperature": 0.2})
gemini_json  = genai.GenerativeModel("gemini-2.5-flash", generation_config={"temperature": 0.0, "response_mime_type": "application/json"})

app = FastAPI(title="Medical Agentic AI Research Subsystem", version="1.5")

# ------------------------ MODELS ------------------------
class EvalRubric(BaseModel):
    coverage: float = 0.0
    grounding: float = 0.0
    coherence: float = 0.0
    overall: float = 0.0
    replan_needed: bool = False
    critique: str = ""

class QueryState(BaseModel):
    question: str
    summary: str = ""
    subqueries: List[str] = Field(default_factory=list)
    answers: Dict[str, str] = Field(default_factory=dict)
    final_answer: str = ""
    evaluation: str = ""                 # "yes"/"no"
    evaluation_reason: str = ""
    feedback: str = ""
    previous_subqueries: List[str] = Field(default_factory=list)
    bad_subqueries: List[str] = Field(default_factory=list)
    loop_count: int = 0
    web_results: List[Dict[str, Any]] = Field(default_factory=list)  # sources with n=1..k
    rag_answer: str = ""
    rag_summary: str = ""
    scores: Optional[EvalRubric] = None

# ---- Request model: ONLY these two fields appear in Swagger ----
class ClientRequest(BaseModel):
    query: str
    end_user_id: str = END_USER_ID

    # Pydantic v2 replacement for schema_extra
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "string",
                "end_user_id": "test-user"
            }
        }
    )

# ---- Response model  ----
class ClientResponse(BaseModel):
    question: str
    summary: str
    rag_answer: str
    rag_summary: str
    web_results: List[Dict[str, Any]]
    subqueries: List[str]
    subquery_answers: Dict[str, str]
    final_answer: str
    evaluation: str
    feedback: str
    rubric: Optional[EvalRubric] = None

# ------------------------ HELPERS ------------------------
def _sleep_backoff(): time.sleep(45)

def _allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return any(host.endswith(d) for d in ALLOWED_DOMAINS)
    except Exception:
        return False

def _headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

def _is_probably_pdf(url: str, content_type: Optional[str]) -> bool:
    if url.lower().endswith(".pdf"): return True
    if content_type and "pdf" in content_type.lower(): return True
    return False

def _dedupe_by_link(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in items:
        k = (r.get("link") or r.get("url") or "").strip()
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out

async def get_concise_summary(text: str) -> str:
    for attempt in range(3):
        try:
            prompt = f"Provide a very concise summary (max 50 words) of the following text:\n\n{(text or '')[:2000]}"
            logger.info(f"get_concise_summary prompt: {prompt}")
            resp = await gemini_model.generate_content_async(prompt)
            logger.info(f"get_concise_summary response: {resp}")
            return (resp.text or "").strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning("Rate limit in get_concise_summary; retry in 45s")
                _sleep_backoff()
            else:
                logger.error(f"Gemini concise summarization error: {e}")
                logger.error(f"get_concise_summary input text: {text}")
                return "Summary not available due to API error."

def expand_and_summarize_web(link: Dict[str, Any], query: str, queue: Queue):
    try:
        url = link.get("link") or link.get("url") or ""
        if not url or not _allowed(url):
            link["summary"] = "Filtered or invalid URL"
            queue.put(link); return

        title = link.get("title", link.get("displayLink", "Source"))
        snippet = link.get("snippet", "")
        page_text = ""

        try:
            resp = httpx.get(url, timeout=12, headers=_headers(), follow_redirects=True)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if _is_probably_pdf(url, ctype):
                link["summary"] = "PDF detected; skipped parsing."
                link["title"] = title
                queue.put(link); return

            content = resp.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script","style","form","ad","advertisement","noscript","iframe"]):
                tag.decompose()
            page_text = soup.get_text(separator=" ", strip=True)
            if not page_text or len(page_text) < 120:
                page_text = snippet or (soup.title.string if soup.title else "")
        except Exception as e:
            logger.warning(f"Fetch fail {url}: {e}")
            page_text = snippet

        prompt = (
            f"Question: {query}\n\n"
            f"Provide a concise summarization (max 50 words) of the following text:\n\n{(page_text or '')[:2000]}"
        )
        for attempt in range(3):
            try:
                summary = gemini_model.generate_content(prompt).text.strip()
                link["summary"] = summary
                link["title"] = title
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    logger.warning("Rate limit in expand_and_summarize_web; retry 45s")
                    _sleep_backoff()
                else:
                    logger.error(f"Gemini summarization error: {e}")
                    link["summary"] = f"Error: {e}"
                    break
    except Exception as e:
        logger.error(f"Error processing link: {e}")
        link["summary"] = f"Error: {e}"
    finally:
        queue.put(link)

def _web_context(results: List[Dict[str, Any]]) -> str:
    return "\n".join([f"[{r.get('n')}]\nTitle: {r.get('title','')}\nSummary: {r.get('summary','No summary')}" for r in results[:3]])

def _rag_context(state: QueryState) -> str:
    return f"RAG Answer: {state.rag_answer[:500]}\nRAG Summary: {state.rag_summary}" if state.rag_answer else ""

# ------------------------ NODES ------------------------
def structured_summarizer(state: QueryState) -> QueryState:
    prompt = f"""
Summarize this medical question clearly:
Question: {state.question}
Extract:
- Condition
- Symptoms
- Tests/Treatments
- Core Clinical Goal
Strict output format:
- Condition:
- Symptoms:
- Tests/Treatments:
- Core Clinical Goal:
"""
    for attempt in range(3):
        try:
            resp = gemini_model.generate_content(prompt)
            state.summary = (resp.text or "").strip()
            return state
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning("Rate limit in structured_summarizer; retry 45s")
                _sleep_backoff()
            else:
                logger.error(f"Gemini API error: {e}")
                state.summary = f"Error: {e}"; return state

def planner(state: QueryState) -> QueryState:
    avoid = list(set(state.previous_subqueries + state.bad_subqueries))
    web_ctx = _web_context(state.web_results) if state.web_results else ""
    rag_ctx = _rag_context(state)

    prompt = f"""
Generate 3-5 subquestions to answer the main query:
"{state.question}"

Use (if present):
- Structured Summary: {state.summary}
- Web Search Results: {web_ctx if web_ctx else "None"}
- RAG Results: {rag_ctx if rag_ctx else "None"}
- Feedback: "{state.feedback if state.feedback else 'None'}"

Avoid repeating:
{chr(10).join(['- ' + a for a in avoid]) if avoid else 'None'}

Output: Numbered list of distinct, feasible, medically valid subquestions.
"""
    for attempt in range(3):
        try:
            resp = gemini_model.generate_content(prompt)
            state.previous_subqueries = state.subqueries
            state.subqueries = [
                line.strip("0123456789. ").strip()
                for line in (resp.text or "").splitlines()
                if line.strip() and not line.lower().startswith("here are")
            ]
            return state
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning("Rate limit in planner; retry 45s")
                _sleep_backoff()
            else:
                logger.error(f"Gemini API error: {e}")
                state.subqueries = []
                return state

async def executor(state: QueryState) -> QueryState:
    logger.info(" Executor starting")
    aggregated: List[Dict[str, Any]] = []

    # ---- RAG on the parent question ----
    try:
        headers = {"Authorization": f"Bearer {GENAI_RAG_TOKEN}"} if GENAI_RAG_TOKEN else None
        rag_res = httpx.post(GENAI_RAG_URL, json={"query": state.question, "end_user_id": END_USER_ID},
                             headers=headers, timeout=20)
        rag_res.raise_for_status()
        rag_data = rag_res.json()
        state.rag_answer  = rag_data.get("output_text", "No answer from RAG.")
        state.rag_summary = await get_concise_summary(state.rag_answer)
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        state.rag_answer = f"RAG Error: {e}"
        state.rag_summary = "RAG summary failed."

    # ---- Per-subquestion search + page expansion (parallel) ----
    for subq in state.subqueries:
        q = Queue(); threads: List[threading.Thread] = []
        try:
            web_res = httpx.post(MCP_SEARCH_URL, json={"query": subq, "count": SUBQ_SEARCH_COUNT}, timeout=20)
            web_res.raise_for_status()
            links = web_res.json().get("results", [])
            links = [l for l in links if _allowed(l.get("link","") or l.get("url",""))][:SUBQ_SEARCH_COUNT]
            for link in links:
                t = threading.Thread(target=expand_and_summarize_web, args=(link, subq, q))
                t.start(); threads.append(t)
        except Exception as e:
            logger.error(f"Websearch error ({subq}): {e}")
            aggregated.append({"title": subq, "link": "", "summary": f"Websearch error: {e}"})

        for t in threads: t.join()
        while not q.empty():
            aggregated.append(q.get())

    # Deduplicate and number sources for citation mapping
    state.web_results = _dedupe_by_link(aggregated)
    for i, r in enumerate(state.web_results[:MAX_SOURCES_FOR_CITATIONS]):  # add n=1..k
        r["n"] = i + 1
    return state

def answer_subquestions(state: QueryState) -> QueryState:
    web_ctx = _web_context(state.web_results)
    rag_ctx = _rag_context(state)

    answers: Dict[str, str] = {}
    for subquery in state.subqueries:
        prompt = f"""
Answer concisely (max 4 sentences).
Use ONLY the Evidence below; add inline citations like [1] using the numbered list.

Parent Question: "{state.question}"
Subquestion: {subquery}

Evidence:
{web_ctx if web_ctx else "None"}
{("\nRAG: " + rag_ctx) if rag_ctx else ""}
"""
        for attempt in range(3):
            try:
                resp = gemini_model.generate_content(prompt)
                answers[subquery] = (resp.text or "").strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    logger.warning(f"Rate limit answering subq '{subquery}'; retry 45s")
                    _sleep_backoff()
                else:
                    logger.error(f"Gemini error for subq '{subquery}': {e}")
                    answers[subquery] = f"Gemini error: {e}"
                    break
    state.answers = answers
    return state

def synthesizer(state: QueryState) -> QueryState:
    # Build numbered source list and snippets
    sources = state.web_results[:MAX_SOURCES_FOR_CITATIONS]
    sources_list_text = "\n".join([f"[{s.get('n')}] {s.get('title','Source')} — {s.get('link') or s.get('url','')}" for s in sources])
    evidence_snips = "\n\n".join([f"[{s.get('n')}] Summary: {(s.get('summary','') or '')[:500]}" for s in sources[:MAX_EVIDENCE_SNIPPETS]])
    rag_ctx = f"RAG Summary: {state.rag_summary}" if state.rag_summary else ""

    prompt = f"""
You are writing a concise, evidence-grounded medical answer.

Question:
{state.question}

Evidence snippets (cite ONLY these numbers):
{evidence_snips}

Sources list (use these numbers for inline citations):
{sources_list_text}

Rules:
- Base claims ONLY on the evidence snippets above.
- Add inline citations like [1], [2] matching the numbered Sources list.
- If evidence is insufficient, state the gap explicitly.
- Max 3 sentences.
{rag_ctx}
"""
    for attempt in range(3):
        try:
            resp = gemini_model.generate_content(prompt)
            state.final_answer = (resp.text or "").strip()
            return state
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning("Rate limit in synthesizer; retry 45s")
                _sleep_backoff()
            else:
                logger.error(f"Gemini API error: {e}")
                state.final_answer = f"Error: {e}"; return state

def evaluator(state: QueryState) -> QueryState:
    state.loop_count += 1
    ev = [{
        "title": r.get("title", ""),
        "url":   r.get("link", "") or r.get("url",""),
        "summary": (r.get("summary", "") or "")[:600]
    } for r in state.web_results[:MAX_EVIDENCE_SNIPPETS]]

    prompt = f"""
Return STRICT JSON with keys:
coverage (0..1), grounding (0..1), coherence (0..1), overall (0..1), replan_needed (true/false), critique (string).

Question: {state.question}
Final Answer: {state.final_answer}
Evidence: {json.dumps(ev, ensure_ascii=False)}
"""
    for attempt in range(3):
        try:
            raw = (gemini_json.generate_content(prompt).text or "").strip()
            if not raw: raise ValueError("Empty eval response")
            try:
                data = json.loads(raw)
            except Exception:
                fixed = (gemini_json.generate_content(f"Return valid JSON only (no prose):\n{raw}").text or "").strip()
                data = json.loads(fixed)

            logger.info(f"Rubric JSON: {data}")  # <-- Added logging for rubric

            rubric = EvalRubric(
                coverage=float(data.get("coverage", 0.0)),
                grounding=float(data.get("grounding", 0.0)),
                coherence=float(data.get("coherence", 0.0)),
                overall=float(data.get("overall", 0.0)),
                replan_needed=bool(data.get("replan_needed", False)),
                critique=str(data.get("critique", "")),
            )
            state.scores = rubric

            if rubric.overall >= MIN_OVERALL:
                state.evaluation, state.feedback, state.bad_subqueries = "yes", "", []
            else:
                state.evaluation = "no"
                state.feedback = rubric.critique or "Needs improvement"
                state.bad_subqueries = state.subqueries.copy()
            return state

        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning("Rate limit in evaluator; retry 45s")
                _sleep_backoff()
            else:
                logger.error(f"Evaluator error: {e}")
                state.scores = EvalRubric(overall=0.0, replan_needed=True, critique=f"Evaluation failed: {e}")
                state.evaluation = "no"
                state.feedback = state.scores.critique
                state.bad_subqueries = state.subqueries.copy()
                return state

def should_replan(state: QueryState) -> str:
    if state.loop_count >= MAX_LOOPS:
        return "end"
    # Only replan if overall score is less than MIN_OVERALL
    if state.scores and state.scores.overall < MIN_OVERALL:
        return "replan"
    return "end"

# ------------------------ GRAPH ------------------------
graph = StateGraph(QueryState)
graph.add_node("summarizer", structured_summarizer)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("answer_subqs", answer_subquestions)
graph.add_node("synthesizer", synthesizer)
graph.add_node("evaluator", evaluator)

graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "answer_subqs")
graph.add_edge("answer_subqs", "synthesizer")
graph.add_edge("synthesizer", "evaluator")
graph.add_conditional_edges("evaluator", should_replan, {"end": END, "replan": "planner"})

compiled_graph = graph.compile()

# ------------------------ API ------------------------
@app.post("/mcp/runLanggraph", response_model=ClientResponse, tags=["LangGraph"])
async def run_pipeline(request_body: ClientRequest, request: Request):
    try:
        initial = QueryState(question=request_body.query)
        final: Optional[QueryState] = None

        async for step in compiled_graph.astream(initial, config={"recursion_limit": MAX_LOOPS * 8}):
            node, raw = list(step.items())[0]
            final = QueryState(**raw)
            logger.info(f"Node: {node} | Loop: {final.loop_count} | Eval: {final.evaluation} | Overall: {final.scores.overall if final.scores else 'NA'}")
            if final.scores and final.scores.overall >= MIN_OVERALL and not final.scores.replan_needed:
                break
            if final.evaluation == "yes":
                break

        if not final:
            raise RuntimeError("Pipeline produced no final state")

        return ClientResponse(
            question=final.question,
            summary=final.summary,
            rag_answer=final.rag_answer,
            rag_summary=final.rag_summary,
            web_results=final.web_results,        # includes n=1..k so [1],[2] map by position
            subqueries=final.subqueries,
            subquery_answers=final.answers,
            final_answer=final.final_answer,
            evaluation=final.evaluation,
            feedback=final.feedback,
            rubric=final.scores
        )
    except Exception as e:
        logger.error(f"LangGraph failed: {e}")
        raise HTTPException(status_code=500, detail=f"LangGraph failed: {e}")

# ------------------------ MAIN ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
