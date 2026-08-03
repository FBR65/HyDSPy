"""Agentic RAG loop (Microsoft AgenticRAG, arXiv:2605.05538v1) layered on top
of the existing HyDE + LanceDB + BM25 infrastructure.

The harness equips a reasoning LLM with four tools and lets it iteratively
retrieve, inspect and summarise evidence until it is ready to answer, instead
of finalising retrieval before reasoning begins (as a plain HyDE pipeline does).

Tools (coarse-to-fine, mirroring the paper):
    SEARCH   - corpus-wide search; up to ``max_queries_per_call`` reformulations,
               returns snippets + reference IDs
    FIND     - keyword search inside a previously retrieved document
    OPEN     - window-based reading of a document's chunk texts
    SUMMARIZE- compresses the running evidence trail to reclaim token budget

Loop control:
    - Iterates up to ``max_calls`` times.
    - Terminates when the LLM produces a final answer without tool calls.
    - Forced final answer when ``max_calls`` is reached.

Reference IDs: every search result gets a stable ``turnNsearchM`` id, which
``find`` and ``open`` reuse to avoid re-querying.

Token budget: the paper uses 128k tokens with summarization at 90%. Here the
budget is configurable; ``token_threshold`` controls when SUMMARIZE is forced.
Per the user's instruction, the token budget may be exceeded (no hard cap).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy

from .bm25_manager import BM25Manager
from .embedding_manager import EmbeddingManager
from .lancedb_manager import LanceDBManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------


class AgenticStep(dspy.Signature):
    """You are an enterprise knowledge-base research agent.

    Use the available tools to gather evidence before answering. Available tools:
      search(queries: list[str]) -> snippets with reference ids
      find(reference_id: str, patterns: list[str]) -> matching passages
      open(reference_id: str, position: int = 0) -> window of chunk text
      summarize() -> compress accumulated evidence

    Rules:
    - Search before answering when uncertain.
    - Progressively explore with find or open when snippets are insufficient.
    - Reuse previous results rather than searching again.
    - When you have enough evidence, respond with a final answer and cite the
      reference ids you used. Do NOT emit a tool call in the final turn.
    """

    question = dspy.InputField()
    tool_history = dspy.InputField(desc="JSON list of previous tool calls and results.")
    next_action = dspy.OutputField(
        desc="Either a JSON tool call object {tool, args} or a final answer string."
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    reference_id: str
    document_id: str
    chunk_id: str
    text: str
    score: float


@dataclass
class AgenticState:
    """Mutable conversation/context state for the agentic loop."""

    question: str
    document_id: Optional[str] = None
    k: int = 3
    reference_counter: int = 0
    references: Dict[str, SearchResult] = field(default_factory=dict)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0


class AgenticTools:
    """Concrete tool implementations backed by LanceDB + BM25 + embeddings."""

    def __init__(
        self,
        db_manager: LanceDBManager,
        embedding_manager: EmbeddingManager,
        bm25_manager: Optional[BM25Manager] = None,
        search_limit: int = 10,
        max_queries_per_call: int = 5,
        open_window: int = 5,
    ) -> None:
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.bm25_manager = bm25_manager
        self.search_limit = search_limit
        self.max_queries_per_call = max_queries_per_call
        self.open_window = open_window

    def search(
        self, state: AgenticState, queries: List[str], limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Corpus-wide search with multiple reformulations, deduplicated.

        The per-turn result limit defaults to ``state.k`` (the ``k`` passed to
        ``AgenticRAG.forward``) so the API's ``k`` parameter actually constrains
        agentic retrieval, falling back to ``self.search_limit`` only when
        ``state.k`` is unset.
        """
        limit = limit or state.k or self.search_limit
        queries = [q for q in (queries or []) if q][: self.max_queries_per_call]
        seen_chunk_ids: set = set()
        results: List[SearchResult] = []
        # Reset the per-turn counter so reference ids are turn-local: turn1
        # produces search1..searchM, turn2 produces search1..searchM again.
        # The turn number is already encoded in the id prefix.
        state.reference_counter = 0
        for q in queries:
            try:
                embeddings = self.embedding_manager.get_embeddings([q])
            except RuntimeError as e:
                logger.warning("Agentic search embedding failed: %s", e)
                continue
            df = self.db_manager.search(
                embeddings[0], limit=limit, document_id=state.document_id
            )
            if df.empty or "text" not in df.columns:
                continue
            for _, row in df.iterrows():
                chunk_id = str(row.get("chunk_id", ""))
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                state.reference_counter += 1
                ref_id = f"turn{state.iteration}search{state.reference_counter}"
                sr = SearchResult(
                    reference_id=ref_id,
                    document_id=str(row.get("document_id", "")),
                    chunk_id=chunk_id,
                    text=str(row["text"]),
                    score=float(row.get("_distance", 0.0)),
                )
                state.references[ref_id] = sr
                results.append(sr)
        return results

    def find(
        self, state: AgenticState, reference_id: str, patterns: List[str]
    ) -> List[str]:
        """Case-insensitive substring search within one retrieved document."""
        sr = state.references.get(reference_id)
        if sr is None:
            return [f"Unknown reference_id: {reference_id}"]
        passages: List[str] = []
        text = sr.text or ""
        for pattern in patterns or []:
            if not pattern:
                continue
            idx = text.lower().find(pattern.lower())
            if idx >= 0:
                start = max(0, idx - 120)
                end = min(len(text), idx + len(pattern) + 240)
                passages.append(text[start:end])
            if len(passages) >= 2:
                break
        return passages or ["No matches found."]

    def open(self, state: AgenticState, reference_id: str, position: int = 0) -> str:
        """Window-based reading of all chunks belonging to the reference's document."""
        sr = state.references.get(reference_id)
        if sr is None:
            return f"Unknown reference_id: {reference_id}"
        doc_chunks = self.db_manager.get_document_chunks(sr.document_id)
        if doc_chunks.empty:
            return f"No chunks for document {sr.document_id}."
        window = self.open_window
        total = len(doc_chunks)
        start = max(0, min(position, total - 1))
        end = min(total, start + window)
        header = (
            f"Viewing chunks [{start}-{end - 1}] of {total} "
            f"for document {sr.document_id}."
        )
        lines: List[str] = [header]
        for i, (_, row) in enumerate(doc_chunks.iloc[start:end].iterrows()):
            lines.append(f"[chunk {start + i}] {row['text']}")
        return "\n".join(lines)

    def summarize(self, state: AgenticState) -> str:
        """Compress the accumulated evidence trail (context management).

        Keeps the reference-id mapping but shortens the stored texts to the first
        ~200 characters so subsequent turns have headroom. Returns a short summary
        listing the surviving reference ids.
        """
        if not state.references:
            return "No evidence accumulated yet."
        kept: List[str] = []
        for ref_id, sr in state.references.items():
            sr.text = (sr.text or "")[:200] + ("..." if len(sr.text) > 200 else "")
            kept.append(f"{ref_id}: {sr.text}")
        return "Summarized evidence:\n" + "\n".join(kept)


# ---------------------------------------------------------------------------
# Agentic RAG module
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_XML_TOOL_RE = re.compile(
    r"<tool[^>]*name=[\"\']?(\w+)[\"\']?[^>]*>(.*?)</tool>", re.DOTALL | re.IGNORECASE
)
_TOOL_NAMES = ("search", "find", "open", "summarize")


class AgenticRAG(dspy.Module):
    """Iterative agentic retrieval loop.

    Complements (does not replace) ``HyDERAG``. Use this for multi-step
    questions where single-shot HyDE retrieval is insufficient.
    """

    def __init__(
        self,
        db_manager: LanceDBManager,
        embedding_manager: EmbeddingManager,
        bm25_manager: Optional[BM25Manager] = None,
        max_calls: int = 15,
        token_threshold: int = 115_200,
        search_limit: int = 10,
        max_queries_per_call: int = 5,
        open_window: int = 5,
        default_k: int = 3,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.bm25_manager = bm25_manager
        self.max_calls = max_calls
        self.token_threshold = token_threshold
        self.default_k = default_k
        self.tools = AgenticTools(
            db_manager=db_manager,
            embedding_manager=embedding_manager,
            bm25_manager=bm25_manager,
            search_limit=search_limit,
            max_queries_per_call=max_queries_per_call,
            open_window=open_window,
        )
        self.step = dspy.ChainOfThought(AgenticStep)

    def forward(
        self,
        question: str,
        document_id: Optional[str] = None,
        k: int = None,
    ):
        state = AgenticState(question=question, document_id=document_id)
        state.k = self.default_k if k is None else k
        last_action = ""
        for _ in range(self.max_calls):
            state.iteration += 1
            tool_history_json = json.dumps(state.tool_history[-12:], ensure_ascii=False)
            try:
                res = self.step(question=state.question, tool_history=tool_history_json)
            except Exception as e:
                # DSPy/ChainOfThought can raise various exceptions (network,
                # parse, API-level) — none of which should crash the agentic
                # loop. We log and break so the forced-final-answer path runs.
                logger.warning("Agentic step LLM call failed: %s", e)
                break
            action_raw = (res.next_action or "").strip()
            last_action = action_raw
            tool_call = self._parse_tool_call(action_raw)
            if tool_call is None:
                # No tool call -> final answer
                answer = self._extract_final_answer(action_raw)
                context = self._gather_context(state)
                return dspy.Prediction(answer=answer, context=context)
            self._execute_tool(state, tool_call)
            if self._estimate_tokens(state) >= self.token_threshold:
                logger.info(
                    "Token threshold reached (%s), forcing summarize.",
                    self._estimate_tokens(state),
                )
                self.tools.summarize(state)
        # Forced final answer
        logger.info(
            "Agentic loop reached max_calls=%d; forcing final answer.", self.max_calls
        )
        context = self._gather_context(state)
        answer = self._force_final_answer(state, last_action)
        return dspy.Prediction(answer=answer, context=context)

    # --- helpers -----------------------------------------------------------

    def _parse_tool_call(self, action: str) -> Optional[Dict[str, Any]]:
        """Parse a tool call out of the LLM action string, tolerating multiple
        output formats that models commonly emit:

        1. Fenced JSON:     ```json\\n{"tool": "...", "args": {...}}\\n```
        2. Inline JSON:    ... {"tool": "...", "args": {...}} ...
        3. XML tool tag:   <tool name="search">{"queries": [...]}</tool>
        4. Bare tool line: search: [\\"q1\\", \\"q2\\"]  (heuristic fallback)

        Returns ``{"tool": str, "args": dict}`` on success, ``None`` when the
        action looks like a final answer (no recognisable tool call).
        """
        if not action:
            return None
        tool_name = None
        args: Dict[str, Any] = {}

        # 1. Fenced JSON block.
        m = _FENCED_JSON_RE.search(action)
        if m:
            parsed = self._safe_json(m.group(1))
            if parsed and "tool" in parsed:
                return parsed

        # 2. Inline JSON object anywhere in the action.
        match = _TOOL_CALL_RE.search(action)
        if match:
            parsed = self._safe_json(match.group(0))
            if parsed and "tool" in parsed:
                return parsed

        # 3. XML-style <tool name="...">payload</tool>.
        m = _XML_TOOL_RE.search(action)
        if m:
            tool_name = m.group(1).lower()
            payload = m.group(2).strip()
            parsed_args = self._safe_json(payload)
            if parsed_args is not None:
                args = parsed_args if isinstance(parsed_args, dict) else {}
            else:
                args = self._coerce_args_from_text(tool_name, payload)
            if tool_name in _TOOL_NAMES:
                return {"tool": tool_name, "args": args}

        # 4. Heuristic: a line starting with a known tool name followed by
        # a colon or parenthesis, e.g. ``search: ["q"]`` or ``open(doc1, 0)``.
        heuristic = self._parse_heuristic_line(action)
        if heuristic is not None:
            return heuristic

        return None

    @staticmethod
    def _safe_json(raw: str) -> Optional[Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_heuristic_line(action: str) -> Optional[Dict[str, Any]]:
        """Best-effort parse of a bare ``tool: args`` or ``tool(args)`` line."""
        for name in _TOOL_NAMES:
            m = re.search(
                rf"\b{name}\s*[:\(]\s*(.+?)(?:\)|$)",
                action,
                re.IGNORECASE,
            )
            if not m:
                continue
            payload = m.group(1).strip().rstrip(")")
            if not payload:
                return {"tool": name, "args": {}}
            parsed = AgenticRAG._safe_json(payload)
            if parsed is not None:
                return {
                    "tool": name,
                    "args": parsed if isinstance(parsed, dict) else {},
                }
            return {
                "tool": name,
                "args": AgenticRAG._coerce_args_from_text(name, payload),
            }
        return None

    @staticmethod
    def _coerce_args_from_text(tool: str, text: str) -> Dict[str, Any]:
        """Coerce a free-text payload into the expected arg shape per tool."""
        text = (text or "").strip()
        if tool == "search":
            if text:
                qs = [q.strip() for q in re.split(r"[;\n]", text) if q.strip()]
                return {"queries": qs}
            return {"queries": []}
        if tool == "find":
            parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]
            ref_id = parts[0] if parts else ""
            patterns = parts[1:] if len(parts) > 1 else []
            return {"reference_id": ref_id, "patterns": patterns}
        if tool == "open":
            parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]
            ref_id = parts[0] if parts else ""
            position = 0
            if len(parts) > 1 and parts[1].isdigit():
                position = int(parts[1])
            return {"reference_id": ref_id, "position": position}
        return {}

    def _execute_tool(self, state: AgenticState, tool_call: Dict[str, Any]) -> None:
        tool = tool_call.get("tool")
        args = tool_call.get("args", {}) or {}
        entry: Dict[str, Any] = {"tool": tool, "args": args, "result": ""}
        try:
            if tool == "search":
                queries = args.get("queries") or []
                if isinstance(queries, str):
                    queries = [queries]
                results = self.tools.search(state, queries)
                entry["result"] = self._format_search_results(results)
            elif tool == "find":
                ref_id = args.get("reference_id")
                patterns = args.get("patterns") or []
                passages = self.tools.find(state, ref_id, patterns)
                entry["result"] = "\n".join(passages)
            elif tool == "open":
                ref_id = args.get("reference_id")
                position = int(args.get("position", 0))
                entry["result"] = self.tools.open(state, ref_id, position)
            elif tool == "summarize":
                entry["result"] = self.tools.summarize(state)
            else:
                entry["result"] = f"Unknown tool: {tool}"
        except Exception as e:
            entry["result"] = f"Tool error: {e}"
        state.tool_history.append(entry)

    @staticmethod
    def _format_search_results(results: List[SearchResult]) -> str:
        if not results:
            return "No results."
        lines = []
        for sr in results:
            snippet = (sr.text or "")[:200].replace("\n", " ")
            lines.append(f"[{sr.reference_id}] ({sr.document_id}) {snippet}")
        return "\n".join(lines)

    def _gather_context(self, state: AgenticState) -> str:
        if not state.references:
            return "No relevant contexts found."
        texts = [sr.text for sr in state.references.values() if sr.text]
        unique = list(dict.fromkeys(texts))
        return "\n\n".join(unique) if unique else "No relevant contexts found."

    @staticmethod
    def _extract_final_answer(action: str) -> str:
        """If the model returned a JSON object with an 'answer' field, use it;
        otherwise return the raw action string as the answer."""
        match = _TOOL_CALL_RE.search(action)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and "answer" in obj:
                    return str(obj["answer"])
            except json.JSONDecodeError:
                pass
        return action

    def _force_final_answer(self, state: AgenticState, last_action: str) -> str:
        context = self._gather_context(state)
        prompt = (
            f"Question: {state.question}\n\n"
            f"Available evidence:\n{context}\n\n"
            f"Previous reasoning: {last_action}\n\n"
            "Based on the available information above, provide a final answer. "
            "Cite reference ids where possible."
        )
        # Use DSPy's configured LM directly for the forced answer. DSPy's LM
        # access API has shifted between versions (dspy.settings.lm vs.
        # dspy.get_lm()), so try the modern accessor first and fall back.
        try:
            lm = None
            get_lm = getattr(dspy, "get_lm", None)
            if callable(get_lm):
                try:
                    lm = get_lm()
                except Exception:
                    lm = None
            if lm is None:
                lm = getattr(dspy.settings, "lm", None)
            if lm is not None:
                resp = lm(prompt)
                if isinstance(resp, str):
                    return resp
                if isinstance(resp, list) and resp:
                    return str(resp[0])
        except Exception as e:
            logger.warning("Forced final answer via LM failed: %s", e)
        return (
            last_action or "Unable to determine an answer from the gathered evidence."
        )

    @staticmethod
    def _estimate_tokens(state: AgenticState) -> int:
        """Rough token estimate (4 chars/token) across tool history + references."""
        chars = 0
        for entry in state.tool_history:
            chars += len(str(entry.get("result", "")))
        for sr in state.references.values():
            chars += len(sr.text or "")
        chars += len(state.question)
        return chars // 4
