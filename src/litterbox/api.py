"""
LitterboxAgent — Python API for the litter box monitoring agent.

Quick start::

    from litterbox import LitterboxAgent

    # Data defaults to ~/.litterbox_monitor/
    agent = LitterboxAgent()

    # Or point at custom locations
    agent = LitterboxAgent(data_dir="/srv/litterbox/data",
                           images_dir="/srv/litterbox/images")

    # Sensor events (tools called directly — no LLM overhead)
    print(agent.record_entry("/path/to/entry.jpg",
                             weight_pre_g=5400, weight_entry_g=8600))
    print(agent.record_exit("/path/to/exit.jpg",
                            weight_exit_g=5480, ammonia_peak_ppb=45))

    # Cat management
    print(agent.register_cat("/path/to/whiskers.jpg", "Whiskers"))
    print(agent.list_cats())

    # Queries (direct tool calls)
    print(agent.get_visits_by_date("2026-03-28"))
    print(agent.get_visits_by_cat("Whiskers"))
    print(agent.get_anomalous_visits())
    print(agent.get_unconfirmed_visits())
    print(agent.get_visit_images(visit_id=5))
    print(agent.confirm_identity(visit_id=5, cat_name="Whiskers"))
    print(agent.retroactive_recognition("Whiskers", "2026-01-01"))

    # Natural language queries via the full LangGraph agent
    response = agent.query("How many times did Whiskers visit this week?")
    print(response)

    # Context manager — ensures the SQLite checkpointer is closed cleanly
    with LitterboxAgent() as agent:
        agent.record_entry(...)
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# System prompt (mirrors litterbox_agent.py)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a litter box monitoring assistant for a household cat health tracking system.

Your responsibilities:
1. REGISTER CAT IMAGES — When a user uploads a reference photo of a cat, call
   register_cat_image(image_path, cat_name). You MUST always have the cat's name
   before registering. If no name is given, ask for it before proceeding.
   After registration succeeds, ALWAYS ask: "When did you get [cat's name]?
   (YYYY-MM-DD format, e.g. 2026-01-15 — used to review any unknown visits that
   may have been this cat before they were registered.)"
   Then call retroactive_recognition(cat_name, since_date) with the date the owner
   provides. If the owner does not know the exact date, ask for their best estimate
   or the month they got the cat and use the first of that month.

2. RECORD ENTRY EVENTS — When the sensor system notifies you of a litter box entry,
   call record_entry(image_path, ...) immediately, passing any sensor readings
   (weight_pre_g, weight_entry_g, ammonia_peak_ppb, methane_peak_ppb) that appear
   in the event message. Omit parameters that are not present in the message.

3. RECORD EXIT EVENTS — When the sensor system notifies you of a litter box exit,
   call record_exit(image_path, ...) immediately, passing any sensor readings
   (weight_exit_g, ammonia_peak_ppb, methane_peak_ppb) that appear in the event
   message. Omit parameters that are not present in the message.

4. CONFIRM IDENTITIES — Help the owner review unconfirmed visits and call
   confirm_identity(visit_id, cat_name) when they identify a cat.

5. RETROACTIVE RECOGNITION — When called directly by the owner (e.g. "scan old
   unknown visits for Whiskers since 2026-02-01"), call
   retroactive_recognition(cat_name, since_date) and report the results.

6. ANSWER QUERIES — Use the available query tools to answer questions about visit
   history, health flags, and cat records.

Important rules:
- Health findings from exit analysis are ALWAYS preliminary. Always remind the owner
  that a licensed veterinarian must review any flagged concerns.
- Never speculate beyond what the tools return.
- Orphan exit records (no matching entry) must always be flagged for human review.
"""


# ---------------------------------------------------------------------------
# LitterboxAgent
# ---------------------------------------------------------------------------

class LitterboxAgent:
    """Python API for the litter box monitoring agent.

    Parameters
    ----------
    data_dir:
        Directory for SQLite databases and the Chroma vector index.
        Defaults to ``~/.litterbox_monitor/data``.
    images_dir:
        Directory tree where cat reference images and visit images are stored.
        Defaults to ``~/.litterbox_monitor/images``.
    openai_api_key:
        Override the ``OPENAI_API_KEY`` environment variable.  If omitted the
        key is read from the environment or a ``.env`` file.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        images_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        load_dotenv()

        if openai_api_key:
            import os
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # --- resolve paths ---------------------------------------------------
        base = Path.home() / ".litterbox_monitor"
        self._data_path = Path(data_dir) if data_dir else base / "data"
        self._images_path = Path(images_dir) if images_dir else base / "images"
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._images_path.mkdir(parents=True, exist_ok=True)
        self._agent_db_path = self._data_path / "agent_memory.db"

        # --- patch module-level path variables before any tools run ----------
        import litterbox.db as _db
        import litterbox.embeddings as _emb
        import litterbox.tools as _tools

        _db.DB_PATH = self._data_path / "litterbox.db"
        _emb.CHROMA_PATH = self._data_path / "chroma"
        _emb._collection = None          # reset cached Chroma collection
        _tools.IMAGES_DIR = self._images_path
        _tools.PROJECT_ROOT = self._images_path.parent

        # --- initialise database schema --------------------------------------
        from litterbox.db import init_db
        init_db()

        # --- open agent checkpointer (lazy agent creation) -------------------
        self._agent_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._saver_ctx = SqliteSaver.from_conn_string(str(self._agent_db_path))
        self._checkpointer = self._saver_ctx.__enter__()
        self._agent = None

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "LitterboxAgent":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the SQLite checkpointer connection."""
        if self._saver_ctx is not None:
            try:
                self._saver_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._saver_ctx = None
            self._checkpointer = None

    # ------------------------------------------------------------------
    # Internal: LangGraph agent (created once, reused)
    # ------------------------------------------------------------------

    def _get_agent(self):
        if self._agent is None:
            from langchain.agents import create_agent
            from langchain.agents.middleware import SummarizationMiddleware
            from litterbox.tools import ALL_TOOLS

            self._agent = create_agent(
                model="gpt-4o",
                system_prompt=_SYSTEM_PROMPT,
                checkpointer=self._checkpointer,
                tools=ALL_TOOLS,
                middleware=[
                    SummarizationMiddleware(
                        model="gpt-4o",
                        trigger=("messages", 10),
                        keep=("messages", 3),
                    )
                ],
            )
        return self._agent

    # ------------------------------------------------------------------
    # Sensor event methods  (tools called directly — no LLM)
    # ------------------------------------------------------------------

    def record_entry(
        self,
        image_path: str,
        weight_pre_g: Optional[float] = None,
        weight_entry_g: Optional[float] = None,
        ammonia_peak_ppb: Optional[float] = None,
        methane_peak_ppb: Optional[float] = None,
    ) -> str:
        """Record a cat entering the litter box.

        Runs the CLIP + GPT-4o identification pipeline and writes a new visit
        record to the database.  Returns a plain-text summary.

        Parameters
        ----------
        image_path:
            Absolute path to the entry image captured by the camera.
        weight_pre_g:
            Box + litter baseline weight before the cat entered (grams).
            Pass ``None`` if no scale is present.
        weight_entry_g:
            Box + litter + cat weight at the moment of entry (grams).
            Pass ``None`` if no scale is present.
        ammonia_peak_ppb:
            Peak NH₃ sensor reading during the entry phase (ppb).
            Pass ``None`` if no gas sensor is present.
        methane_peak_ppb:
            Peak CH₄ sensor reading during the entry phase (ppb).
            Pass ``None`` if no gas sensor is present.
        """
        from litterbox.tools import record_entry as _record_entry
        kwargs: dict = {"image_path": image_path}
        if weight_pre_g is not None:
            kwargs["weight_pre_g"] = weight_pre_g
        if weight_entry_g is not None:
            kwargs["weight_entry_g"] = weight_entry_g
        if ammonia_peak_ppb is not None:
            kwargs["ammonia_peak_ppb"] = ammonia_peak_ppb
        if methane_peak_ppb is not None:
            kwargs["methane_peak_ppb"] = methane_peak_ppb
        return _record_entry.invoke(kwargs)

    def record_exit(
        self,
        image_path: str,
        weight_exit_g: Optional[float] = None,
        ammonia_peak_ppb: Optional[float] = None,
        methane_peak_ppb: Optional[float] = None,
    ) -> str:
        """Record a cat exiting the litter box and run a health analysis.

        Associates the exit with the most recent open visit.  If no open visit
        exists an orphan record is created.  Returns a plain-text summary that
        includes the health analysis result.

        Parameters
        ----------
        image_path:
            Absolute path to the exit image captured by the camera.
        weight_exit_g:
            Box + litter + waste weight after the cat has left (grams).
            Pass ``None`` if no scale is present.
        ammonia_peak_ppb:
            Peak NH₃ sensor reading during/after the exit phase (ppb).
            Pass ``None`` if no gas sensor is present.
        methane_peak_ppb:
            Peak CH₄ sensor reading during/after the exit phase (ppb).
            Pass ``None`` if no gas sensor is present.
        """
        from litterbox.tools import record_exit as _record_exit
        kwargs: dict = {"image_path": image_path}
        if weight_exit_g is not None:
            kwargs["weight_exit_g"] = weight_exit_g
        if ammonia_peak_ppb is not None:
            kwargs["ammonia_peak_ppb"] = ammonia_peak_ppb
        if methane_peak_ppb is not None:
            kwargs["methane_peak_ppb"] = methane_peak_ppb
        return _record_exit.invoke(kwargs)

    # ------------------------------------------------------------------
    # Cat registration
    # ------------------------------------------------------------------

    def register_cat(self, image_path: str, cat_name: str) -> str:
        """Register a reference photo for a named cat.

        Parameters
        ----------
        image_path:
            Absolute path to a clear photo of the cat.
        cat_name:
            The cat's name.  A new cat record is created if the name is new;
            otherwise an additional reference image is added to the existing cat.
        """
        from litterbox.tools import register_cat_image
        return register_cat_image.invoke({"image_path": image_path, "cat_name": cat_name})

    # ------------------------------------------------------------------
    # Identity management
    # ------------------------------------------------------------------

    def confirm_identity(self, visit_id: int, cat_name: str) -> str:
        """Permanently confirm the cat's identity for a given visit.

        Parameters
        ----------
        visit_id:
            The visit number to confirm (as returned by any query tool).
        cat_name:
            The cat's registered name.
        """
        from litterbox.tools import confirm_identity as _confirm_identity
        return _confirm_identity.invoke({"visit_id": visit_id, "cat_name": cat_name})

    def retroactive_recognition(self, cat_name: str, since_date: str) -> str:
        """Re-run the CLIP + GPT-4o pipeline on unknown visits since a given date.

        Useful after registering a new cat to identify visits that occurred before
        the cat was registered.

        Parameters
        ----------
        cat_name:
            The cat's registered name.
        since_date:
            Start date in ``YYYY-MM-DD`` format (typically the date the cat
            arrived in the home).
        """
        from litterbox.tools import retroactive_recognition as _retro
        return _retro.invoke({"cat_name": cat_name, "since_date": since_date})

    # ------------------------------------------------------------------
    # Query methods  (direct tool calls — no LLM)
    # ------------------------------------------------------------------

    def list_cats(self) -> str:
        """Return a summary of all registered cats and their reference image counts."""
        from litterbox.tools import list_cats as _list_cats
        return _list_cats.invoke({})

    def get_visits_by_date(self, date_str: str) -> str:
        """List all litter box visits for a given date.

        Parameters
        ----------
        date_str:
            Date in ``YYYY-MM-DD`` format.
        """
        from litterbox.tools import get_visits_by_date as _gvbd
        return _gvbd.invoke({"date_str": date_str})

    def get_visits_by_cat(self, cat_name: str) -> str:
        """List all visits for a cat (by confirmed or tentative name).

        Parameters
        ----------
        cat_name:
            The cat's registered name.
        """
        from litterbox.tools import get_visits_by_cat as _gvbc
        return _gvbc.invoke({"cat_name": cat_name})

    def get_anomalous_visits(self) -> str:
        """Return all visits flagged as potentially anomalous by the health analysis."""
        from litterbox.tools import get_anomalous_visits as _gav
        return _gav.invoke({})

    def get_unconfirmed_visits(self) -> str:
        """Return all visits that still have a tentative (unconfirmed) cat identity."""
        from litterbox.tools import get_unconfirmed_visits as _guv
        return _guv.invoke({})

    def get_visit_images(self, visit_id: int) -> str:
        """Return the stored image paths (entry and exit) for a given visit.

        Parameters
        ----------
        visit_id:
            The visit number.
        """
        from litterbox.tools import get_visit_images as _gvi
        return _gvi.invoke({"visit_id": visit_id})

    # ------------------------------------------------------------------
    # Natural language queries via the full LangGraph agent
    # ------------------------------------------------------------------

    def query(self, message: str, thread_id: str = "api") -> str:
        """Send a natural language message to the agent and return the response.

        The agent has access to all 11 tools and maintains conversation history
        within the given ``thread_id``.

        Parameters
        ----------
        message:
            Any question or instruction in plain English.
        thread_id:
            Conversation thread identifier.  Use different values to maintain
            separate conversation histories (e.g. per-user or per-session).
            Defaults to ``"api"``.

        Example::

            agent.query("How many times did Whiskers visit in March?")
            agent.query("Show me all visits flagged as anomalous this month")
            agent.query("I want to register this cat. File path: /tmp/mochi.jpg")
        """
        config = {"configurable": {"thread_id": thread_id}}
        response = self._get_agent().invoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )
        parts = []
        for msg in response["messages"]:
            if isinstance(msg, ToolMessage):
                parts.append(f"[tool] {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                parts.append(msg.content)
        return "\n".join(parts)
