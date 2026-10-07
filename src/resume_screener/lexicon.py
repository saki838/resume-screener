"""Term lexicon used for evidence extraction.

Patterns are deliberately specific. Generic words like "AI", "smart" or
"AI-powered" are NOT evidence on their own: "Leveraged AI-assisted development
tools" must not make a React developer look like an AI engineer.
"""
from __future__ import annotations

import re
from functools import lru_cache


def _p(*alternatives: str) -> str:
    return r"(?<![A-Za-z0-9])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9])"


# ---------------------------------------------------------------------------
# Canonical skills -> pattern. Used for matched_skills and score signals.
# ---------------------------------------------------------------------------
SKILLS: dict[str, str] = {
    # Python & backend
    "Python": _p(r"python3?"),
    "FastAPI": _p(r"fast\s?api"),
    "Django": _p(r"django"),
    "Flask": _p(r"flask"),
    "Pydantic": _p(r"pydantic"),
    "SQLAlchemy": _p(r"sqlalchemy"),
    "Celery": _p(r"celery"),
    "Async": _p(r"asyncio", r"async(?:hronous)?", r"aiohttp", r"async\s*/\s*await"),
    "PostgreSQL": _p(r"postgres(?:ql)?", r"pgvector", r"neon"),
    "Redis": _p(r"redis"),
    "MySQL": _p(r"mysql"),
    "MongoDB": _p(r"mongo\s?db", r"mongo"),
    "REST APIs": _p(r"rest(?:ful)?\s?apis?", r"restful"),
    "WebSockets/SSE": _p(r"web\s?sockets?", r"sse", r"server[- ]sent events", r"socket\.io"),
    "pandas": _p(r"pandas"),
    "NumPy": _p(r"numpy"),
    # GenAI / agentic
    "LangChain": _p(r"lang\s?chain"),
    "LangGraph": _p(r"lang\s?graph"),
    "LlamaIndex": _p(r"llama\s?index"),
    "Google ADK": _p(r"google\s+adk", r"adk"),
    "CrewAI": _p(r"crew\s?ai"),
    "AutoGen": _p(r"autogen"),
    "MCP": _p(r"mcp", r"model context protocol"),
    "OpenAI API": _p(r"openai", r"gpt-?[345o][\w.-]*", r"chatgpt"),
    "Gemini": _p(r"gemini"),
    "Claude": _p(r"claude", r"anthropic"),
    "Ollama": _p(r"ollama"),
    "Hugging Face": _p(r"hugging\s?face", r"transformers"),
    "RAG": _p(r"rag", r"retrieval[- ]augmented", r"graph\s?rag"),
    "Vector DB": _p(
        r"vector\s?(?:db|dbs|database|databases|store|stores|search)",
        r"faiss", r"chroma(?:db)?", r"pinecone", r"weaviate", r"qdrant", r"milvus", r"pgvector",
    ),
    "Embeddings": _p(r"embeddings?"),
    "Prompt Engineering": _p(r"prompt\s+engineering"),
    "Fine-tuning": _p(r"fine[- ]?tun(?:e|ed|ing)", r"lora", r"qlora", r"peft"),
    # Classic ML / DL / CV
    "PyTorch": _p(r"pytorch", r"torch"),
    "TensorFlow": _p(r"tensorflow(?:\s+lite)?", r"keras", r"tflite"),
    "scikit-learn": _p(r"scikit[- ]?learn", r"sklearn"),
    "XGBoost": _p(r"xgboost", r"lightgbm"),
    "OpenCV": _p(r"opencv"),
    "YOLO": _p(r"yolo\w*"),
    "Machine Learning": _p(r"machine\s+learning"),
    "Deep Learning": _p(r"deep\s+learning", r"neural\s+networks?", r"cnns?", r"lstms?"),
    "Computer Vision": _p(r"computer\s+vision"),
    "NLP": _p(r"nlp", r"natural\s+language\s+processing"),
    # Cloud / deployment
    "GCP": _p(r"gcp", r"google\s+cloud(?:\s+platform)?", r"cloud\s+run", r"vertex\s?ai", r"bigquery", r"gke"),
    "AWS": _p(r"aws", r"amazon\s+web\s+services", r"lambda", r"ec2", r"s3", r"bedrock", r"sagemaker"),
    "Azure": _p(r"azure"),
    "Docker": _p(r"docker(?:ized|file)?", r"containeri[sz]ed"),
    "Kubernetes": _p(r"kubernetes", r"k8s"),
    "CI/CD": _p(r"ci\s*/\s*cd", r"github\s+actions", r"jenkins", r"gitlab\s+ci"),
    # Frontend
    "React": _p(r"react(?:\.?js)?", r"react\s+native"),
    "Next.js": _p(r"next\.?js"),
    "TypeScript": _p(r"typescript"),
    "JavaScript": _p(r"javascript", r"es6\+?"),
    "Node.js": _p(r"node\.?js", r"nest\.?js", r"express\.?js"),
    # Non-target stacks (reported for transparency, e.g. in rejections)
    "Java": _p(r"java(?!\s*script)"),
    "Spring Boot": _p(r"spring\s?boot", r"spring\s+mvc"),
    "C++": r"(?<![A-Za-z0-9])c\+\+",
    "Go": _p(r"golang"),
    # Engineering
    "Kafka": _p(r"kafka"),
    "RabbitMQ": _p(r"rabbit\s?mq"),
    "pytest": _p(r"pytest"),
}

# Frameworks that imply Python is the implementation language.
PYTHON_IMPLIED = ["FastAPI", "Django", "Flask", "Pydantic", "SQLAlchemy", "Celery", "pandas",
                  "NumPy", "PyTorch", "scikit-learn", "LangGraph", "LlamaIndex", "Google ADK",
                  "CrewAI", "AutoGen"]

GENAI_FRAMEWORKS = ["LangChain", "LangGraph", "LlamaIndex", "Google ADK", "CrewAI", "AutoGen", "MCP"]

# ---------------------------------------------------------------------------
# AI evidence categories. A project/experience entry is classified by which of
# these it contains. "llm" alone (a model call) is the thin-wrapper signal; the
# remaining GenAI categories describe *depth*.
# ---------------------------------------------------------------------------
AI_SIGNALS: dict[str, str] = {
    "llm": _p(
        r"llms?", r"large\s+language\s+models?", r"gen\s?ai", r"generative\s+ai", r"openai",
        r"gpt-?[345o][\w.-]*", r"chatgpt", r"gemini", r"claude", r"anthropic", r"llama\s?\d*",
        r"mistral\w*", r"qwen[\w.:-]*", r"ollama", r"groq", r"lang\s?chain",
        r"prompt(?:s|ing)?", r"bedrock", r"vertex\s?ai", r"nl[- ]to[- ]sql", r"text[- ]to[- ]sql",
        r"chat\s?bots?",
    ),
    "agentic": _p(
        r"multi[- ]agents?", r"tool[- ]calling", r"tool[- ]use",
        r"function[- ]calling", r"lang\s?graph", r"crew\s?ai", r"autogen", r"google\s+adk", r"adk",
        r"mcp", r"model context protocol", r"react\s+agent", r"supervisor", r"planner",
        r"human[- ]in[- ]the[- ]loop",
    ),
    "retrieval": _p(
        r"rag", r"retrieval[- ]augmented", r"vector\s?(?:db|dbs|database|databases|store|stores|search|index)",
        r"embeddings?", r"pgvector", r"faiss", r"chroma(?:db)?", r"pinecone", r"weaviate", r"qdrant",
        r"milvus", r"llama\s?index", r"semantic\s+search", r"hybrid\s+search", r"bm25", r"graph\s?rag",
    ),
    # Bare "agent"/"agentic" in a project *title* ("Stock News AI Agent") proves little;
    # these count only when they occur in the entry's bullets. Resolved in evidence.py.
    "agentic_weak": _p(r"agents?", r"agentic", r"ai\s+agents?"),
    # Generic words ("data retrieval", "chunked uploads") that only mean RAG-style
    # retrieval when the same entry also involves an LLM. Resolved in evidence.py.
    "retrieval_weak": _p(
        r"retrieval", r"retriever", r"chunk\w*", r"re-?rank\w*", r"knowledge\s+graph", r"neo4j",
    ),
    "state_memory": _p(
        r"memory", r"state(?:ful)?", r"state\s?graph", r"checkpoint\w*", r"session\w*",
        r"conversation(?:al)?\s+history", r"context\s+management", r"multi[- ]turn",
    ),
    "evaluation": _p(
        r"evals?", r"evaluat\w*", r"benchmark\w*", r"accuracy", r"precision", r"recall", r"f1",
        r"ragas", r"llm[- ]as[- ]a?[- ]?judge", r"hallucinat\w*", r"guardrails?", r"ground(?:ed|ing)",
    ),
    "data_pipeline": _p(
        r"ingest\w*", r"pipelines?", r"etl", r"pre-?process\w*", r"pars(?:e|ing|er)", r"ocr",
        r"extract(?:s|ion|ed)?", r"structured\s+(?:output|data)", r"schemas?", r"json\s+schema",
    ),
    "backend": _p(
        r"fast\s?api", r"django", r"flask", r"rest(?:ful)?\s+apis?", r"endpoints?", r"backend", r"postgres(?:ql)?",
        r"redis", r"database", r"sql", r"websockets?", r"sse", r"streaming", r"microservices?",
    ),
    "training": _p(
        r"fine[- ]?tun\w*", r"lora", r"qlora", r"peft", r"trained", r"training", r"distill\w*",
    ),
    "classic_ml": _p(
        r"machine\s+learning", r"deep\s+learning", r"neural\s+networks?", r"cnns?", r"lstms?",
        r"rnns?", r"tensorflow(?:\s+lite)?", r"keras", r"pytorch", r"scikit[- ]?learn", r"sklearn",
        r"xgboost", r"lightgbm", r"random\s+forest", r"svm", r"svr", r"decision\s+tree\w*",
        r"opencv", r"yolo\w*", r"computer\s+vision", r"object\s+detection", r"image\s+classification",
        r"nlp", r"natural\s+language\s+processing", r"mobilenet", r"resnet", r"bert",
        r"regression\s+model", r"classifier", r"predictive\s+model\w*", r"ml\s+models?",
        r"hugging\s?face", r"transformers?",
    ),
}

GENAI_CATEGORIES = ("llm", "agentic", "retrieval")
DEPTH_CATEGORIES = ("agentic", "retrieval", "state_memory", "evaluation", "data_pipeline",
                    "backend", "training")

# Engineering-depth signals (5-point category).
ENGINEERING_SIGNALS: dict[str, str] = {
    "testing": _p(r"pytest", r"unit\s+tests?", r"integration\s+tests?", r"test\s+(?:suite|coverage|cases)",
                  r"tdd", r"unit[- ]tested", r"e2e\s+tests?"),
    "caching": _p(r"cach\w+", r"lru"),
    "queues": _p(r"kafka", r"rabbit\s?mq", r"celery", r"bullmq", r"sqs", r"pub\s?/\s?sub", r"queues?",
                 r"background\s+(?:jobs?|workers?|tasks?)"),
    "observability": _p(r"logging", r"monitoring", r"prometheus", r"grafana", r"opentelemetry",
                        r"tracing", r"sentry", r"observability", r"alerting", r"langsmith", r"langfuse"),
    "concurrency": _p(r"asyncio", r"async(?:hronous)?", r"concurren\w+", r"parallel\w*", r"multi-?thread\w*",
                      r"multiprocessing"),
    "failure_handling": _p(r"retr(?:y|ies)", r"fallbacks?", r"circuit\s+breakers?", r"rate[- ]limit\w*",
                           r"idempoten\w+", r"graceful\w*", r"fault[- ]toleran\w+", r"backoff", r"timeouts?"),
    "architecture": _p(r"microservices?", r"event[- ]driven", r"system\s+design", r"architect\w*",
                       r"distributed"),
}

DEPLOYMENT_SIGNAL = _p(r"deploy\w*", r"production", r"live\s+(?:demo|link|app|url)?", r"hosted",
                       r"render", r"vercel", r"railway", r"heroku", r"hugging\s?face\s+spaces")

# Phrases where "AI" is about tooling the candidate *used*, not built.
AI_TOOLING_NOISE = re.compile(
    r"(ai[- ]assisted\s+development|github\s+copilot|ai\s+(?:development\s+)?tools|cursor\s+ai)", re.I
)


@lru_cache(maxsize=None)
def compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


def find_terms(text: str, table: dict[str, str]) -> set[str]:
    return {name for name, pat in table.items() if compiled(pat).search(text)}


def count_terms(text: str, pattern: str) -> int:
    return len(compiled(pattern).findall(text))
