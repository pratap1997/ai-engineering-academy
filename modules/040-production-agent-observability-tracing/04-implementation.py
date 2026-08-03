import time
import json
import uuid
from typing import Dict, List, Optional, Any

class Span:
    def __init__(self, name: str, parent_id: Optional[str] = None):
        self.span_id = str(uuid.uuid4())
        self.name = name
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.status = "OK"

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def end(self, status: str = "OK"):
        if self.end_time is None:
            self.end_time = time.time()
            self.status = status

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class Trace:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans: List[Span] = []

    def add_span(self, span: Span):
        self.spans.append(span)


class Tracer:
    def __init__(self):
        self.current_trace: Optional[Trace] = None
        self.active_spans: List[Span] = []

    def start_trace(self) -> str:
        trace_id = str(uuid.uuid4())
        self.current_trace = Trace(trace_id)
        self.active_spans = []
        return trace_id

    def start_span(self, name: str) -> Span:
        parent_id = self.active_spans[-1].span_id if self.active_spans else None
        span = Span(name, parent_id)
        if self.current_trace:
            self.current_trace.add_span(span)
        self.active_spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "OK"):
        span.end(status)
        if span in self.active_spans:
            self.active_spans.remove(span)


class TokenCounter:
    @staticmethod
    def count_tokens(text: str) -> int:
        # A simple approximation: 1 token ~ 4 characters
        return len(text) // 4 + (1 if len(text) % 4 != 0 else 0)


class CostCalculator:
    PRICING = {
        "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-3.5": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
    }

    @classmethod
    def calculate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = cls.PRICING.get(model, {"prompt": 0.0, "completion": 0.0})
        return (prompt_tokens * pricing["prompt"]) + (completion_tokens * pricing["completion"])


class SpanExporter:
    @staticmethod
    def export_to_json(trace: Trace) -> str:
        data = {
            "trace_id": trace.trace_id,
            "spans": []
        }
        for span in trace.spans:
            data["spans"].append({
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "name": span.name,
                "duration_ms": span.duration * 1000,
                "attributes": span.attributes,
                "events": span.events,
                "status": span.status
            })
        return json.dumps(data, indent=2)


class AgentProfiler:
    def __init__(self):
        self.latencies: List[float] = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = 0
        self.total_runs = 0

    def record_run(self, trace: Trace):
        self.total_runs += 1
        root_spans = [s for s in trace.spans if s.parent_id is None]
        if root_spans:
            self.latencies.append(root_spans[0].duration)
        
        for span in trace.spans:
            if span.status == "ERROR":
                self.errors += 1
            if "llm.usage.total_tokens" in span.attributes:
                self.total_tokens += span.attributes["llm.usage.total_tokens"]
            if "llm.cost" in span.attributes:
                self.total_cost += span.attributes["llm.cost"]

    def get_p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        if idx >= len(sorted_latencies):
            idx = len(sorted_latencies) - 1
        return sorted_latencies[idx]
