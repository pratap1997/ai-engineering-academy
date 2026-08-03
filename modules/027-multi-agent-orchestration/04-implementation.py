import time
from dataclasses import dataclass
from typing import Callable, Any, Optional

@dataclass
class Message:
    """A message passed between agents."""
    sender: str
    recipient: str  # "*" for broadcast
    content: str
    message_type: str  # "task", "result", "query", "response", "terminate"
    timestamp: int

class MessageBus:
    """Central message bus for agent communication."""
    def __init__(self):
        self._inboxes: dict[str, list[Message]] = {}

    def register_agent(self, agent_id: str) -> None:
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = []

    def send(self, message: Message) -> None:
        if message.recipient == "*":
            self.broadcast(message.sender, message.content, message.message_type)
            return
            
        if message.recipient in self._inboxes:
            self._inboxes[message.recipient].append(message)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")

    def receive(self, agent_id: str) -> list[Message]:
        if agent_id not in self._inboxes:
            return []
        messages = self._inboxes[agent_id]
        self._inboxes[agent_id] = []  # Clear inbox after reading
        return messages

    def broadcast(self, sender: str, content: str, msg_type: str = "broadcast") -> None:
        for agent_id in self._inboxes:
            if agent_id != sender:
                self._inboxes[agent_id].append(
                    Message(
                        sender=sender,
                        recipient=agent_id,
                        content=content,
                        message_type=msg_type,
                        timestamp=int(time.time())
                    )
                )

class BaseAgent:
    """Base class for all agents."""
    def __init__(self, agent_id: str, role: str, llm_fn: Callable[[str], str], bus: MessageBus):
        self.agent_id = agent_id
        self.role = role
        self.llm_fn = llm_fn
        self.bus = bus
        self.bus.register_agent(agent_id)
        self.active = True

    def send_message(self, recipient: str, content: str, msg_type: str) -> None:
        msg = Message(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=msg_type,
            timestamp=int(time.time())
        )
        self.bus.send(msg)

    def receive_messages(self) -> list[Message]:
        return self.bus.receive(self.agent_id)

    def think(self, task: str, messages: list[Message]) -> str:
        context = f"Task: {task}\nMessages: {[m.content for m in messages]}"
        return self.llm_fn(context)

    def run_step(self) -> bool:
        # Default behavior: process one message if available
        messages = self.receive_messages()
        if messages:
            for msg in messages:
                if msg.message_type == "terminate":
                    self.active = False
        return self.active

class WorkerAgent(BaseAgent):
    """Specialized worker agent."""
    def __init__(self, agent_id: str, specialization: str, llm_fn: Callable[[str], str], bus: MessageBus):
        super().__init__(agent_id, f"worker-{specialization}", llm_fn, bus)
        self.specialization = specialization
        self.current_task: Optional[str] = None
        self.supervisor_id: Optional[str] = None
        
    def run_step(self) -> bool:
        messages = self.receive_messages()
        for msg in messages:
            if msg.message_type == "terminate":
                self.active = False
            elif msg.message_type == "task":
                self.current_task = msg.content
                self.supervisor_id = msg.sender
                
                # Perform task
                result = self.think(msg.content, messages)
                
                if self.supervisor_id:
                    self.send_message(self.supervisor_id, result, "result")
                self.current_task = None
                
        return self.active

class SupervisorAgent(BaseAgent):
    """Supervisor that assigns tasks and collects results."""
    def __init__(self, agent_id: str, workers: list[str], llm_fn: Callable[[str], str], bus: MessageBus):
        super().__init__(agent_id, "supervisor", llm_fn, bus)
        self.workers = workers
        self.results: dict[str, str] = {}
        self.tasks_assigned = 0
        
    def assign_task(self, task: str, worker_id: str) -> None:
        if worker_id in self.workers:
            self.send_message(worker_id, task, "task")
            self.tasks_assigned += 1

    def collect_results(self) -> dict[str, str]:
        messages = self.receive_messages()
        for msg in messages:
            if msg.message_type == "result":
                self.results[msg.sender] = msg.content
        return self.results

    def synthesize(self, results: dict[str, str]) -> str:
        context = "Synthesize these results: " + str(results)
        return self.llm_fn(context)
        
    def run_step(self) -> bool:
        self.collect_results()
        if len(self.results) >= self.tasks_assigned and self.tasks_assigned > 0:
            return False  # Done when all tasks are completed
        return self.active

class DebaterAgent(BaseAgent):
    """Competitive debater agent for debate-style multi-agent reasoning."""
    def __init__(self, agent_id: str, position: str, llm_fn: Callable[[str], str], bus: MessageBus):
        super().__init__(agent_id, "debater", llm_fn, bus)
        self.position = position
        
    def generate_argument(self, topic: str, opponent_arguments: list[str]) -> str:
        context = f"Topic: {topic}\nPosition: {self.position}\nOpponent arguments: {opponent_arguments}"
        return self.llm_fn(context)
        
    def judge_winner(self, arguments: list[tuple[str, str]]) -> str:
        context = f"Judge the winner based on arguments: {arguments}"
        return self.llm_fn(context)

class HierarchicalOrchestrator:
    """Orchestrates a hierarchy of supervisor + worker agents."""
    def __init__(self, supervisor: SupervisorAgent, workers: list[WorkerAgent], bus: MessageBus):
        self.supervisor = supervisor
        self.workers = workers
        self.bus = bus
        
    def run(self, task: str) -> dict[str, Any]:
        steps = 0
        
        # Initial assignment
        for worker in self.workers:
            self.supervisor.assign_task(f"{task} for {worker.specialization}", worker.agent_id)
            steps += 1
            
        # Run event loop
        active = True
        timeout = 20
        while active and steps < timeout:
            active_workers = False
            for worker in self.workers:
                worker_active = worker.run_step()
                if worker_active:
                    active_workers = True
                    
            supervisor_active = self.supervisor.run_step()
            
            steps += 1
            if not supervisor_active and not active_workers:
                active = False
                
        final_results = self.supervisor.results
        synthesis = self.supervisor.synthesize(final_results)
        
        return {
            "result": synthesis, 
            "steps": steps,
            "messages_sent": steps * 2 # simplified metric
        }

class DebateOrchestrator:
    """Orchestrates a multi-round debate between agents."""
    def __init__(self, debaters: list[DebaterAgent], rounds: int, bus: MessageBus):
        self.debaters = debaters
        self.rounds = rounds
        self.bus = bus
        self.transcript = []
        
    def run_debate(self, topic: str) -> dict[str, Any]:
        for round_num in range(self.rounds):
            round_args = {}
            for debater in self.debaters:
                arg = debater.generate_argument(topic, [str(a) for a in round_args.values()])
                round_args[debater.agent_id] = arg
                self.transcript.append((debater.agent_id, arg))
                
        # Simple judging: first debater is judge here for demo, though typically a separate agent
        if self.debaters:
            judge = self.debaters[0]
            winner = judge.judge_winner([(agent, arg) for agent, arg in self.transcript])
        else:
            winner = "None"
            
        return {
            "winner": winner,
            "reasoning": "Debate concluded.",
            "transcript": self.transcript
        }

class VotingMechanism:
    """Condorcet voting for agent consensus."""
    def __init__(self, agents: list[str]):
        self.agents = agents
        
    def collect_votes(self, options: list[str], agent_preferences: dict[str, list[str]]) -> str:
        # Returns simple majority winner
        if not agent_preferences:
            return ""
            
        first_choices = [prefs[0] for prefs in agent_preferences.values() if prefs]
        if not first_choices:
            return ""
            
        return max(set(first_choices), key=first_choices.count)
        
    def condorcet_winner(self, preferences: dict[str, list[str]]) -> Optional[str]:
        if not preferences:
            return None
            
        all_options = set()
        for prefs in preferences.values():
            all_options.update(prefs)
            
        for option_a in all_options:
            wins_all_matchups = True
            for option_b in all_options:
                if option_a == option_b:
                    continue
                    
                # count how many agents prefer a over b
                a_prefers = 0
                b_prefers = 0
                for agent_id, prefs in preferences.items():
                    if option_a in prefs and option_b in prefs:
                        if prefs.index(option_a) < prefs.index(option_b):
                            a_prefers += 1
                        elif prefs.index(option_b) < prefs.index(option_a):
                            b_prefers += 1
                    elif option_a in prefs:
                        a_prefers += 1
                    elif option_b in prefs:
                        b_prefers += 1
                        
                if a_prefers <= b_prefers:
                    wins_all_matchups = False
                    break
                    
            if wins_all_matchups:
                return option_a
                
        return None

if __name__ == "__main__":
    def dummy_llm(prompt: str) -> str:
        return f"Response based on: {prompt[:30]}..."
        
    bus = MessageBus()
    worker1 = WorkerAgent("w1", "coder", dummy_llm, bus)
    worker2 = WorkerAgent("w2", "reviewer", dummy_llm, bus)
    supervisor = SupervisorAgent("sup1", ["w1", "w2"], dummy_llm, bus)
    
    orchestrator = HierarchicalOrchestrator(supervisor, [worker1, worker2], bus)
    result = orchestrator.run("Build a web server")
    print(f"Hierarchical Orchestration Result: {result}")
