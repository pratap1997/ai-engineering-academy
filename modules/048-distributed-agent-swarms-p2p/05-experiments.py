import time
import importlib.util
import os

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "04-implementation.py"))
spec = importlib.util.spec_from_file_location("module_048_impl", module_path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

def run_experiments():
    print("Experiment 1: Gossip convergence")
    swarm = impl.SwarmNetwork(10)
    gossip = impl.GossipProtocol(swarm.nodes)
    gossip.propagate("update", "v1", 3)

    print("Experiment 2: Raft Leader Election")
    raft = impl.RaftConsensus(swarm.nodes)
    raft.elect_leader()
    
    print("Experiment 3: Byzantine node filtering accuracy")
    print("Experiment 4: Network message overhead")

if __name__ == "__main__":
    run_experiments()
