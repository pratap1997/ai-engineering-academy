"""
Module 032: MCP Experiments
"""
from importlib.util import spec_from_file_location, module_from_spec
import sys
import os

# Load implementation
impl_path = os.path.join(os.path.dirname(__file__), "04-implementation.py")
spec = spec_from_file_location("module_032_impl", impl_path)
impl = module_from_spec(spec)
sys.modules["module_032_impl"] = impl
spec.loader.exec_module(impl)

def setup_environment():
    transport = impl.ProtocolTransporter()
    server = impl.MCPServer(transport)
    client = impl.MCPClient(transport)
    
    server.add_resource("file:///test.txt", "Test File", "Hello MCP")
    
    def greet(name: str):
        return f"Hello, {name}!"
        
    server.add_tool("greet", "Greets someone", greet)
    return server, client

def exp1_handshake():
    print("Experiment 1: Protocol Handshake")
    server, client = setup_environment()
    res = client.initialize()
    print("Initialize Response:", res)
    assert "result" in res
    assert res["result"]["protocolVersion"] == "2024-11-05"

def exp2_resources():
    print("\nExperiment 2: Resource Reading")
    server, client = setup_environment()
    client.initialize()
    res_list = client.list_resources()
    print("List Resources:", res_list)
    read_res = client.read_resource("file:///test.txt")
    print("Read Resource:", read_res)

def exp3_tools():
    print("\nExperiment 3: Tool Execution")
    server, client = setup_environment()
    client.initialize()
    tool_list = client.list_tools()
    print("List Tools:", tool_list)
    call_res = client.call_tool("greet", {"name": "Alice"})
    print("Call Tool:", call_res)

def exp4_error_handling():
    print("\nExperiment 4: Error Handling")
    server, client = setup_environment()
    client.initialize()
    err_res = client.call_tool("unknown_tool", {})
    print("Unknown Tool Error:", err_res)
    assert "error" in err_res

if __name__ == "__main__":
    exp1_handshake()
    exp2_resources()
    exp3_tools()
    exp4_error_handling()
