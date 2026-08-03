import pytest
import json
from importlib.util import spec_from_file_location, module_from_spec
import sys
import os

# Load implementation
impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04-implementation.py"))
spec = spec_from_file_location("module_032_impl", impl_path)
impl = module_from_spec(spec)
sys.modules["module_032_impl"] = impl
spec.loader.exec_module(impl)

@pytest.fixture
def mcp_env():
    transport = impl.ProtocolTransporter()
    server = impl.MCPServer(transport)
    client = impl.MCPClient(transport)
    return server, client

# Category 1: Protocol Handshake (4 tests)
def test_handshake_success(mcp_env):
    server, client = mcp_env
    res = client.initialize()
    assert "result" in res
    assert res["result"]["protocolVersion"] == "2024-11-05"

def test_uninitialized_error(mcp_env):
    server, client = mcp_env
    res = client.list_tools()
    assert "error" in res
    assert res["error"]["code"] == -32002

def test_handshake_state_transition(mcp_env):
    server, client = mcp_env
    assert not server.initialized
    client.initialize()
    assert server.initialized

def test_invalid_jsonrpc_format(mcp_env):
    server, client = mcp_env
    server.transport.send_to_server("invalid json")
    # client should not crash, error is sent back but unparseable id means it's discarded by client logic easily
    assert True

# Category 2: Resource Management (4 tests)
def test_resource_listing(mcp_env):
    server, client = mcp_env
    server.add_resource("file:///a.txt", "A", "Data A")
    client.initialize()
    res = client.list_resources()
    assert len(res["result"]["resources"]) == 1
    assert res["result"]["resources"][0]["uri"] == "file:///a.txt"

def test_resource_reading_success(mcp_env):
    server, client = mcp_env
    server.add_resource("file:///a.txt", "A", "Data A")
    client.initialize()
    res = client.read_resource("file:///a.txt")
    assert res["result"]["contents"][0]["text"] == "Data A"

def test_resource_reading_not_found(mcp_env):
    server, client = mcp_env
    client.initialize()
    res = client.read_resource("file:///unknown.txt")
    assert "error" in res
    assert res["error"]["code"] == -32602

def test_resource_multiple(mcp_env):
    server, client = mcp_env
    server.add_resource("file:///a.txt", "A", "Data A")
    server.add_resource("file:///b.txt", "B", "Data B")
    client.initialize()
    res = client.list_resources()
    assert len(res["result"]["resources"]) == 2

# Category 3: Tool Dispatching (4 tests)
def test_tool_listing(mcp_env):
    server, client = mcp_env
    server.add_tool("add", "Add nums", lambda x, y: x + y)
    client.initialize()
    res = client.list_tools()
    assert len(res["result"]["tools"]) == 1
    assert res["result"]["tools"][0]["name"] == "add"

def test_tool_execution_success(mcp_env):
    server, client = mcp_env
    server.add_tool("add", "Add nums", lambda x, y: x + y)
    client.initialize()
    res = client.call_tool("add", {"x": 2, "y": 3})
    assert res["result"]["content"][0]["text"] == "5"

def test_tool_execution_not_found(mcp_env):
    server, client = mcp_env
    client.initialize()
    res = client.call_tool("unknown", {})
    assert "error" in res
    assert res["error"]["code"] == -32601

def test_tool_execution_exception(mcp_env):
    server, client = mcp_env
    def fail_func():
        raise ValueError("Oops")
    server.add_tool("fail", "Fails", fail_func)
    client.initialize()
    res = client.call_tool("fail", {})
    assert res["result"]["isError"] is True
    assert "Oops" in res["result"]["content"][0]["text"]

# Category 4: Error & Transport (4 tests)
def test_jsonrpc_message_creation_request():
    req = impl.JSONRPCMessage.create_request("test", {"a": 1}, "id1")
    assert req["jsonrpc"] == "2.0"
    assert req["method"] == "test"
    assert req["id"] == "id1"

def test_jsonrpc_message_creation_response():
    resp = impl.JSONRPCMessage.create_response("id1", result={"val": 1})
    assert resp["jsonrpc"] == "2.0"
    assert resp["result"]["val"] == 1

def test_transport_bidirectional():
    transport = impl.ProtocolTransporter()
    srv_received = []
    cli_received = []
    transport.server_handler = lambda m: srv_received.append(m)
    transport.client_handler = lambda m: cli_received.append(m)
    
    transport.send_to_server("hello server")
    transport.send_to_client("hello client")
    
    assert srv_received == ["hello server"]
    assert cli_received == ["hello client"]

def test_server_ignores_responses(mcp_env):
    server, client = mcp_env
    # Server should ignore messages that don't have a 'method' (i.e. responses)
    # Just asserting it doesn't crash or send back an error
    server.transport.send_to_server(json.dumps({"jsonrpc": "2.0", "id": "1", "result": {}}))
    assert True
