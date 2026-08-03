"""
Module 032: Model Context Protocol (MCP) Implementation
A from-scratch implementation of the MCP Client and Server using JSON-RPC 2.0.
"""
import json
import uuid
from typing import Dict, List, Any, Optional, Callable

class JSONRPCMessage:
    @staticmethod
    def create_request(method: str, params: Optional[Dict] = None, msg_id: Optional[str] = None) -> Dict:
        req = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        if msg_id is not None:
            req["id"] = msg_id
        return req

    @staticmethod
    def create_response(msg_id: str, result: Optional[Dict] = None, error: Optional[Dict] = None) -> Dict:
        resp = {"jsonrpc": "2.0", "id": msg_id}
        if error is not None:
            resp["error"] = error
        else:
            resp["result"] = result if result is not None else {}
        return resp

class ProtocolTransporter:
    """Simulates an in-memory transport layer (like STDIO or SSE)."""
    def __init__(self):
        self.server_handler: Optional[Callable] = None
        self.client_handler: Optional[Callable] = None

    def send_to_server(self, message: str):
        if self.server_handler:
            self.server_handler(message)

    def send_to_client(self, message: str):
        if self.client_handler:
            self.client_handler(message)

class MCPServer:
    def __init__(self, transport: ProtocolTransporter):
        self.transport = transport
        self.transport.server_handler = self.handle_message
        self.resources: Dict[str, Dict] = {}
        self.tools: Dict[str, Callable] = {}
        self.initialized = False

    def add_resource(self, uri: str, name: str, content: str):
        self.resources[uri] = {"name": name, "content": content}

    def add_tool(self, name: str, description: str, func: Callable):
        self.tools[name] = {"description": description, "func": func}

    def handle_message(self, message_str: str):
        try:
            msg = json.loads(message_str)
        except json.JSONDecodeError:
            self._send_error(None, -32700, "Parse error")
            return

        msg_id = msg.get("id")
        method = msg.get("method")

        if not method:
            if "result" in msg or "error" in msg:
                return # Ignore responses received by server
            self._send_error(msg_id, -32600, "Invalid Request")
            return

        if method == "initialize":
            self.initialized = True
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"resources": {}, "tools": {}},
                "serverInfo": {"name": "TestServer", "version": "1.0.0"}
            }
            self._send_response(msg_id, result=result)
        elif not self.initialized:
            self._send_error(msg_id, -32002, "Server not initialized")
        elif method == "resources/list":
            resources_list = [{"uri": uri, "name": res["name"]} for uri, res in self.resources.items()]
            self._send_response(msg_id, result={"resources": resources_list})
        elif method == "resources/read":
            uri = msg.get("params", {}).get("uri")
            if uri in self.resources:
                self._send_response(msg_id, result={"contents": [{"uri": uri, "text": self.resources[uri]["content"]}]})
            else:
                self._send_error(msg_id, -32602, "Resource not found")
        elif method == "tools/list":
            tools_list = [{"name": name, "description": tool["description"]} for name, tool in self.tools.items()]
            self._send_response(msg_id, result={"tools": tools_list})
        elif method == "tools/call":
            name = msg.get("params", {}).get("name")
            args = msg.get("params", {}).get("arguments", {})
            if name in self.tools:
                try:
                    res = self.tools[name]["func"](**args)
                    self._send_response(msg_id, result={"content": [{"type": "text", "text": str(res)}]})
                except Exception as e:
                    self._send_response(msg_id, result={"isError": True, "content": [{"type": "text", "text": str(e)}]})
            else:
                self._send_error(msg_id, -32601, "Method not found")
        else:
            self._send_error(msg_id, -32601, "Method not found")

    def _send_response(self, msg_id: str, result: Optional[Dict] = None):
        if msg_id is None:
            return
        resp = JSONRPCMessage.create_response(msg_id, result=result)
        self.transport.send_to_client(json.dumps(resp))

    def _send_error(self, msg_id: Optional[str], code: int, message: str):
        if msg_id is None:
            return
        resp = JSONRPCMessage.create_response(msg_id, error={"code": code, "message": message})
        self.transport.send_to_client(json.dumps(resp))


class MCPClient:
    def __init__(self, transport: ProtocolTransporter):
        self.transport = transport
        self.transport.client_handler = self.handle_message
        self.pending_requests: Dict[str, Dict] = {}

    def handle_message(self, message_str: str):
        msg = json.loads(message_str)
        msg_id = msg.get("id")
        if msg_id in self.pending_requests:
            self.pending_requests[msg_id] = msg

    def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        msg_id = str(uuid.uuid4())
        req = JSONRPCMessage.create_request(method, params, msg_id)
        self.pending_requests[msg_id] = {} # placeholder
        self.transport.send_to_server(json.dumps(req))
        # In a real sync/async implementation we would wait. Here we assume synchronous mock transport processing.
        return self.pending_requests.get(msg_id, {})

    def initialize(self) -> Dict:
        return self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "TestClient", "version": "1.0.0"}
        })

    def list_resources(self) -> Dict:
        return self._send_request("resources/list")

    def read_resource(self, uri: str) -> Dict:
        return self._send_request("resources/read", {"uri": uri})

    def list_tools(self) -> Dict:
        return self._send_request("tools/list")

    def call_tool(self, name: str, arguments: Dict) -> Dict:
        return self._send_request("tools/call", {"name": name, "arguments": arguments})
