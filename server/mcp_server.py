from mcp.server import Server
from mcp.types import Tool, TextContent
import json

from .engine.tools import execute_tool, VALID_TOOLS

class ShopEasyMCPServer:
    def __init__(self, db):
        self.db = db
        self.app = Server("shopeasy-support")
        self._verified_facts = {}
        self._ticket_id = "MCP-INTERNAL"

        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="lookup_order",
                    description="Look up an order by order_id to retrieve full order details.",
                    inputSchema={
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"]
                    }
                ),
                Tool(
                    name="process_refund",
                    description="Process a refund for an order.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                            "amount": {"type": "number"}
                        },
                        "required": ["order_id", "reason"]
                    }
                ),
                Tool(
                    name="search_kb",
                    description="Search the knowledge base.",
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="escalate_to_human",
                    description="Escalate the ticket to a human supervisor.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "priority": {"type": "string", "enum": ["normal", "high"]}
                        },
                        "required": ["reason"]
                    }
                ),
                Tool(
                    name="cancel_subscription",
                    description="Cancel a customer subscription.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "subscription_id": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["subscription_id", "reason"]
                    }
                ),
                Tool(
                    name="check_payment",
                    description="Check payment details for an order (duplicate charge detection).",
                    inputSchema={
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"]
                    }
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name not in VALID_TOOLS:
                return [TextContent(type="text", text=json.dumps({
                    "success": False, 
                    "error": f"Unknown tool '{name}'"
                }))]
                
            try:
                result = execute_tool(
                    tool_name=name,
                    tool_args=arguments or {},
                    db=self.db,
                    verified_facts=self._verified_facts,
                    ticket_id=self._ticket_id,
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False, 
                    "error": str(e)
                }))]
