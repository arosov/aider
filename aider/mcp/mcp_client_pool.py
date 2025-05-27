import asyncio
import logging
from typing import List, Dict, Any, Optional

from aider.mcp.server import McpServer # Assuming McpServer is in this path


class MCPClientPool:
    """
    Manages a pool of MCPClient instances, one for each configured MCP server.
    Provides a way to get available tools from all connected servers and
    to shut down all connections gracefully.
    """

    def __init__(self, io: Any, model: Any, edit_format: str, server_configs: List[Dict[str, Any]]):
        """
        Initializes the MCPClientPool.

        Args:
            io: The input/output object for logging and user interaction.
            model: The main language model being used.
            edit_format: The edit format being used.
            server_configs: A list of configurations for each MCP server.
        """
        self.io = io
        self.model = model
        self.edit_format = edit_format
        self.server_configs = server_configs
        self.clients: Dict[str, McpServer] = {}
        self._init_clients()

    def _init_clients(self):
        """
        Initializes McpServer instances based on the server configurations.
        """
        for config in self.server_configs:
            server_name = config.get("name")
            if not server_name:
                logging.warning("MCP server config missing 'name', skipping.")
                if self.io and hasattr(self.io, 'tool_warning'):
                    self.io.tool_warning("An MCP server configuration was missing a 'name' and has been skipped.")
                continue
            
            if server_name in self.clients:
                logging.warning(f"Duplicate MCP server name '{server_name}' found in configuration, skipping.")
                if self.io and hasattr(self.io, 'tool_warning'):
                    self.io.tool_warning(f"Duplicate MCP server name '{server_name}' found. Only the first configuration will be used.")
                continue

            try:
                self.clients[server_name] = McpServer(server_config=config)
                # TODO: Potentially connect here or connect on-demand when tools are requested.
                # For now, just initializing. Connection will happen when tools are fetched.
                logging.info(f"Initialized McpServer for '{server_name}'.")
            except Exception as e:
                logging.error(f"Failed to initialize McpServer for '{server_name}': {e}")
                if self.io and hasattr(self.io, 'tool_error'):
                    self.io.tool_error(f"Failed to initialize MCP server '{server_name}': {e}")

    async def get_all_tools(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Connects to all MCP servers and retrieves their available tools.

        Args:
            context: Optional context to pass to the MCP servers.

        Returns:
            A list of tool definitions from all connected MCP servers.
        """
        all_tools = []
        # TODO: Implement tool fetching logic. This will involve:
        # 1. Iterating through self.clients.
        # 2. For each client, ensuring it's connected (await client.connect()).
        # 3. Calling a method on the client/session to get tools (e.g., await client.session.experimental_get_tools(context=context, format="openai_function"))
        # 4. Aggregating the tools.
        # 5. Handling potential errors during connection or tool retrieval for individual servers.

        # Placeholder implementation:
        if self.io and hasattr(self.io, 'tool_output'):
            self.io.tool_output("MCPClientPool.get_all_tools() is not fully implemented yet.")
        
        # Example of how one might connect and get tools (needs actual mcp.py library methods)
        # for server_name, client in self.clients.items():
        #     try:
        #         session = await client.connect()
        #         # Assuming a method like `experimental_get_tools` exists on the session
        #         # The actual method name and parameters might differ based on mcp.py
        #         server_tools = await session.experimental_get_tools(format="openai_function", context=context)
        #         if server_tools:
        #             all_tools.extend(server_tools)
        #         logging.info(f"Retrieved {len(server_tools)} tools from {server_name}")
        #     except Exception as e:
        #         logging.error(f"Error getting tools from MCP server {server_name}: {e}")
        #         if self.io and hasattr(self.io, 'tool_error'):
        #             self.io.tool_error(f"Could not retrieve tools from MCP server '{server_name}': {e}")
        return all_tools

    def shutdown(self):
        """
        Shuts down all MCP client connections.
        This method is synchronous but calls an asynchronous helper.
        """
        logging.info("Shutting down MCPClientPool...")
        try:
            asyncio.run(self._shutdown_async())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # If we're already in an event loop, schedule _shutdown_async
                # This might happen if shutdown is called from an async context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._shutdown_async())
                    logging.info("Scheduled MCPClientPool shutdown in existing event loop.")
                else: # Should not happen if RuntimeError is for running loop
                    logging.error(f"MCPClientPool shutdown error: {e}")
            else:
                logging.error(f"MCPClientPool shutdown error: {e}")
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"Unexpected error during MCPClientPool shutdown: {e}")


    async def _shutdown_async(self):
        """
        Asynchronously disconnects all McpServer instances.
        """
        logging.info("Starting asynchronous shutdown of MCP clients.")
        shutdown_tasks = []
        for server_name, client in self.clients.items():
            logging.info(f"Initiating shutdown for MCP server: {server_name}")
            shutdown_tasks.append(client.disconnect())
        
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        for server_name, result in zip(self.clients.keys(), results):
            if isinstance(result, Exception):
                logging.error(f"Error disconnecting from MCP server {server_name}: {result}")
                if self.io and hasattr(self.io, 'tool_warning'):
                    self.io.tool_warning(f"Error during disconnect from MCP server '{server_name}': {result}")
            else:
                logging.info(f"Successfully disconnected from MCP server: {server_name}")
        
        self.clients.clear()
        logging.info("All MCP clients have been processed for shutdown.")

    # TODO: Add methods for coder to interact with tools, e.g.:
    # async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
    #     pass
    # This would require routing the call to the correct McpServer instance.
