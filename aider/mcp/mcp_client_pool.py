import asyncio
import logging
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

from aider.mcp.server import McpServer # Assuming McpServer is in this path
from aider.mcp.tool_filter import filter_tools_for_server


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
        self.cached_tools_by_server: Optional[Dict[str, List[Dict[str, Any]]]] = None
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

    async def _fetch_tools_for_server(self, server_name: str, client: McpServer, context: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Helper to fetch tools for a single server."""
        from litellm import experimental_mcp_client # Local import to avoid circular dependency if mcp.py imports aider stuff
        try:
            session = await client.connect()
            # The context parameter for load_mcp_tools is not standard, removing for now.
            # If needed, it would be passed to experimental_get_tools if that was the target.
            server_tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
            logging.info(f"Retrieved {len(server_tools)} tools from {server_name}")
            return server_tools
        except Exception as e:
            logging.error(f"Error getting tools from MCP server {server_name}: {e}")
            if self.io and hasattr(self.io, 'tool_error'):
                self.io.tool_error(f"Could not retrieve tools from MCP server '{server_name}': {e}")
            return None
        finally:
            # Ensure client is disconnected even if tool fetching fails
            await client.disconnect()


    async def _fetch_tools_for_all_servers(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Connects to all MCP servers and retrieves their available tools.

        Args:
            context: Optional context to pass to the MCP servers (currently unused by load_mcp_tools).

        Returns:
            A dictionary mapping server names to their list of tool definitions.
        """
        tools_by_server: Dict[str, List[Dict[str, Any]]] = {}
        
        tasks = []
        for server_name, client in self.clients.items():
            tasks.append(self._fetch_tools_for_server(server_name, client, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, server_name in enumerate(self.clients.keys()):
            result = results[i]
            if isinstance(result, Exception):
                logging.error(f"Exception fetching tools from {server_name}: {result}")
                # Optionally store empty list or skip this server
            elif result is not None:
                tools_by_server[server_name] = result
        
        return tools_by_server

    async def fetch_and_cache_tools(self, context: Optional[Dict[str, Any]] = None):
        """
        Fetches tools from all configured servers and caches them.
        """
        if self.io and hasattr(self.io, 'tool_output'):
            self.io.tool_output("Fetching tools from enabled MCP profile servers...")
        self.cached_tools_by_server = await self._fetch_tools_for_all_servers(context)
        if self.io and hasattr(self.io, 'tool_output'):
            num_tools = sum(len(tools) for tools in self.cached_tools_by_server.values())
            num_servers = len(self.cached_tools_by_server)
            self.io.tool_output(f"Fetched {num_tools} tool(s) from {num_servers} server(s).")


    def get_cached_tools_flat_list(self, mcp_profile_manager) -> List[Dict[str, Any]]:
        """
        Returns a flat list of tool definitions from all servers in the pool,
        filtered by the enabled_tools setting in the active MCP profile for each server.
        """
        all_effective_tools_flat = []
        if self.cached_tools_by_server and mcp_profile_manager and mcp_profile_manager.active_profile_name:
            for server_name, tools_definitions_for_server in self.cached_tools_by_server.items():
                if tools_definitions_for_server: # Ensure the list is not empty
                    # Get enabled tool names for this server from the active profile
                    enabled_tool_names = mcp_profile_manager.get_enabled_tools_for_server(server_name)
                    
                    # Filter the tools for this server
                    filtered_server_tools = filter_tools_for_server(tools_definitions_for_server, enabled_tool_names)
                    all_effective_tools_flat.extend(filtered_server_tools)
        
        # Basic deduplication by function name, preferring the first encountered.
        # More sophisticated deduplication might be needed if tools can have same name but different defs.
        seen_tool_names = set()
        deduplicated_tools = []
        for tool_def in all_effective_tools_flat:
            try:
                if isinstance(tool_def, dict) and isinstance(tool_def.get("function"), dict):
                    tool_name = tool_def["function"].get("name")
                    if tool_name and tool_name not in seen_tool_names:
                        seen_tool_names.add(tool_name)
                        deduplicated_tools.append(tool_def)
            except (KeyError, TypeError):
                continue # Skip malformed tool_def
        return deduplicated_tools

    def get_effective_tools_and_tool_choice(
        self,
        mcp_profile_manager, # MCPProfileManager instance
        current_tool_choice: Optional[Any], # The tool_choice that would be used if no MCP filtering
        io: Optional[Any] # For logging warnings
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Gets the list of MCP tool definitions, filtered by enabled status in the profile,
        and logs a warning if current_tool_choice refers to a disabled MCP tool.

        Args:
            mcp_profile_manager: The MCPProfileManager to get enabled tool lists.
            current_tool_choice: The tool_choice value that would be active.
            io: InputOutput object for logging.

        Returns:
            A tuple containing:
                - List of effective (filtered) MCP tool definitions.
                - The original current_tool_choice (adjustment is for logging only here).
        """
        effective_mcp_tools_flat_list = self.get_cached_tools_flat_list(mcp_profile_manager)
        
        # Check if the current_tool_choice (if it's specific) refers to an MCP tool that got disabled.
        if isinstance(current_tool_choice, dict) and current_tool_choice.get("type") == "function":
            function_spec = current_tool_choice.get("function")
            if isinstance(function_spec, dict):
                chosen_tool_name = function_spec.get("name")
                if chosen_tool_name:
                    # Check if this chosen tool was an MCP tool by seeing if it was defined by any MCP server
                    # (regardless of its current enabled status).
                    was_an_mcp_tool = False
                    if self.cached_tools_by_server:
                        for _, tools_definitions_for_server in self.cached_tools_by_server.items():
                            if tools_definitions_for_server:
                                for tool_def in tools_definitions_for_server:
                                    try:
                                        if isinstance(tool_def, dict) and \
                                           isinstance(tool_def.get("function"), dict) and \
                                           tool_def["function"].get("name") == chosen_tool_name:
                                            was_an_mcp_tool = True
                                            break
                                    except (KeyError, TypeError):
                                        continue
                            if was_an_mcp_tool:
                                break
                    
                    if was_an_mcp_tool:
                        # Now check if this chosen MCP tool is in the *effective* (enabled) list
                        is_chosen_mcp_tool_enabled = False
                        for tool_def in effective_mcp_tools_flat_list:
                            try:
                                if isinstance(tool_def, dict) and \
                                   isinstance(tool_def.get("function"), dict) and \
                                   tool_def["function"].get("name") == chosen_tool_name:
                                    is_chosen_mcp_tool_enabled = True
                                    break
                            except (KeyError, TypeError):
                                continue
                        
                        if not is_chosen_mcp_tool_enabled:
                            if io and hasattr(io, 'tool_warning'):
                                io.tool_warning(
                                    f"Warning: Tool '{chosen_tool_name}' is specified by `tool_choice` but is not "
                                    f"enabled for any server in the active MCP profile. "
                                    f"The `tool_choice` may effectively be 'auto' or 'none' by the LLM."
                                )
        
        return effective_mcp_tools_flat_list, current_tool_choice

    def get_cached_tools_by_server(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Returns the cached tools, structured as a dictionary mapping server name to its tools.
        """
        return self.cached_tools_by_server

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
