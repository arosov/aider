import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml
import json # Added
from aider.mcp import load_mcp_servers # Added
from aider.mcp.mcp_client_pool import MCPClientPool

MCP_PROFILES_YAML_PATH = Path.home() / ".aider.mcp.profiles.yml"

@dataclass
class MCPProfile:
    name: str
    server_names: List[str]

def load_mcp_profiles() -> Dict[str, MCPProfile]:
    try:
        with open(MCP_PROFILES_YAML_PATH, 'r') as f:
            profiles_data = yaml.safe_load(f)
        if isinstance(profiles_data, list):
            profiles = {}
            for profile_dict in profiles_data:
                if not isinstance(profile_dict, dict):
                    print(f"Warning: Skipping non-dictionary profile entry: {profile_dict}")
                    continue
                try:
                    # Ensure server_names is a list.
                    loaded_server_names = profile_dict.get('server_names')
                    if not isinstance(loaded_server_names, list):
                        profile_name = profile_dict.get('name', 'UnknownProfile')
                        if loaded_server_names is not None: # Log if it was present but wrong type
                            print(
                                f"Warning: 'server_names' for profile '{profile_name}' is not a list"
                                f" (type: {type(loaded_server_names)}). Initializing as empty list."
                            )
                        profile_dict['server_names'] = [] # Default to empty list

                    profile = MCPProfile(**profile_dict)
                    profiles[profile.name] = profile
                except TypeError as e: # Catches issues like missing 'name' or other fields
                    profile_name_for_error = profile_dict.get('name', str(profile_dict))
                    print(f"Warning: Skipping malformed profile entry for '{profile_name_for_error}': {e}")
            return profiles
        elif profiles_data is None: # File is empty
            return {}
        else: # File content is not a list as expected
            # Potentially log this
            print(f"Warning: MCP profiles YAML content is not a list: {MCP_PROFILES_YAML_PATH}")
            return {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        # Potentially log this, e.g., using io if available
        print(f"Warning: Error parsing MCP profiles YAML: {e}")
        return {}

def save_mcp_profiles(profiles: Dict[str, MCPProfile]):
    try:
        profiles_list = [asdict(profile) for profile in profiles.values()]
        with open(MCP_PROFILES_YAML_PATH, 'w') as f:
            yaml.dump(profiles_list, f, sort_keys=False)
    except (IOError, yaml.YAMLError) as e:
        # Potentially log this, e.g., using io if available
        print(f"Warning: Error saving MCP profiles YAML: {e}")

def get_known_mcp_server_names(
    mcpservers_arg: Optional[Any],  # This is args.mcpservers
    mcpservers_file_arg: Optional[str],  # This is args.mcpservers_file
    io: Optional[Any]
) -> List[str]:
    loaded_mcp_objects = []

    # 1. Process mcpservers_arg
    if mcpservers_arg:
        temp_servers_from_arg = []
        if isinstance(mcpservers_arg, dict):
            try:
                # Convert dict to JSON string to pass to load_mcp_servers
                mcp_json_string = json.dumps(mcpservers_arg)
                # Call load_mcp_servers with the JSON string
                temp_servers_from_arg = load_mcp_servers(mcp_servers=mcp_json_string, mcp_servers_file=None, io=io, verbose=False)
            except TypeError as e: # json.dumps might fail
                if io and hasattr(io, 'tool_warning'):
                    io.tool_warning(f"Could not serialize MCP servers dict to JSON: {e}")
        elif isinstance(mcpservers_arg, str):
            # load_mcp_servers will first try mcp_servers as a JSON string.
            # If that doesn't yield results, we then try mcpservers_arg as a file path.
            servers_from_json_str = load_mcp_servers(mcp_servers=mcpservers_arg, mcp_servers_file=None, io=io, verbose=False)
            if servers_from_json_str:
                temp_servers_from_arg = servers_from_json_str
            else:
                # If not a valid JSON string or it was empty, try as a file path
                temp_servers_from_arg = load_mcp_servers(mcp_servers=None, mcp_servers_file=mcpservers_arg, io=io, verbose=False)
        elif isinstance(mcpservers_arg, list):
             if io and hasattr(io, 'tool_warning'):
                io.tool_warning(
                   f"MCP servers config from direct config was a list, skipping. Expected a dictionary or JSON string."
                )
        # Other types for mcpservers_arg will be implicitly ignored or handled by load_mcp_servers

        if temp_servers_from_arg:
            loaded_mcp_objects.extend(temp_servers_from_arg)

    # 2. Process mcpservers_file_arg
    if mcpservers_file_arg:
        # If mcpservers_arg was a path and is the same as mcpservers_file_arg,
        # load_mcp_servers might process it again.
        # The set operation for names at the end handles duplicates.
        servers_from_file_arg = load_mcp_servers(mcp_servers=None, mcp_servers_file=mcpservers_file_arg, io=io, verbose=False)
        if servers_from_file_arg:
            loaded_mcp_objects.extend(servers_from_file_arg)
    
    # Extract unique names from the McpServer objects
    server_names_set = set()
    for server_obj in loaded_mcp_objects:
        if hasattr(server_obj, 'name') and isinstance(server_obj.name, str) and server_obj.name:
            server_names_set.add(server_obj.name)
    
    return sorted(list(server_names_set))

class MCPProfileManager:
    def __init__(self, io, settings):
        self.io = io
        self.settings = settings
        self.profiles: Dict[str, MCPProfile] = {}
        self.active_profile_name: Optional[str] = None
        self.active_mcp_client_pool: Optional[MCPClientPool] = None

    def load_or_initialize_profiles(self):
        self.profiles = load_mcp_profiles()

        mcpservers_arg_val = getattr(self.settings, 'mcp_servers', None)
        mcpservers_file_arg_val = getattr(self.settings, 'mcp_servers_file', None)

        current_known_server_names = get_known_mcp_server_names(
            mcpservers_arg_val,
            mcpservers_file_arg_val,
            self.io
        )

        needs_save = False

        old_all_profile_server_names_sorted = None
        if "all" in self.profiles:
            # self.profiles["all"].server_names is guaranteed to be a list by robust load_mcp_profiles
            old_all_profile_server_names_sorted = sorted(self.profiles["all"].server_names)

        # Create/Update the 'all' profile in the in-memory dictionary to match current reality.
        # This ensures self.profiles["all"] is always up-to-date after this step.
        self.profiles["all"] = MCPProfile(name="all", server_names=current_known_server_names)

        if old_all_profile_server_names_sorted is None:
            # The 'all' profile was not in the loaded file, so we created it.
            needs_save = True
            if self.io and hasattr(self.io, 'tool_output'):
                self.io.tool_output("Initialized default MCP profile 'all'.")
        elif old_all_profile_server_names_sorted != current_known_server_names:
            # The 'all' profile was in the loaded file, but its servers differed from current reality.
            needs_save = True
            if self.io and hasattr(self.io, 'tool_output'):
                self.io.tool_output("Updated MCP profile 'all' with current server configurations.")
        # If old_all_profile_server_names_sorted == current_known_server_names, then the loaded 'all' profile
        # was already correct. No message, needs_save remains False.

        if needs_save:
            save_mcp_profiles(self.profiles)

    def get_profile(self, name: str) -> Optional[MCPProfile]:
        return self.profiles.get(name)

    def list_profile_names(self) -> List[str]:
        return list(self.profiles.keys())

    def list_profiles_details(self) -> List[Tuple[str, List[str]]]:
        return [(profile.name, profile.server_names) for profile in self.profiles.values()]

    def enable_profile(self, profile_name: str, main_model, main_edit_format):
        if self.active_profile_name == profile_name and self.active_mcp_client_pool:
            self.io.tool_output(f"MCP profile '{profile_name}' is already active.")
            return

        if self.active_profile_name is not None:
            self.disable_profile()

        profile_to_enable = self.get_profile(profile_name)

        if not profile_to_enable:
            self.io.tool_error(f"MCP profile '{profile_name}' not found.")
            return

        if not profile_to_enable.server_names:
            self.io.tool_warning(f"MCP profile '{profile_name}' has no servers configured. Profile enabled but no connections made.")
            self.active_profile_name = profile_name
            self.active_mcp_client_pool = None # Ensure it's cleared
            return

        all_server_configs = getattr(self.settings, 'mcpservers', [])
        if not isinstance(all_server_configs, list): # Ensure it's a list if it was some other type
            all_server_configs = []
            
        matched_server_configs = [s_conf for s_conf in all_server_configs if isinstance(s_conf, dict) and s_conf.get('name') in profile_to_enable.server_names]

        if not matched_server_configs:
            self.io.tool_error(f"No configured MCP servers found for profile '{profile_name}'. Check server names and .aider.conf.yml.")
            return

        pool = MCPClientPool(self.io, main_model, main_edit_format, matched_server_configs)
        try:
            # Run the async method to fetch tools in a blocking way
            asyncio.run(pool.fetch_and_cache_tools())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # This case should ideally be avoided by design.
                    # If enable_profile is called from an async context, it should be awaited.
                    # For now, log a warning and proceed without tools if this rare case occurs.
                    self.io.tool_warning("Warning: Could not fetch MCP tools due to running asyncio event loop during profile enabling.")
                else:
                    self.io.tool_error(f"Error fetching MCP tools: {e}")
                    return # Stop if asyncio.run fails not due to existing loop
            else:
                self.io.tool_error(f"Error fetching MCP tools: {e}")
                return # Stop on other runtime errors

        self.active_mcp_client_pool = pool
        self.active_profile_name = profile_name
        self.io.tool_output(f"MCP profile '{profile_name}' enabled with {len(matched_server_configs)} server(s).")

    def disable_profile(self):
        if self.active_mcp_client_pool:
            # Assuming MCPClientPool has a shutdown method.
            # If not, this will need to be added to MCPClientPool.
            if hasattr(self.active_mcp_client_pool, 'shutdown') and callable(getattr(self.active_mcp_client_pool, 'shutdown')):
                self.active_mcp_client_pool.shutdown()
            self.active_mcp_client_pool = None

        if self.active_profile_name:
            self.io.tool_output(f"MCP profile '{self.active_profile_name}' disabled.")
            self.active_profile_name = None
        else:
            self.io.tool_output("No MCP profile was active.")
