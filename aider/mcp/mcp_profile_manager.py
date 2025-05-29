import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml
import json # Added
from aider.mcp import load_mcp_servers # Added
from aider.mcp.mcp_client_pool import MCPClientPool
from aider.coders.base_prompts import CoderPrompts


MCP_PROFILES_YAML_PATH = Path.home() / ".aider.mcp.profiles.yml"

# Constants for YAML keys
PROFILE_NAME_KEY = "name"
PROFILE_SERVERS_KEY = "servers"
SERVER_NAME_KEY = "name"
SERVER_NO_CONFIRM_KEY = "no_confirm"
# Old key, for removal during load if present
OLD_PROFILE_SERVER_NAMES_KEY = "server_names"
# New top-level keys for the YAML file structure
YAML_PROFILES_KEY = "profiles"
YAML_ACTIVE_PROFILE_KEY = "active_profile_name"


@dataclass
class MCPProfile:
    name: str
    servers: List[Dict[str, Any]] # Each dict: {'name': str, 'no_confirm': bool}

def load_mcp_profiles() -> Tuple[Dict[str, MCPProfile], Optional[str]]:
    profiles: Dict[str, MCPProfile] = {}
    persisted_active_profile_name: Optional[str] = None
    try:
        with open(MCP_PROFILES_YAML_PATH, 'r') as f:
            raw_data = yaml.safe_load(f)

        profiles_list_from_yaml = []

        if isinstance(raw_data, dict):
            persisted_active_profile_name = raw_data.get(YAML_ACTIVE_PROFILE_KEY)
            profiles_list_from_yaml = raw_data.get(YAML_PROFILES_KEY, [])
            if not isinstance(profiles_list_from_yaml, list):
                print(f"Warning: '{YAML_PROFILES_KEY}' key in MCP profiles YAML at {MCP_PROFILES_YAML_PATH} is not a list. Profiles section will be ignored.")
                profiles_list_from_yaml = []
        elif isinstance(raw_data, list):  # Old format compatibility
            profiles_list_from_yaml = raw_data
            # persisted_active_profile_name remains None
        elif raw_data is None:  # File is empty
            # profiles_list_from_yaml remains [], persisted_active_profile_name remains None
            pass
        else:  # File content is not a list or dict as expected
            print(f"Warning: MCP profiles YAML content at {MCP_PROFILES_YAML_PATH} is not a list or dictionary. File will be treated as empty.")
            # profiles_list_from_yaml remains [], persisted_active_profile_name remains None
            
        # Process the extracted list of profiles
        for profile_dict in profiles_list_from_yaml:
            if not isinstance(profile_dict, dict):
                    print(f"Warning: Skipping non-dictionary profile entry: {profile_dict}")
                    continue
                try:
                    # Ensure servers is a list of dicts with name and no_confirm.
                    loaded_servers_data = profile_dict.get(PROFILE_SERVERS_KEY)
                    parsed_servers_list = []
                    profile_name_for_error = profile_dict.get(PROFILE_NAME_KEY, 'UnknownProfile')

                    if isinstance(loaded_servers_data, list):
                        for server_entry in loaded_servers_data:
                            if isinstance(server_entry, str):
                                parsed_servers_list.append({SERVER_NAME_KEY: server_entry, SERVER_NO_CONFIRM_KEY: False})
                            elif isinstance(server_entry, dict):
                                s_name = server_entry.get(SERVER_NAME_KEY)
                                if not isinstance(s_name, str) or not s_name:
                                    print(f"Warning: Server entry in profile '{profile_name_for_error}' is missing a valid '{SERVER_NAME_KEY}': {server_entry}. Skipping this server.")
                                    continue
                                s_no_confirm = server_entry.get(SERVER_NO_CONFIRM_KEY, False)
                                if not isinstance(s_no_confirm, bool):
                                    print(f"Warning: '{SERVER_NO_CONFIRM_KEY}' for server '{s_name}' in profile '{profile_name_for_error}' is not a boolean (type: {type(s_no_confirm)}). Defaulting to False.")
                                    s_no_confirm = False
                                parsed_servers_list.append({SERVER_NAME_KEY: s_name, SERVER_NO_CONFIRM_KEY: s_no_confirm})
                            else:
                                print(f"Warning: Invalid server entry type in profile '{profile_name_for_error}': {server_entry} (type: {type(server_entry)}). Skipping this server.")
                                continue
                    elif loaded_servers_data is not None: # It was present but not a list
                        print(
                            f"Warning: '{PROFILE_SERVERS_KEY}' for profile '{profile_name_for_error}' is not a list"
                            f" (type: {type(loaded_servers_data)}). Initializing as empty list."
                        )
                        # parsed_servers_list remains empty

                    profile_dict[PROFILE_SERVERS_KEY] = parsed_servers_list # Store the parsed list
                    if OLD_PROFILE_SERVER_NAMES_KEY in profile_dict: # Remove old key if present for clean object creation
                        del profile_dict[OLD_PROFILE_SERVER_NAMES_KEY]
                    
                    # Ensure 'name' key is present for MCPProfile dataclass
                    if PROFILE_NAME_KEY not in profile_dict:
                        print(f"Warning: Profile entry is missing '{PROFILE_NAME_KEY}': {profile_dict}. Skipping this profile.")
                        continue


                    # Rename 'servers' to 'servers' for MCPProfile dataclass if needed (already done by key const)
                    # profile_dict_for_dataclass = {
                    #     'name': profile_dict[PROFILE_NAME_KEY],
                    #     'servers': profile_dict[PROFILE_SERVERS_KEY]
                    # }


                    profile = MCPProfile(**profile_dict) 
                    profiles[profile.name] = profile
                except TypeError as e: # Catches issues like missing 'name' or other fields for MCPProfile
                    profile_name_display = profile_dict.get(PROFILE_NAME_KEY, str(profile_dict))
                    print(f"Warning: Skipping malformed profile entry for '{profile_name_display}': {e}")
        return profiles, persisted_active_profile_name
    except FileNotFoundError:
        return {}, None
    except yaml.YAMLError as e:
        # Potentially log this, e.g., using io if available
        print(f"Warning: Error parsing MCP profiles YAML: {e}")
        return {}, None

def save_mcp_profiles(profiles: Dict[str, MCPProfile], active_profile_name: Optional[str]):
    try:
        profiles_list = [asdict(profile) for profile in profiles.values()]
        data_to_save = {YAML_PROFILES_KEY: profiles_list}
        if active_profile_name is not None:
            data_to_save[YAML_ACTIVE_PROFILE_KEY] = active_profile_name
        
        with open(MCP_PROFILES_YAML_PATH, 'w') as f:
            yaml.dump(data_to_save, f, sort_keys=False)
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
        self.persisted_active_profile_name: Optional[str] = None # For storing loaded active profile name
        self.active_mcp_client_pool: Optional[MCPClientPool] = None

    def load_or_initialize_profiles(self):
        self.profiles, self.persisted_active_profile_name = load_mcp_profiles()

        mcpservers_arg_val = getattr(self.settings, 'mcp_servers', None)
        mcpservers_file_arg_val = getattr(self.settings, 'mcp_servers_file', None)

        current_known_server_names = get_known_mcp_server_names(
            mcpservers_arg_val,
            mcpservers_file_arg_val,
            self.io
        )

        needs_save = False
        old_all_profile_data = self.profiles.get("all") # This is an MCPProfile object or None

        # Construct the new 'all' profile's server list, preserving no_confirm where possible
        new_all_profile_servers_list = []
        for s_name in current_known_server_names:
            no_confirm_val = False # Default
            if old_all_profile_data:
                # Check if this server was in the old 'all' profile
                existing_server_config = next(
                    (s_conf for s_conf in old_all_profile_data.servers if s_conf.get(SERVER_NAME_KEY) == s_name),
                    None
                )
                if existing_server_config:
                    no_confirm_val = existing_server_config.get(SERVER_NO_CONFIRM_KEY, False)
            new_all_profile_servers_list.append({SERVER_NAME_KEY: s_name, SERVER_NO_CONFIRM_KEY: no_confirm_val})
        
        # Sort for consistent comparison later
        new_all_profile_servers_list_sorted = sorted(new_all_profile_servers_list, key=lambda x: x[SERVER_NAME_KEY])

        # Create/Update the 'all' profile in the in-memory dictionary
        self.profiles["all"] = MCPProfile(name="all", servers=new_all_profile_servers_list_sorted)

        if old_all_profile_data is None:
            # The 'all' profile was not in the loaded file, so we created it.
            needs_save = True
            if self.io and hasattr(self.io, 'tool_output'):
                self.io.tool_output("Initialized default MCP profile 'all'.")
        else:
            # The 'all' profile was in the loaded file. Check if it changed.
            # Sort old server data for consistent comparison
            old_all_profile_servers_list_sorted = sorted(old_all_profile_data.servers, key=lambda x: x[SERVER_NAME_KEY])
            if old_all_profile_servers_list_sorted != new_all_profile_servers_list_sorted:
                needs_save = True
                if self.io and hasattr(self.io, 'tool_output'):
                    self.io.tool_output("Updated MCP profile 'all' with current server configurations (names or no_confirm values changed).")
        
        if needs_save:
            # self.active_profile_name is None at this point, so this will either
            # not write the active_profile_name key or write it as null if it was newly created.
            save_mcp_profiles(self.profiles, self.active_profile_name)

    def get_profile(self, name: str) -> Optional[MCPProfile]:
        return self.profiles.get(name)

    def list_profile_names(self) -> List[str]:
        return list(self.profiles.keys())

    def list_profiles_details(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [(profile.name, profile.servers) for profile in self.profiles.values()]

    def get_known_server_names(self) -> List[str]:
        """Helper to get current known server names based on settings."""
        mcpservers_arg_val = getattr(self.settings, 'mcp_servers', None)
        mcpservers_file_arg_val = getattr(self.settings, 'mcp_servers_file', None)
        return get_known_mcp_server_names(
            mcpservers_arg_val,
            mcpservers_file_arg_val,
            self.io
        )

    def create_new_profile(self, profile_name: str, selected_servers_config: List[Dict[str, Any]]):
        if profile_name in self.profiles:
            self.io.tool_error(f"Profile '{profile_name}' already exists. Cannot create.")
            return

        # selected_servers_config is already in the format: [{'name': str, 'no_confirm': bool}, ...]
        # So, we can use it directly.
        
        new_profile = MCPProfile(name=profile_name, servers=selected_servers_config)
        self.profiles[profile_name] = new_profile
        save_mcp_profiles(self.profiles, self.active_profile_name)
        self.io.tool_output(f"MCP Profile '{profile_name}' created and saved.")

    def delete_profile(self, profile_name: str):
        if profile_name == "all":
            self.io.tool_error("The 'all' profile cannot be deleted.")
            return

        if profile_name not in self.profiles:
            self.io.tool_error(f"Profile '{profile_name}' not found.")
            return

        if self.active_profile_name == profile_name:
            self.disable_profile() # This will also save the state with active_profile_name as None

        del self.profiles[profile_name]
        # Save the deletion of the profile, preserving the current active_profile_name
        # (which would be None if the deleted profile was the active one).
        save_mcp_profiles(self.profiles, self.active_profile_name)
        self.io.tool_output(f"MCP profile '{profile_name}' deleted.")

    def is_server_no_confirm(self, server_name_to_check: str) -> bool:
        """
        Checks if a specific server in the currently active profile is set to 'no_confirm'.
        Defaults to False (meaning, confirmation is required) if the profile is not active,
        the server is not found in the profile, or 'no_confirm' is not set/False.
        """
        if not self.active_profile_name:
            return False

        active_profile = self.get_profile(self.active_profile_name)
        if not active_profile:
            return False

        for server_config in active_profile.servers:
            if server_config.get(SERVER_NAME_KEY) == server_name_to_check:
                return server_config.get(SERVER_NO_CONFIRM_KEY, False)
        
        return False # Server not found in profile or no_confirm not set

    def get_tool_prompt_if_active(self) -> Optional[str]:
        """
        Returns the standard tool prompt string if an MCP profile is active,
        otherwise None.
        """
        if self.active_profile_name:
            return CoderPrompts.tool_prompt
        return None

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

        server_configs_in_profile = profile_to_enable.servers
        if not server_configs_in_profile:
            self.io.tool_warning(f"MCP profile '{profile_name}' has no servers configured. Profile enabled but no connections made.")
            self.active_profile_name = profile_name
            self.active_mcp_client_pool = None # Ensure it's cleared
            return

        # Retrieve server configurations using the centralized load_mcp_servers function
        mcpservers_arg_val = getattr(self.settings, 'mcp_servers', None)
        mcpservers_file_arg_val = getattr(self.settings, 'mcp_servers_file', None)
        verbose_setting = getattr(self.settings, 'verbose', False)

        all_mcp_server_objects = load_mcp_servers(
            mcp_servers=mcpservers_arg_val,
            mcp_servers_file=mcpservers_file_arg_val,
            io=self.io,
            verbose=verbose_setting
        )

        # Filter these McpServer objects by name and get their config dictionaries
        profile_server_names = [s_conf[SERVER_NAME_KEY] for s_conf in server_configs_in_profile]
        matched_server_configs = [
            s_obj.config for s_obj in all_mcp_server_objects
            if s_obj.name in profile_server_names
        ]

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
        save_mcp_profiles(self.profiles, self.active_profile_name)
        self.io.tool_output(f"MCP profile '{profile_name}' enabled with {len(matched_server_configs)} server(s).")

    def disable_profile(self):
        if self.active_mcp_client_pool:
            # Assuming MCPClientPool has a shutdown method.
            # If not, this will need to be added to MCPClientPool.
            if hasattr(self.active_mcp_client_pool, 'shutdown') and callable(getattr(self.active_mcp_client_pool, 'shutdown')):
                self.active_mcp_client_pool.shutdown()
            self.active_mcp_client_pool = None

        was_active = self.active_profile_name is not None
        if was_active:
            self.io.tool_output(f"MCP profile '{self.active_profile_name}' disabled.")
            self.active_profile_name = None
            save_mcp_profiles(self.profiles, self.active_profile_name)
        else:
            self.io.tool_output("No MCP profile was active.")
