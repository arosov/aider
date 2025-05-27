from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml

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
    server_names_set = set()
    file_encoding = 'utf-8'  # Default encoding

    if io and hasattr(io, 'encoding') and io.encoding:
        file_encoding = io.encoding

    # Helper to process data loaded from YAML/JSON
    def _extract_server_names_from_data(data: Any, source_description: str):
        
        if not isinstance(data, dict):
            if data is not None and io and hasattr(io, 'tool_warning'):
                io.tool_warning(
                    f"MCP servers config from {source_description} is not a dictionary (type:"
                    f" {type(data)}), skipping. Expected a top-level dictionary."
                )
            return

        mcp_servers_object = data.get("mcpServers")
        
        if not isinstance(mcp_servers_object, dict):
            if mcp_servers_object is not None and io and hasattr(io, 'tool_warning'):
                io.tool_warning(
                    f"MCP servers config from {source_description}: 'mcpServers' key does not point to a dictionary (type:"
                    f" {type(mcp_servers_object)}), skipping."
                )
            elif "mcpServers" not in data and io and hasattr(io, 'tool_warning'):
                 io.tool_warning(
                    f"MCP servers config from {source_description}: Missing 'mcpServers' top-level key, skipping."
                )
            return
            
        extracted_names = list(mcp_servers_object.keys())
        for server_name in extracted_names:
            server_names_set.add(server_name)
        
    # Helper to process a file path
    def _process_file_path(file_path_str: str, source_description: str):
        try:
            path_obj = Path(file_path_str)
            # Check if path_obj is None or empty string before calling expanduser
            if not path_obj or not str(path_obj).strip():
                if io and hasattr(io, 'tool_warning'):
                    io.tool_warning(f"Empty file path provided for {source_description}, skipping.")
                return

            file_path = path_obj.expanduser()
            if file_path.exists():
                with open(file_path, 'r', encoding=file_encoding) as f:
                    data = yaml.safe_load(f)
                _extract_server_names_from_data(data, f"file {file_path_str}")
            # Silently ignore if file not found
        except yaml.YAMLError as e:
            if io and hasattr(io, 'tool_warning'):
                io.tool_warning(f"Error parsing MCP servers {source_description} file {file_path_str}: {e}")
        except IOError as e:
            if io and hasattr(io, 'tool_warning'):
                io.tool_warning(f"Error reading MCP servers {source_description} file {file_path_str}: {e}")
        except Exception as e:
            if io and hasattr(io, 'tool_warning'):
                io.tool_warning(f"Unexpected error processing MCP {source_description} file {file_path_str}: {e}")

    # 1. Process mcpservers_arg (args.mcpservers)
    if mcpservers_arg:
        if isinstance(mcpservers_arg, dict): # Already parsed as a dict (e.g. from aider.conf.yml)
            _extract_server_names_from_data(mcpservers_arg, "direct config (dict)")
        elif isinstance(mcpservers_arg, str):
            parsed_as_yaml_dict = False
            try:
                data = yaml.safe_load(mcpservers_arg)
                if isinstance(data, dict): # Parsed as a dict
                    _extract_server_names_from_data(data, "direct config (YAML string)")
                    parsed_as_yaml_dict = True
                elif data is not None and io and hasattr(io, 'tool_warning'): # Parsed as something else
                    io.tool_warning(
                        f"MCP servers config from direct YAML string was not a dictionary (type: {type(data)}), skipping."
                    )
            except yaml.YAMLError:
                # Not valid YAML, so it's likely a path. Fall through.
                pass
            
            if not parsed_as_yaml_dict:
                # If it wasn't parsed as a YAML dict treat the original string as a path.
                _process_file_path(mcpservers_arg, "direct config (path)")
        elif isinstance(mcpservers_arg, list) and io and hasattr(io, 'tool_warning'):
             io.tool_warning(
                f"MCP servers config from direct config was a list, skipping. Expected a dictionary."
             )
        # Else: other types are ignored or handled by _extract_server_names_from_data if it's a dict.

    # 2. Process mcpservers_file_arg (args.mcpservers_file)
    if mcpservers_file_arg:
        _process_file_path(mcpservers_file_arg, "file arg")

    final_names = list(server_names_set)

    return final_names

class MCPProfileManager:
    def __init__(self, io, settings):
        self.io = io
        self.settings = settings
        self.profiles: Dict[str, MCPProfile] = {}
        self.active_profile_name: Optional[str] = None
        self.active_mcp_client_pool: Optional[Any] = None # Replace Any with MCPClientPool later

    def load_or_initialize_profiles(self):
        self.profiles = load_mcp_profiles()

        mcpservers_arg_val = getattr(self.settings, 'mcp_servers', None)
        mcpservers_file_arg_val = getattr(self.settings, 'mcp_servers_file', None)

        current_known_server_names = sorted(list(set(get_known_mcp_server_names(
            mcpservers_arg_val,
            mcpservers_file_arg_val,
            self.io
        ))))

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
