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
                try:
                    profile = MCPProfile(**profile_dict)
                    profiles[profile.name] = profile
                except TypeError as e:
                    # Handle cases where a dict in YAML doesn't match MCPProfile fields
                    # Potentially log this, for now, we'll skip malformed entries
                    print(f"Warning: Skipping malformed profile entry: {profile_dict} due to {e}")
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

def get_known_mcp_server_names(settings_mcpservers: Optional[List[Dict]]) -> List[str]:
    if not settings_mcpservers:
        return []
    
    server_names = []
    for server_config in settings_mcpservers:
        if isinstance(server_config, dict) and 'name' in server_config:
            server_names.append(server_config['name'])
    return server_names

class MCPProfileManager:
    def __init__(self, io, settings):
        self.io = io
        self.settings = settings
        self.profiles: Dict[str, MCPProfile] = {}
        self.active_profile_name: Optional[str] = None
        self.active_mcp_client_pool: Optional[Any] = None # Replace Any with MCPClientPool later

    def load_or_initialize_profiles(self):
        self.profiles = load_mcp_profiles()
        
        if "all" not in self.profiles:
            known_server_names = get_known_mcp_server_names(
                self.settings.mcpservers if hasattr(self.settings, 'mcpservers') else None
            )
            all_profile = MCPProfile(name="all", server_names=known_server_names)
            self.profiles["all"] = all_profile
            save_mcp_profiles(self.profiles)
            if self.io:
                self.io.tool_output("Initialized default MCP profile 'all'.")

    def get_profile(self, name: str) -> Optional[MCPProfile]:
        return self.profiles.get(name)

    def list_profile_names(self) -> List[str]:
        return list(self.profiles.keys())

    def list_profiles_details(self) -> List[Tuple[str, List[str]]]:
        return [(profile.name, profile.server_names) for profile in self.profiles.values()]
