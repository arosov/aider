from typing import List, Dict, Optional, Any

def filter_tools_for_server(
    original_tool_definitions: List[Dict[str, Any]],
    enabled_tool_names: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """
    Filters a list of tool definitions based on a list of enabled tool names.

    Args:
        original_tool_definitions: The original list of tool definition dictionaries.
        enabled_tool_names: An optional list of tool names that are enabled.
                            If None, all original_tool_definitions are considered enabled.

    Returns:
        A new list containing only the tool definitions that are enabled.
    """
    if enabled_tool_names is None:
        # If enabled_tool_names is None, it means all tools are implicitly enabled for this server in the profile.
        return list(original_tool_definitions)  # Return a copy

    filtered_tools = []
    if original_tool_definitions: # Ensure there are tools to iterate over
        for tool_def in original_tool_definitions:
            try:
                # Ensure tool_def is a dict and has the expected structure
                if isinstance(tool_def, dict) and \
                   tool_def.get("type") == "function" and \
                   isinstance(tool_def.get("function"), dict) and \
                   isinstance(tool_def["function"].get("name"), str) and \
                   tool_def["function"]["name"] in enabled_tool_names:
                    filtered_tools.append(tool_def)
            except (KeyError, TypeError):
                # Handle cases where tool_def might not have the expected structure
                # This might happen if tool definitions are malformed.
                # Optionally, log a warning here if io is available.
                continue
    return filtered_tools
