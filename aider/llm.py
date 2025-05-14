import importlib
import os
import warnings
import time # <--- ADD THIS IMPORT

from aider.dump import dump  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

try:
    import google.generativeai as genai
    import google.generativeai.types as glm
    from google.generativeai.types import GenerationConfig
    from google.api_core import exceptions as google_exceptions
    from google.generativeai.types.generation_types import Candidate
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None # Keep for basic genai reference if needed elsewhere
    # Mock classes for type hinting and preventing NameErrors if genai is not installed
    class Mockglm_LLM: Part = type('Part', (), {}); Content = type('Content', (), {}) # type: ignore
    glm = Mockglm_LLM() # type: ignore
    GenerationConfig = type('GenerationConfig', (), {}) # type: ignore
    google_exceptions = type('google_exceptions', (), {'GoogleAPIError': Exception, 'AlreadyExists': Exception, 'InvalidArgument': Exception}) # type: ignore
    # Ensure all FinishReason members used are defined in the mock
    _MockFinishReason = type('FinishReason', (), {
        'MAX_TOKENS':1, 'STOP':2, 'SAFETY':3, 'RECITATION':4, 'OTHER': 5, 'UNSPECIFIED': 0
    })
    Candidate = type('Candidate', (), {'FinishReason': _MockFinishReason }) # type: ignore

AIDER_SITE_URL = "https://aider.chat"
AIDER_APP_NAME = "Aider"

os.environ["OR_SITE_URL"] = AIDER_SITE_URL
os.environ["OR_APP_NAME"] = AIDER_APP_NAME
os.environ["LITELLM_MODE"] = "PRODUCTION"

# `import litellm` takes 1.5 seconds, defer it!

VERBOSE = False

_genai_configured = False

def _configure_genai(api_key_to_use): # Changed parameter name for clarity
    global _genai_configured
    if not GENAI_AVAILABLE:
        return
    # If already configured globally (e.g., by a previous call with a key),
    # assume it's correctly set up. genai.configure is a global setting.
    if _genai_configured:
        return

    if api_key_to_use: # Only proceed if an API key is actually provided
        try:
            genai.configure(api_key=api_key_to_use)
            _genai_configured = True # Set global flag on success
        except Exception as e:
            print(f"Warning (llm.py): Failed to configure google-generativeai with provided API key: {e}")
            _genai_configured = False # Ensure it's marked as not configured on failure
    # else: No API key provided to this function, so _genai_configured remains as it was.


class LazyLiteLLM:
    _lazy_module = None

    def __init__(self, verbose=False):
        self.verbose = verbose
        # NEW: Track if this instance has successfully ensured genai is configured
        self.genai_configured_internally = False
        self.gemini_api_key_for_instance = None

        # Initial attempt to configure if a key was somehow globally set before this instance
        # and _genai_configured is already true.
        if GENAI_AVAILABLE and _genai_configured:
             self.genai_configured_internally = True

    # NEW METHOD
    def set_gemini_api_key(self, api_key):
        self.gemini_api_key_for_instance = api_key
        if GENAI_AVAILABLE and api_key:
            _configure_genai(api_key) # Attempt to configure (sets global _genai_configured)
            if _genai_configured: # Check global status after attempt
                self.genai_configured_internally = True
            else:
                self.genai_configured_internally = False
                # This warning indicates a provided key didn't work
                print(
                    "Warning (llm.py): Provided Gemini API key for LazyLiteLLM did not result in successful"
                    " google-generativeai configuration. Gemini calls may fail."
                )
        elif not api_key:
             self.genai_configured_internally = False # No key provided to this instance
        # If GENAI_AVAILABLE is false, genai_configured_internally remains false.

    def _is_gemini_model(self, model_name):
        return model_name and (model_name.startswith("gemini-") or model_name.startswith("models/gemini-"))

    def _completion_gemini(self, model, messages, stream=False, **kwargs):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-generativeai library is not installed. Cannot use Gemini models.")

        if not self.genai_configured_internally:
            error_message = (
                "Gemini API key not configured for use with google-generativeai library. "
                "Please ensure the Gemini API key is correctly set in Aider's configuration "
                "and passed to this LLM handler."
            )
            if self.gemini_api_key_for_instance and not _genai_configured:
                error_message = (
                    "google-generativeai failed to initialize using the provided Gemini API key. "
                    "The key might be invalid or there could be another configuration issue. "
                    "Cannot use Gemini model."
                )
            raise ValueError(error_message)

        original_model_name = model # Keep original name for response
        if model.startswith("models/"):
            model = model[len("models/"):]

        gemini_formatted_messages = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if system_instruction is None: system_instruction = ""
                system_instruction += str(content) + "\n"
                continue

            gemini_role = "user" if role == "user" else "model"
            parts = []
            if isinstance(content, str):
                parts.append(glm.Part(text=content))
            elif isinstance(content, list): # OpenAI multimodal format
                for item in content:
                    if item.get("type") == "text":
                        parts.append(glm.Part(text=item["text"]))
                    elif item.get("type") == "image_url":
                        # Image handling would go here if supported
                        print(f"Warning (llm.py): Image part encountered for Gemini model {model}. Image content is currently NOT sent to Gemini.")
            if parts:
                gemini_formatted_messages.append(glm.Content(parts=parts, role=gemini_role))
        
        if system_instruction:
            system_instruction = system_instruction.strip()

        gen_config_params = {}
        if "temperature" in kwargs: gen_config_params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs: gen_config_params["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs: gen_config_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs: gen_config_params["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            stop_val = kwargs["stop"]
            gen_config_params["stop_sequences"] = [stop_val] if isinstance(stop_val, str) else stop_val

        generation_config_obj = GenerationConfig(**gen_config_params) if gen_config_params else None

        try:
            gemini_model_obj = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction if system_instruction else None,
            )
            response_genai = gemini_model_obj.generate_content(
                contents=gemini_formatted_messages,
                generation_config=generation_config_obj,
                stream=stream,
            )
        except google_exceptions.InvalidArgument as e:
            raise ValueError(f"Gemini API call failed for model {model} due to invalid argument (check messages/roles): {e}")
        except google_exceptions.GoogleAPIError as e:
            raise RuntimeError(f"Gemini API call failed for model {model}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to call Gemini model {model}: {e}")

        if stream:
            def stream_adapter():
                try:
                    for chunk in response_genai:
                        delta_content = ""
                        chunk_finish_reason_str = None
                        if chunk.parts:
                            for part in chunk.parts:
                                if hasattr(part, 'text'): delta_content += part.text
                        elif hasattr(chunk, 'text') and chunk.text is not None:
                            delta_content = chunk.text
                        
                        if chunk.candidates:
                            raw_fr = chunk.candidates[0].finish_reason
                            if raw_fr == Candidate.FinishReason.STOP: chunk_finish_reason_str = "stop"
                            elif raw_fr == Candidate.FinishReason.MAX_TOKENS: chunk_finish_reason_str = "length"
                            elif raw_fr in [Candidate.FinishReason.SAFETY, Candidate.FinishReason.RECITATION]:
                                chunk_finish_reason_str = raw_fr.name.lower()
                                if self.verbose: print(f"Verbose (llm.py): Gemini stream chunk indicates stop due to {chunk_finish_reason_str}")

                        yield {
                            "choices": [{"delta": {"content": delta_content, "role": "assistant"}, "finish_reason": chunk_finish_reason_str, "index": 0}],
                            "model": original_model_name,
                            "id": f"chatcmpl-gemini-stream-{os.urandom(8).hex()}",
                            "object": "chat.completion.chunk", "created": int(time.time()),
                        }
                except google_exceptions.GoogleAPIError as e:
                    print(f"Error (llm.py): During Gemini stream: {e}")
                    yield {"choices": [{"delta": {"content": f"\n\nError during stream: {e}"}, "finish_reason": "error"}]}
                except Exception as e:
                    print(f"Error (llm.py): Generic error during Gemini stream processing: {e}")
                    yield {"choices": [{"delta": {"content": f"\n\nGeneric error during stream: {e}"}, "finish_reason": "error"}]}
            return stream_adapter()
        else: # Non-streaming
            full_content = ""
            final_finish_reason_str = None
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            try:
                if response_genai.candidates:
                    candidate = response_genai.candidates[0]
                    if candidate.content and candidate.content.parts:
                        full_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    raw_fr = candidate.finish_reason
                    if raw_fr == Candidate.FinishReason.STOP: final_finish_reason_str = "stop"
                    elif raw_fr == Candidate.FinishReason.MAX_TOKENS: final_finish_reason_str = "length"
                    elif raw_fr in [Candidate.FinishReason.SAFETY, Candidate.FinishReason.RECITATION]:
                        final_finish_reason_str = raw_fr.name.lower()
                        print(f"Warning (llm.py): Gemini response stopped due to {final_finish_reason_str}")
                    elif raw_fr == Candidate.FinishReason.OTHER: final_finish_reason_str = "other"
                    else: final_finish_reason_str = "unknown"
                elif hasattr(response_genai, 'text') and response_genai.text is not None:
                    full_content = response_genai.text
                    if response_genai.prompt_feedback and response_genai.prompt_feedback.block_reason:
                        final_finish_reason_str = f"blocked: {response_genai.prompt_feedback.block_reason.name.lower()}"
                    else: final_finish_reason_str = "stop"

                if not full_content and response_genai.prompt_feedback and response_genai.prompt_feedback.block_reason:
                    block_reason_name = response_genai.prompt_feedback.block_reason.name
                    final_finish_reason_str = f"blocked: {block_reason_name.lower()}"
                    print(f"Warning (llm.py): Gemini response blocked. Reason: {block_reason_name}")

                if hasattr(response_genai, 'usage_metadata') and response_genai.usage_metadata:
                    prompt_tokens = response_genai.usage_metadata.prompt_token_count
                    completion_tokens = response_genai.usage_metadata.candidates_token_count
                    total_tokens = response_genai.usage_metadata.total_token_count
            except google_exceptions.GoogleAPIError as e:
                raise RuntimeError(f"Gemini API error processing non-streaming response for model {model}: {e}")
            except Exception as e:
                if hasattr(response_genai, 'prompt_feedback') and response_genai.prompt_feedback and response_genai.prompt_feedback.block_reason:
                    block_reason_name = response_genai.prompt_feedback.block_reason.name
                    raise RuntimeError(f"Gemini response blocked for model {model}. Reason: {block_reason_name}. Details: {e}")
                raise RuntimeError(f"Error processing Gemini non-streaming response for model {model}: {e}")

            return {
                "choices": [{"message": {"role": "assistant", "content": full_content}, "finish_reason": final_finish_reason_str, "index": 0}],
                "model": original_model_name,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
                "id": f"chatcmpl-gemini-{os.urandom(8).hex()}", "object": "chat.completion", "created": int(time.time()),
            }


    def completion(self, model, messages, **kwargs):
        if self._is_gemini_model(model) and GENAI_AVAILABLE:
            if self.genai_configured_internally:
                try:
                    return self._completion_gemini(model, messages, **kwargs)
                except NotImplementedError: # Catch if placeholder is hit
                    if self.verbose:
                        print("Warning (llm.py): _completion_gemini is not fully implemented. Falling back to litellm module.")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning (llm.py): Direct Gemini call via _completion_gemini failed: {e}. Falling back to litellm module.")
            elif self.gemini_api_key_for_instance and self.verbose: # Key provided, but configure failed
                 print(
                    "Warning (llm.py): Direct Gemini call skipped as configuration with provided key failed. "
                    "Falling back to litellm module."
                 )
            # Fall through to litellm if direct call not attempted, not implemented, or failed

        # Fallback to litellm module
        self._load_litellm()
        return self._lazy_module.completion(model=model, messages=messages, **kwargs)

    def __getattr__(self, name):
        if name == "_lazy_module": # pragma: no cover
            return super().__getattr__(name) # Use super() for attribute access on self
        if name == "completion":
            return self.completion # Return the bound method

        self._load_litellm()
        return getattr(self._lazy_module, name)

    def _load_litellm(self):
        if self._lazy_module is not None:
            return

        if self.verbose: # Use instance verbose
            print("Loading litellm...")

        self._lazy_module = importlib.import_module("litellm")

        self._lazy_module.suppress_debug_info = True
        self._lazy_module.set_verbose = False # LiteLLM's own verbosity, not ours
        self._lazy_module.drop_params = True
        self._lazy_module._logging._disable_debugging()


litellm = LazyLiteLLM(verbose=VERBOSE)

__all__ = [litellm]
