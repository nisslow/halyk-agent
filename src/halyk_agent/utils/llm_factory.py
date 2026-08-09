import os
import logging
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def get_llm(temperature: float = None) -> ChatOpenAI:
        """
        Creates and returns a ChatOpenAI instance based on environment configuration.
        Defaults to OpenRouter for speed and accuracy.
        """
        provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
        
        if temperature is None:
            # Default to 0.0 for OpenRouter, but 0.6 for local models (Qwen requires 0.6 for thinking)
            env_temp = os.getenv("LLM_TEMPERATURE")
            if env_temp is not None:
                temperature = float(env_temp)
            else:
                temperature = 0.0 if provider == "openrouter" else 0.6
        
        if provider == "openrouter":
            model = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash-0731")
            api_keys_str = os.getenv("OPENROUTER_API_KEY", "")
            
            # Split by comma and strip whitespace to support multiple keys
            import random
            api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
            api_key = random.choice(api_keys) if api_keys else ""
            
            base_url = "https://openrouter.ai/api/v1"
            logger.debug(f"Using OpenRouter: {model} with a randomly selected API key.")
        else:
            # Default to local LM Studio / Ollama (OpenAI compatible)
            model = os.getenv("LLM_MODEL", "local-model")
            api_key = os.getenv("LOCAL_API_KEY", "lm-studio")
            base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
            logger.debug(f"Using Local LLM at {base_url}")

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=1,
            request_timeout=120.0
        )
        return llm
