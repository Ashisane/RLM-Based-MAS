"""
Gemini API client using google-genai package.
"""
import time
import json
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, ROOT_MODEL, SUB_MODEL, PRO_MODEL, cost_tracker, MAX_RETRIES

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY)


def gemini_query(prompt: str, model: str = ROOT_MODEL, temperature: float = 0.2,
                 max_tokens: int = 8192, description: str = "") -> str:
    """
    Query Gemini API with automatic retry and cost tracking.
    
    Args:
        prompt: The prompt to send
        model: Model name (default: gemini-2.5-flash)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        description: Description for cost tracking
    
    Returns:
        Response text
    
    Raises:
        RuntimeError: If all retries fail or gibberish output detected
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            # Track cost
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0
            else:
                # Estimate if metadata not available
                input_tokens = len(prompt) // 4
                output_tokens = len(response.text) // 4 if response.text else 0
            
            cost_tracker.add_call(model, input_tokens, output_tokens, description)
            
            if response.text:
                return response.text.strip()
            else:
                print(f"WARNING: Empty response on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"ERROR on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise RuntimeError(f"Gemini query failed after {MAX_RETRIES} attempts: {e}")
    
    raise RuntimeError("Gemini query returned empty response after all retries")


# Async semaphores for rate limiting
import asyncio
_flash_semaphore = None
_pro_semaphore = None

def get_semaphores():
    """Get or create async semaphores for rate limiting."""
    global _flash_semaphore, _pro_semaphore
    from config import MAX_CONCURRENT_FLASH, MAX_CONCURRENT_PRO
    if _flash_semaphore is None:
        _flash_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FLASH)
    if _pro_semaphore is None:
        _pro_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PRO)
    return _flash_semaphore, _pro_semaphore


async def async_gemini_query(prompt: str, model: str = ROOT_MODEL, temperature: float = 0.2,
                              max_tokens: int = 8192, description: str = "") -> str:
    """
    Async version of gemini_query with rate limiting via semaphore.
    """
    flash_sem, pro_sem = get_semaphores()
    semaphore = pro_sem if "pro" in model.lower() else flash_sem
    
    async with semaphore:
        # Run sync query in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: gemini_query(prompt, model, temperature, max_tokens, description)
        )


def gemini_json_query(prompt: str, model: str = ROOT_MODEL, 
                       description: str = "") -> dict | list:
    """
    Query Gemini and parse JSON response.
    
    Raises:
        RuntimeError: If JSON parsing fails (gibberish output)
    """
    response = gemini_query(prompt, model=model, description=description)
    
    # Try to extract JSON from response
    try:
        # Direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in response
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find array or object
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        match = re.search(pattern, response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    
    raise RuntimeError(f"GIBBERISH OUTPUT - Failed to parse JSON from response: {response[:500]}...")


def llm_query(prompt: str, chunk: str) -> str:
    """
    Sub-LLM query function for RLM REPL environment.
    Used for semantic extraction within code execution.
    """
    full_prompt = f"{prompt}\n\nText to analyze:\n{chunk}"
    return gemini_query(full_prompt, model=SUB_MODEL, description="RLM sub-call")
