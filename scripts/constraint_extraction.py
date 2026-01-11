"""
Phase 1: Dense RLM Constraint Extraction with Logging and Parallel Processing

Features:
1. Persistent file-based logging for debugging
2. Parallel API calls for chunk processing
3. Robust error handling with full response logging
"""
import re
import json
import asyncio
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Any

from config import CACHE_DIR, cost_tracker
from gemini_client import gemini_query, ROOT_MODEL

# Logging directory
LOG_DIR = CACHE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Chunking configuration
CHUNK_SIZE = 80000
CHUNK_OVERLAP = 20000


class ExtractionLogger:
    """Persistent file-based logger for extraction debugging."""
    
    def __init__(self, book_name: str):
        safe_name = book_name.replace(" ", "_").replace("'", "")[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"extraction_{safe_name}_{timestamp}.log"
        self.book_name = book_name
        self._write(f"=== Extraction Log for {book_name} ===")
        self._write(f"Started at: {datetime.now().isoformat()}")
    
    def _write(self, msg: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    
    def info(self, msg: str):
        self._write(f"[INFO] {msg}")
    
    def warn(self, msg: str):
        self._write(f"[WARN] {msg}")
    
    def error(self, msg: str):
        self._write(f"[ERROR] {msg}")
    
    def chunk_request(self, chunk_idx: int, prompt: str):
        self._write(f"\n{'='*60}")
        self._write(f"[CHUNK {chunk_idx}] Request sent")
        self._write(f"Prompt length: {len(prompt)} chars")
        # Save full prompt to separate file for debugging
        prompt_file = LOG_DIR / f"chunk_{chunk_idx}_prompt.txt"
        prompt_file.write_text(prompt[:5000] + "\n...[truncated]...", encoding="utf-8")
    
    def chunk_response(self, chunk_idx: int, response: str, parse_success: bool):
        self._write(f"[CHUNK {chunk_idx}] Response received: {len(response)} chars")
        self._write(f"[CHUNK {chunk_idx}] Parse success: {parse_success}")
        # Save full response for debugging
        response_file = LOG_DIR / f"chunk_{chunk_idx}_response.txt"
        response_file.write_text(response, encoding="utf-8")
        if not parse_success:
            self._write(f"[CHUNK {chunk_idx}] Response preview: {response[:500]}...")
    
    def result(self, chunk_idx: int, events: int, relationships: int):
        self._write(f"[CHUNK {chunk_idx}] Extracted: {events} events, {relationships} relationships")
    
    def summary(self, constraints: dict):
        self._write(f"\n{'='*60}")
        self._write(f"=== EXTRACTION SUMMARY ===")
        self._write(f"Total events: {len(constraints.get('events', []))}")
        self._write(f"Total relationships: {len(constraints.get('relationships', []))}")
        self._write(f"Total capabilities: {len(constraints.get('capabilities', []))}")
        self._write(f"Cost so far: INR {cost_tracker.get_total_cost_inr():.2f}")
        self._write(f"Log file: {self.log_file}")


def create_overlapping_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for dense extraction."""
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 1000:
            chunks.append(chunk)
    
    return chunks


def build_extraction_prompt(chunk: str, chunk_idx: int, total_chunks: int) -> str:
    """Build the extraction prompt for a chunk."""
    return f"""You are extracting narrative facts from a novel passage. Return ONLY valid JSON.

Extract these elements from the passage:
1. EVENTS: What happened (actions, occurrences)
2. TIMELINE: Ages, dates, chronological info
3. RELATIONSHIPS: Character connections
4. CAPABILITIES: Skills characters demonstrate

RESPONSE FORMAT - Return ONLY this JSON structure, nothing else:
{{
  "events": [
    {{"description": "brief description of what happened", "characters": ["char1"], "when": "timing if mentioned"}}
  ],
  "timeline_facts": [
    {{"character": "name", "age_or_date": "value", "context": "brief context"}}
  ],
  "relationships": [
    {{"char_a": "name1", "char_b": "name2", "type": "family/friend/enemy/romantic/professional", "description": "brief"}}
  ],
  "capabilities": [
    {{"character": "name", "skill": "what they can do"}}
  ]
}}

CRITICAL RULES:
- Return ONLY the JSON object, no markdown, no explanation
- Extract 5-15 items total if they exist in the text
- Use empty arrays [] if no items found for a category
- Keep descriptions under 100 characters

NOVEL PASSAGE ({chunk_idx + 1}/{total_chunks}):
{chunk[:70000]}"""


def parse_extraction_response(response: str) -> tuple[dict, bool]:
    """
    Parse LLM response into structured data.
    Returns (parsed_data, success_flag)
    """
    default = {"events": [], "timeline_facts": [], "relationships": [], "capabilities": []}
    
    if not response or len(response) < 10:
        return default, False
    
    # Clean response - remove markdown if present
    cleaned = response.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Try direct JSON parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result, True
    except json.JSONDecodeError as e:
        pass
    
    # Try to find JSON object in response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if isinstance(result, dict):
                return result, True
        except:
            pass
    
    # Try to find just the arrays if object parsing fails
    try:
        events_match = re.search(r'"events"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        rels_match = re.search(r'"relationships"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        
        result = {"events": [], "timeline_facts": [], "relationships": [], "capabilities": []}
        
        if events_match:
            try:
                events_str = "[" + events_match.group(1) + "]"
                result["events"] = json.loads(events_str)
            except:
                pass
        
        if rels_match:
            try:
                rels_str = "[" + rels_match.group(1) + "]"
                result["relationships"] = json.loads(rels_str)
            except:
                pass
        
        if result["events"] or result["relationships"]:
            return result, True
    except:
        pass
    
    return default, False


def extract_single_chunk(args: tuple) -> dict:
    """Extract from a single chunk (for parallel processing)."""
    chunk, chunk_idx, total_chunks, logger = args
    
    prompt = build_extraction_prompt(chunk, chunk_idx, total_chunks)
    logger.chunk_request(chunk_idx, prompt)
    
    try:
        response = gemini_query(prompt, description=f"Chunk {chunk_idx + 1}/{total_chunks}")
        result, success = parse_extraction_response(response)
        logger.chunk_response(chunk_idx, response, success)
        
        events_count = len(result.get("events", []))
        rels_count = len(result.get("relationships", []))
        logger.result(chunk_idx, events_count, rels_count)
        
        return result
        
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
        return {"events": [], "timeline_facts": [], "relationships": [], "capabilities": []}


def extract_chunks_parallel(chunks: list[str], logger: ExtractionLogger, max_workers: int = 4) -> list[dict]:
    """Process chunks in parallel using ThreadPoolExecutor."""
    logger.info(f"Starting parallel extraction with {max_workers} workers for {len(chunks)} chunks")
    
    # Prepare arguments for each chunk
    args_list = [(chunk, idx, len(chunks), logger) for idx, chunk in enumerate(chunks)]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single_chunk, args): args[1] for args in args_list}
        
        for future in concurrent.futures.as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Chunk {chunk_idx} execution failed: {e}")
                results.append((chunk_idx, {"events": [], "timeline_facts": [], "relationships": [], "capabilities": []}))
    
    # Sort by chunk index to maintain order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def merge_extraction_results(results: list[dict]) -> dict:
    """Merge results from multiple chunks, deduplicating where possible."""
    merged = {
        "events": [],
        "timeline_facts": [],
        "relationships": [],
        "capabilities": [],
        "characters": set()
    }
    
    seen_events = set()
    seen_relationships = set()
    
    for result in results:
        # Events
        for event in result.get("events", []):
            if isinstance(event, dict):
                desc = str(event.get("description", ""))[:80]
                if desc and desc not in seen_events:
                    seen_events.add(desc)
                    merged["events"].append(event)
        
        # Timeline facts
        for fact in result.get("timeline_facts", []):
            if isinstance(fact, dict):
                merged["timeline_facts"].append(fact)
        
        # Relationships
        for rel in result.get("relationships", []):
            if isinstance(rel, dict):
                key = tuple(sorted([str(rel.get("char_a", "")), str(rel.get("char_b", ""))]))
                if key not in seen_relationships and key[0]:
                    seen_relationships.add(key)
                    merged["relationships"].append(rel)
        
        # Capabilities
        for cap in result.get("capabilities", []):
            if isinstance(cap, dict):
                merged["capabilities"].append(cap)
    
    merged["characters"] = list(merged["characters"])
    return merged


def extract_with_regex(novel_text: str) -> dict:
    """Fast regex extraction for structured data."""
    from collections import Counter
    
    # Age patterns
    age_patterns = [
        r'(\w+)\s+(?:was|is)\s+(\d+)\s+years?\s+old',
        r'at\s+(?:the\s+)?age\s+of\s+(\d+)',
        r'(\w+),?\s+aged?\s+(\d+)',
    ]
    
    ages = []
    for pattern in age_patterns:
        for match in re.finditer(pattern, novel_text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                if groups[0].isdigit():
                    ages.append({"character": groups[1], "age": groups[0]})
                else:
                    ages.append({"character": groups[0], "age": groups[1]})
    
    # Date patterns
    dates = []
    for match in re.finditer(r'(?:in|during)\s+(\d{4})', novel_text, re.IGNORECASE):
        year = match.group(1)
        start = max(0, match.start() - 50)
        end = min(len(novel_text), match.end() + 50)
        context = novel_text[start:end]
        dates.append({"date": year, "context": context})
    
    # Character names
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    candidates = re.findall(name_pattern, novel_text[:200000])
    counts = Counter(candidates)
    
    stop_words = {'The', 'Chapter', 'Part', 'Book', 'And', 'But', 'This', 'That', 
                  'When', 'Where', 'They', 'There', 'Have', 'Had', 'Was', 'Were'}
    
    names = [n for n, c in counts.most_common(100) if 3 < c < 1000 and n not in stop_words][:50]
    
    return {"ages": ages[:50], "dates": dates[:30], "character_names": names}


def extract_constraints_dense(novel_text: str, book_name: str = "Unknown", 
                               max_chunks: int = 8, parallel_workers: int = 4) -> dict:
    """
    Extract comprehensive constraints from a novel using dense parallel chunking.
    """
    logger = ExtractionLogger(book_name)
    logger.info(f"Novel length: {len(novel_text):,} characters")
    
    # Step 1: Fast regex extraction
    logger.info("Running regex extraction...")
    regex_data = extract_with_regex(novel_text)
    logger.info(f"Regex found: {len(regex_data['ages'])} ages, {len(regex_data['dates'])} dates, {len(regex_data['character_names'])} names")
    
    # Step 2: Create chunks
    chunks = create_overlapping_chunks(novel_text)
    logger.info(f"Created {len(chunks)} overlapping chunks")
    
    # Limit chunks for cost control
    if len(chunks) > max_chunks:
        indices = [int(i * len(chunks) / max_chunks) for i in range(max_chunks)]
        chunks = [chunks[i] for i in indices]
        logger.info(f"Limited to {len(chunks)} evenly-spaced chunks")
    
    # Step 3: Parallel extraction
    chunk_results = extract_chunks_parallel(chunks, logger, parallel_workers)
    
    # Step 4: Merge results
    logger.info("Merging extraction results...")
    merged = merge_extraction_results(chunk_results)
    
    # Build final constraints
    constraints = {
        "temporal": {
            "ages": regex_data["ages"],
            "dates": regex_data["dates"],
            "timeline_facts": merged["timeline_facts"]
        },
        "events": merged["events"],
        "relationships": merged["relationships"],
        "capabilities": merged["capabilities"],
        "characters": list(set(regex_data["character_names"]) | set(merged.get("characters", []))),
        "metadata": {
            "novel_length": len(novel_text),
            "chunks_processed": len(chunks),
            "total_events": len(merged["events"]),
            "total_relationships": len(merged["relationships"]),
            "log_file": str(logger.log_file),
            "cost_so_far": cost_tracker.get_total_cost_inr()
        }
    }
    
    logger.summary(constraints)
    return constraints


def cache_constraints(book_name: str, constraints: dict):
    """Cache extracted constraints."""
    cache_file = CACHE_DIR / f"constraints_{book_name.replace(' ', '_')}.json"
    cache_file.write_text(json.dumps(constraints, indent=2, default=str), encoding='utf-8')


def load_cached_constraints(book_name: str) -> dict | None:
    """Load cached constraints if available."""
    cache_file = CACHE_DIR / f"constraints_{book_name.replace(' ', '_')}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding='utf-8'))
    return None


# Legacy compatibility
def extract_constraints_rlm(novel_text: str, character_name: str = None) -> dict:
    return extract_constraints_dense(novel_text, "Unknown")
