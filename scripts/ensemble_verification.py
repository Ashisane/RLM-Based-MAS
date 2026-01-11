"""
Phase 2: Ensemble Verification with Supervisor Arbitration (Optimized)

Improvements:
1. Async processing with rate-limited semaphores
2. Pro model for supervisor (better judgment)  
3. Progress logging with ETA
4. Zero-constraint fallback (larger novel context)
5. BDH-inspired fallback agent for sparse constraints
"""
import re
import json
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from config import cost_tracker, CACHE_DIR, REPORT_DIR, SUPERVISOR_MODEL, ROOT_MODEL
from gemini_client import gemini_query, async_gemini_query
from data_loader import Sample

# Import BDH-inspired fallback agent
try:
    from bdh_agent import bdh_fallback_verify
    BDH_FALLBACK_ENABLED = True
except ImportError:
    BDH_FALLBACK_ENABLED = False
    async def bdh_fallback_verify(*args, **kwargs):
        return "consistent", 0.5, "BDH fallback not available"

# Logging directory
LOG_DIR = CACHE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

FALLBACK_CONSTRAINT_THRESHOLD = 1  # Use BDH when constraints <= 1

@dataclass
class AgentResult:
    """Result from a single agent."""
    verdict: str
    confidence: float
    reasoning: str


@dataclass
class EnsembleResult:
    """Full ensemble verification result."""
    sample_id: int
    book_name: str
    character: str
    backstory: str
    actual_label: Optional[str]
    
    # Agent results
    conservative: AgentResult
    aggressive: AgentResult
    agents_agree: bool
    
    # Supervisor result (if needed)
    supervisor_chosen: Optional[str] = None
    supervisor_reasoning: Optional[str] = None
    
    # Final result
    final_verdict: str = ""
    is_correct: Optional[bool] = None
    total_cost: float = 0.0


class ProgressTracker:
    """Track progress with ETA calculation."""
    
    def __init__(self, total: int, name: str = "Processing"):
        self.total = total
        self.name = name
        self.completed = 0
        self.start_time = time.time()
        self.correct = 0
        self.last_cost = 0
    
    def update(self, is_correct: bool = None, cost: float = 0):
        self.completed += 1
        if is_correct is not None and is_correct:
            self.correct += 1
        self.last_cost = cost
        self._print_progress()
    
    def _print_progress(self):
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.completed if self.completed > 0 else 0
        remaining = (self.total - self.completed) * avg_time
        
        pct = (self.completed / self.total) * 100
        accuracy = (self.correct / self.completed * 100) if self.completed > 0 else 0
        total_cost = cost_tracker.get_total_cost_inr()
        
        # Format ETA
        if remaining > 60:
            eta_str = f"{remaining/60:.1f}m"
        else:
            eta_str = f"{remaining:.0f}s"
        
        # Clear line and print progress
        print(f"\r[{self.completed}/{self.total}] {pct:.1f}% | "
              f"Acc: {accuracy:.1f}% | "
              f"Cost: ₹{total_cost:.2f} | "
              f"ETA: {eta_str}    ", end="", flush=True)
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n[DONE] {self.completed} samples in {elapsed:.1f}s")


class EnsembleLogger:
    """Logger for ensemble verification."""
    
    def __init__(self, run_name: str = "ensemble"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"{run_name}_{timestamp}.log"
        self.results_file = LOG_DIR / f"{run_name}_{timestamp}_results.jsonl"
        self._write(f"=== Ensemble Verification Log ===")
        self._write(f"Started: {datetime.now().isoformat()}")
    
    def _write(self, msg: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    
    def log_sample(self, sample_id: int, character: str):
        self._write(f"\n[SAMPLE {sample_id}] Character: {character}")
    
    def log_agents(self, conservative: AgentResult, aggressive: AgentResult):
        self._write(f"  Conservative: {conservative.verdict} ({conservative.confidence:.2f})")
        self._write(f"    Reason: {conservative.reasoning[:60]}...")
        self._write(f"  Aggressive: {aggressive.verdict} ({aggressive.confidence:.2f})")
        self._write(f"    Reason: {aggressive.reasoning[:60]}...")
    
    def log_agreement(self, agree: bool, verdict: str):
        if agree:
            self._write(f"  [AGREE] Both agents say: {verdict}")
        else:
            self._write(f"  [DISAGREE] Calling supervisor (Pro)...")
    
    def log_supervisor(self, chosen: str, reasoning: str):
        self._write(f"  Supervisor chose: {chosen}")
        self._write(f"    Reason: {reasoning[:60]}...")
    
    def log_result(self, result: EnsembleResult):
        status = "[CORRECT]" if result.is_correct else "[WRONG]" if result.is_correct is not None else "[TEST]"
        self._write(f"  {status} Final: {result.final_verdict} | Actual: {result.actual_label or 'N/A'}")
        
        # Save to JSONL
        with open(self.results_file, "a", encoding="utf-8") as f:
            data = {
                "sample_id": result.sample_id,
                "book_name": result.book_name,
                "character": result.character,
                "backstory": result.backstory[:200],
                "actual_label": result.actual_label,
                "conservative_verdict": result.conservative.verdict,
                "conservative_reasoning": result.conservative.reasoning,
                "aggressive_verdict": result.aggressive.verdict,
                "aggressive_reasoning": result.aggressive.reasoning,
                "agents_agree": result.agents_agree,
                "supervisor_chosen": result.supervisor_chosen,
                "supervisor_reasoning": result.supervisor_reasoning,
                "final_verdict": result.final_verdict,
                "is_correct": result.is_correct,
                "total_cost": result.total_cost
            }
            f.write(json.dumps(data, default=str) + "\n")


def filter_constraints_by_character(constraints: dict, character: str) -> dict:
    """Filter constraints to character."""
    char_lower = character.lower()
    char_parts = char_lower.replace("/", " ").replace("-", " ").split()
    
    def matches(text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        if char_lower in text_lower:
            return True
        for part in char_parts:
            if len(part) > 2 and part in text_lower:
                return True
        return False
    
    filtered = {"events": [], "relationships": [], "capabilities": [], "timeline_facts": []}
    
    for event in constraints.get("events", []):
        if matches(event.get("description", "")) or any(matches(c) for c in event.get("characters", [])):
            filtered["events"].append(event)
    
    for rel in constraints.get("relationships", []):
        if matches(rel.get("char_a", "")) or matches(rel.get("char_b", "")):
            filtered["relationships"].append(rel)
    
    for cap in constraints.get("capabilities", []):
        if matches(cap.get("character", "")):
            filtered["capabilities"].append(cap)
    
    for fact in constraints.get("temporal", {}).get("timeline_facts", []):
        if matches(fact.get("character", "")):
            filtered["timeline_facts"].append(fact)
    
    return filtered


def find_character_passages(character: str, novel_text: str, max_chars: int = 50000) -> str:
    """
    Find passages in novel that mention the character.
    Used as fallback when constraints are sparse.
    """
    char_lower = character.lower().split("/")[0].split()[0]  # First part of name
    
    # Find all occurrences of character name
    passages = []
    chunk_size = 10000
    
    for i in range(0, len(novel_text), chunk_size):
        chunk = novel_text[i:i + chunk_size]
        if char_lower in chunk.lower():
            # Expand context around this chunk
            start = max(0, i - 5000)
            end = min(len(novel_text), i + chunk_size + 5000)
            passage = novel_text[start:end]
            passages.append(passage)
            
            if sum(len(p) for p in passages) >= max_chars:
                break
    
    if passages:
        return "\n\n---\n\n".join(passages)[:max_chars]
    
    # Fallback: return first and last portions of novel
    return novel_text[:25000] + "\n\n...\n\n" + novel_text[-25000:]


def build_conservative_prompt(backstory: str, character: str, constraints: dict, excerpt: str) -> str:
    """Conservative prompt - leans toward consistent."""
    constraints_str = json.dumps(constraints, indent=2, default=str)[:6000]
    
    return f"""You are verifying if a backstory is CONSISTENT with a novel.

CHARACTER: {character}
BACKSTORY: {backstory}

NOVEL FACTS:
{constraints_str}

NOVEL EXCERPT:
{excerpt[:12000]}...

RULES:
- CONTRADICT only if there is a DIRECT, EXPLICIT conflict
- If the novel doesn't mention something, that is NOT a contradiction
- Absence of information = CONSISTENT
- Only flag CLEAR, UNDENIABLE conflicts

If in doubt, say CONSISTENT.

Return JSON only:
{{"verdict": "consistent" or "contradict", "confidence": 0.5-1.0, "reasoning": "explanation"}}"""


def build_aggressive_prompt(backstory: str, character: str, constraints: dict, excerpt: str) -> str:
    """Aggressive prompt - looks hard for contradictions."""
    constraints_str = json.dumps(constraints, indent=2, default=str)[:6000]
    
    return f"""You are a fact-checker finding errors in a backstory.

CHARACTER: {character}
BACKSTORY: {backstory}

NOVEL FACTS:
{constraints_str}

NOVEL EXCERPT:
{excerpt[:18000]}...

CHECK FOR ERRORS:
1. Historical errors ("Napoleon's triumph at Waterloo" - he LOST)
2. Wrong titles/roles ("first mate" when novel says "quartermaster")
3. Timeline impossibilities
4. Geographic impossibilities
5. Character relationship errors
6. Any factual inaccuracy

If you find ANY error, say CONTRADICT.
Even one wrong word or title = CONTRADICT.

Return JSON only:
{{"verdict": "consistent" or "contradict", "confidence": 0.5-1.0, "reasoning": "the specific error found"}}"""


def build_fallback_verifier_prompt(backstory: str, character: str, retrieved_passages: str) -> str:
    """Fallback verifier prompt - used when constraints are sparse, uses semantically retrieved passages."""
    
    return f"""You are verifying if a backstory is CONSISTENT with a novel using retrieved passages.

CHARACTER: {character}
BACKSTORY TO VERIFY:
{backstory}

---

RELEVANT PASSAGES FROM THE NOVEL (retrieved via semantic search):
{retrieved_passages[:15000]}

---

YOUR TASK:
Carefully compare the backstory claims against the novel passages above.

Look for:
1. FACTUAL ERRORS: Wrong dates, locations, titles, or roles
2. HISTORICAL ERRORS: Events described incorrectly (e.g., "Napoleon triumphed at Waterloo" when he lost)
3. RELATIONSHIP ERRORS: Wrong relationships between characters
4. TIMELINE ERRORS: Events in wrong order or impossible timing
5. CHARACTER ERRORS: Actions attributed to wrong characters

DECISION RULES:
- If you find a CLEAR, SPECIFIC factual error → CONTRADICT
- If the backstory adds details not mentioned in the novel but doesn't conflict → CONSISTENT
- If you cannot verify a claim because the passages don't cover it → CONSISTENT (absence ≠ contradiction)
- Only mark CONTRADICT if you can point to a SPECIFIC error

Return JSON only:
{{"verdict": "consistent" or "contradict", "confidence": 0.5-1.0, "reasoning": "explain your decision, citing specific passages if contradicting"}}"""


def build_supervisor_prompt(backstory: str, conservative: AgentResult, aggressive: AgentResult) -> str:
    """Supervisor prompt - acts as a JUDGE evaluating both arguments."""
    
    return f"""You are a JUDGE deciding a dispute between two analysts about whether a backstory contradicts a novel.

THE BACKSTORY IN QUESTION:
{backstory}

---

ANALYST A (Conservative) verdict: {conservative.verdict.upper()}
Their argument: "{conservative.reasoning}"

---

ANALYST B (Aggressive) verdict: {aggressive.verdict.upper()}  
Their argument: "{aggressive.reasoning}"

---

YOUR TASK AS JUDGE:

Evaluate BOTH arguments and decide which is correct. Consider:

1. DOES ANALYST B CITE A REAL ERROR?
   - If they say "Napoleon triumphed at Waterloo" - that's a real historical error (he lost)
   - If they say "called X a first mate but novel says quartermaster" - that's a real title error
   - If they say "novel doesn't mention X" - that's NOT an error (absence ≠ contradiction)

2. IS ANALYST B'S ERROR ACTUALLY IN THE BACKSTORY?
   - Read the backstory above carefully
   - Does it actually contain the error they claim?
   - Sometimes analysts imagine errors that aren't there

3. IS ANALYST A CORRECT THAT THERE'S NO CONFLICT?
   - Did they miss something obvious?
   - Or are they right that the backstory is compatible?

DECISION RULES:
- If Analyst B found a REAL, VERIFIABLE factual error in the backstory → Choose AGGRESSIVE
- If Analyst B's "error" is just "novel doesn't mention this" → Choose CONSERVATIVE  
- If Analyst B claims an error but can't cite specific conflicting text → Choose CONSERVATIVE
- If Analyst B found a genuine contradiction (wrong date, wrong name, wrong historical fact, wrong title) → Choose AGGRESSIVE
- When in doubt and both make reasonable arguments: Choose AGGRESSIVE if the backstory seems suspicious

Return JSON only:
{{"chosen": "conservative" or "aggressive", "reasoning": "explain your verdict as judge"}}"""


def parse_response(response: str) -> tuple[str, float, str]:
    """Parse agent response."""
    default = ("consistent", 0.5, "Failed to parse")
    
    if not response:
        return default
    
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    
    try:
        data = json.loads(cleaned)
        verdict = data.get("verdict", "consistent").lower().strip()
        if verdict not in ["consistent", "contradict"]:
            verdict = "consistent"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        reasoning = str(data.get("reasoning", "No reasoning"))[:300]
        return verdict, confidence, reasoning
    except:
        if "contradict" in response.lower():
            return "contradict", 0.6, response[:200]
        return default


def parse_supervisor_response(response: str) -> tuple[str, str]:
    """Parse supervisor response."""
    default = ("conservative", "Failed to parse supervisor response")
    
    if not response:
        return default
    
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    
    try:
        data = json.loads(cleaned)
        chosen = data.get("chosen", "conservative").lower().strip()
        if chosen not in ["conservative", "aggressive"]:
            chosen = "conservative"
        reasoning = str(data.get("reasoning", "No reasoning"))[:300]
        return chosen, reasoning
    except:
        if "aggressive" in response.lower():
            return "aggressive", response[:200]
        return default


async def verify_single_sample_async(sample: Sample, constraints: dict, novel_text: str, 
                                      logger: EnsembleLogger, book_name: str = "") -> EnsembleResult:
    """Verify a single sample using async ensemble approach."""
    
    cost_before = cost_tracker.get_total_cost_inr()
    logger.log_sample(sample.id, sample.char)
    
    # Filter constraints
    relevant = filter_constraints_by_character(constraints, sample.char)
    constraint_count = sum(len(v) for v in relevant.values() if isinstance(v, list))
    
    # Check if we should use BDH fallback
    use_bdh_fallback = constraint_count <= FALLBACK_CONSTRAINT_THRESHOLD and BDH_FALLBACK_ENABLED
    
    if use_bdh_fallback:
        # Use BDH-inspired multi-neuron fallback for sparse constraints
        logger._write(f"  [BDH FALLBACK] Constraints: {constraint_count}, using Hebbian neuron fusion")
        
        bdh_verdict, bdh_conf, bdh_reason = await bdh_fallback_verify(
            backstory=sample.content,
            character=sample.char,
            novel_text=novel_text,
            book_name=book_name or sample.book_name,
            sample_id=sample.id
        )
        
        # Create synthetic agent results from BDH output
        conservative = AgentResult(bdh_verdict, bdh_conf, bdh_reason)
        aggressive = AgentResult(bdh_verdict, bdh_conf, bdh_reason)
        
        logger._write(f"  [BDH] Verdict: {bdh_verdict} ({bdh_conf:.2f})")
        logger._write(f"  [BDH] Reasoning: {bdh_reason[:100]}...")
        
        # Skip normal agent path - we already have the verdict
        excerpt = ""  # Not used in BDH path
    elif constraint_count < 5:
        # Zero-constraint fallback without Pathway: find character passages
        excerpt = find_character_passages(sample.char, novel_text, max_chars=50000)
    else:
        # Normal path: find first chunk with character
        char_lower = sample.char.lower().split("/")[0].split()[0]
        excerpt = ""
        for i in range(0, min(len(novel_text), 500000), 50000):
            section = novel_text[i:i+50000]
            if char_lower in section.lower():
                excerpt = section
                break
        if not excerpt:
            excerpt = novel_text[:30000]
    
    # Check if using BDH fallback path (skip normal agents)
    if use_bdh_fallback:
        # BDH already computed the verdict above, agents are synthetic
        pass  # conservative and aggressive already set
    else:
        # Run conservative and aggressive agents in parallel (normal path)
        conservative_prompt = build_conservative_prompt(sample.content, sample.char, relevant, excerpt)
        aggressive_prompt = build_aggressive_prompt(sample.content, sample.char, relevant, excerpt)
        
        # Parallel agent calls
        conservative_task = async_gemini_query(conservative_prompt, ROOT_MODEL, description=f"Conservative {sample.id}")
        aggressive_task = async_gemini_query(aggressive_prompt, ROOT_MODEL, description=f"Aggressive {sample.id}")
        
        conservative_response, aggressive_response = await asyncio.gather(conservative_task, aggressive_task)
        
        con_verdict, con_conf, con_reason = parse_response(conservative_response)
        conservative = AgentResult(con_verdict, con_conf, con_reason)
        
        agg_verdict, agg_conf, agg_reason = parse_response(aggressive_response)
        aggressive = AgentResult(agg_verdict, agg_conf, agg_reason)
    
    logger.log_agents(conservative, aggressive)
    
    # Check agreement
    agents_agree = (conservative.verdict == aggressive.verdict)
    logger.log_agreement(agents_agree, conservative.verdict if agents_agree else "")
    
    # Determine final verdict
    supervisor_chosen = None
    supervisor_reasoning = None
    
    if agents_agree:
        final_verdict = conservative.verdict
    else:
        # Call supervisor with PRO model
        supervisor_prompt = build_supervisor_prompt(sample.content, conservative, aggressive)
        supervisor_response = await async_gemini_query(
            supervisor_prompt, 
            SUPERVISOR_MODEL,  # Use Pro for supervisor
            description=f"Supervisor(Pro) {sample.id}"
        )
        supervisor_chosen, supervisor_reasoning = parse_supervisor_response(supervisor_response)
        
        logger.log_supervisor(supervisor_chosen, supervisor_reasoning)
        
        if supervisor_chosen == "conservative":
            final_verdict = conservative.verdict
        else:
            final_verdict = aggressive.verdict
    
    cost_after = cost_tracker.get_total_cost_inr()
    
    is_correct = None
    if sample.label:
        is_correct = (final_verdict == sample.label)
    
    result = EnsembleResult(
        sample_id=sample.id,
        book_name=sample.book_name,
        character=sample.char,
        backstory=sample.content,
        actual_label=sample.label,
        conservative=conservative,
        aggressive=aggressive,
        agents_agree=agents_agree,
        supervisor_chosen=supervisor_chosen,
        supervisor_reasoning=supervisor_reasoning,
        final_verdict=final_verdict,
        is_correct=is_correct,
        total_cost=cost_after - cost_before
    )
    
    logger.log_result(result)
    return result


# Sync wrapper for backward compatibility
def verify_single_sample(sample: Sample, constraints: dict, novel_text: str, 
                         logger: EnsembleLogger) -> EnsembleResult:
    """Sync wrapper for verify_single_sample_async."""
    return asyncio.run(verify_single_sample_async(sample, constraints, novel_text, logger))


async def verify_samples_ensemble_async(samples: list[Sample], constraints_cache: dict, 
                                         novels_cache: dict, logger: EnsembleLogger = None,
                                         progress: ProgressTracker = None) -> list[EnsembleResult]:
    """Async version - processes samples with controlled concurrency."""
    
    if logger is None:
        logger = EnsembleLogger("ensemble")
    
    if progress is None:
        progress = ProgressTracker(len(samples), "Ensemble Verification")
    
    logger._write(f"\n[START] Verifying {len(samples)} samples with async ensemble")
    
    results = []
    
    # Process in batches to control concurrency
    batch_size = 8  # Process 8 samples at a time
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        
        tasks = []
        for sample in batch:
            constraints = constraints_cache.get(sample.book_name, {})
            novel_text = novels_cache.get(sample.book_name, "")
            tasks.append(verify_single_sample_async(sample, constraints, novel_text, logger, book_name=sample.book_name))
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"\n[ERROR] {result}")
                continue
            results.append(result)
            progress.update(result.is_correct, result.total_cost)
    
    progress.finish()
    
    # Summary
    logger._write(f"\n{'='*60}")
    logger._write(f"=== ENSEMBLE SUMMARY ===")
    
    correct = sum(1 for r in results if r.is_correct)
    total = len([r for r in results if r.actual_label])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    agreements = sum(1 for r in results if r.agents_agree)
    logger._write(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    logger._write(f"Agent agreements: {agreements}/{len(results)}")
    logger._write(f"Supervisor calls: {len(results) - agreements}")
    logger._write(f"Total cost: INR {sum(r.total_cost for r in results):.2f}")
    
    return results


def verify_samples_ensemble(samples: list[Sample], constraints_cache: dict, 
                            novels_cache: dict, logger: EnsembleLogger = None) -> list[EnsembleResult]:
    """Sync wrapper for ensemble verification."""
    
    if logger is None:
        logger = EnsembleLogger("ensemble")
    
    progress = ProgressTracker(len(samples), "Ensemble Verification")
    
    return asyncio.run(
        verify_samples_ensemble_async(samples, constraints_cache, novels_cache, logger, progress)
    )


def load_all_constraints() -> dict:
    """Load cached constraints."""
    cache = {}
    for f in CACHE_DIR.glob("constraints_*.json"):
        book = f.stem.replace("constraints_", "").replace("_", " ")
        try:
            cache[book] = json.loads(f.read_text(encoding='utf-8'))
        except:
            pass
    return cache
