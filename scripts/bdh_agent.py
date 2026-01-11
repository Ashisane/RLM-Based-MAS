"""
BDH-Inspired Fallback Agent for Narrative Consistency Checking.

Implements core concepts from Baby Dragon Hatchling (BDH) architecture:
1. Multiple "neuron" units processing evidence in parallel (local computation)
2. Hebbian fusion - evidence that co-occurs strengthens the signal
3. Excitatory/Inhibitory dynamics - some neurons support consistency, others contradiction
4. Sparse activations - only top-k evidence chunks contribute

This is used as a fallback when extracted constraints are sparse (≤1).
"""
import re
import json
import asyncio
from dataclasses import dataclass, asdict
from typing import Optional
import hashlib
from pathlib import Path
from datetime import datetime

from config import cost_tracker, GEMINI_API_KEY, ROOT_MODEL, CACHE_DIR
from gemini_client import async_gemini_query


# ============================================================================
# Caching (avoids redundant processing for same character+book)
# ============================================================================

_novel_chunks_cache: dict[str, list[dict]] = {}
_character_evidence_cache: dict[str, dict] = {}


def get_cache_key(book_name: str, character: str = "") -> str:
    """Generate cache key."""
    key = f"{book_name}:{character}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def get_cached_chunks(book_name: str, novel_text: str) -> list[dict]:
    """Get or create cached chunks for a novel."""
    cache_key = get_cache_key(book_name)
    if cache_key not in _novel_chunks_cache:
        _novel_chunks_cache[cache_key] = chunk_novel(novel_text)
    return _novel_chunks_cache[cache_key]


def get_cached_evidence(book_name: str, character: str, novel_text: str, backstory: str) -> dict:
    """Get or create cached evidence for a character."""
    cache_key = get_cache_key(book_name, character)
    
    if cache_key not in _character_evidence_cache:
        chunks = get_cached_chunks(book_name, novel_text)
        keyword_chunks = keyword_neuron_search(character, backstory, chunks, top_k=5)
        context_passages = context_neuron_search(character, novel_text, max_passages=2)
        
        _character_evidence_cache[cache_key] = {
            "keyword_chunks": keyword_chunks,
            "context_passages": context_passages,
            "keyword_evidence": "\n\n---\n\n".join(c["text"] for c in keyword_chunks),
            "context_evidence": "\n\n---\n\n".join(context_passages) if context_passages else ""
        }
    
    return _character_evidence_cache[cache_key]


def clear_bdh_cache():
    """Clear all BDH caches."""
    global _novel_chunks_cache, _character_evidence_cache
    _novel_chunks_cache.clear()
    _character_evidence_cache.clear()


# ============================================================================
# BDH Core Data Structures
# ============================================================================

@dataclass
class NeuronActivation:
    """Output of a single neuron."""
    neuron_id: str
    neuron_type: str
    signal_type: str  # "excitatory" or "inhibitory"
    activation: float
    reasoning: str


@dataclass
class HebbianFusionResult:
    """Result of Hebbian fusion."""
    verdict: str
    confidence: float
    excitatory_sum: float
    inhibitory_sum: float
    agreement_strength: float
    reasoning: str


# ============================================================================
# Evidence Extraction
# ============================================================================

def chunk_novel(novel_text: str, chunk_size: int = 1500, overlap: int = 300) -> list[dict]:
    """Split novel into overlapping chunks."""
    chunks = []
    i = 0
    chunk_id = 0
    
    while i < len(novel_text):
        end = min(i + chunk_size, len(novel_text))
        chunk_text = novel_text[i:end]
        
        if end < len(novel_text):
            for sep in ['. ', '.\n', '? ', '!\n']:
                last_sep = chunk_text.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk_text = chunk_text[:last_sep + 1]
                    end = i + last_sep + 1
                    break
        
        chunks.append({"id": chunk_id, "text": chunk_text})
        chunk_id += 1
        i = end - overlap if end < len(novel_text) else end
    
    return chunks


def keyword_neuron_search(character: str, backstory: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Find chunks by character name and backstory keywords."""
    char_name = character.lower().split()[0] if character else ""
    backstory_words = set(w.lower() for w in backstory.split() if len(w) > 4)
    
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        score = 0
        if char_name and char_name in chunk_lower:
            score += 5
        for word in backstory_words:
            if word in chunk_lower:
                score += 1
        if score > 0:
            scored_chunks.append((score, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored_chunks[:top_k]]


def context_neuron_search(character: str, novel_text: str, max_passages: int = 3) -> list[str]:
    """Find passages surrounding character mentions."""
    char_name = character.lower().split()[0] if character else ""
    if not char_name:
        return []
    
    passages = []
    search_pos = 0
    window_size = 2000
    novel_lower = novel_text.lower()
    
    while len(passages) < max_passages:
        pos = novel_lower.find(char_name, search_pos)
        if pos == -1:
            break
        start = max(0, pos - window_size // 2)
        end = min(len(novel_text), pos + window_size // 2)
        passage = novel_text[start:end]
        if not any(p[:100] == passage[:100] for p in passages):
            passages.append(passage)
        search_pos = pos + len(char_name)
    
    return passages


# ============================================================================
# Neuron Activation (LLM-based)
# ============================================================================

async def activate_semantic_neuron(backstory: str, character: str, evidence: str) -> NeuronActivation:
    """Semantic analysis of evidence vs backstory."""
    prompt = f"""Analyze if EVIDENCE supports or contradicts BACKSTORY.

CHARACTER: {character}
BACKSTORY: {backstory[:500]}

EVIDENCE:
{evidence[:3000]}

Return JSON: {{"signal": "excitatory" or "inhibitory", "activation": 0.0-1.0, "reasoning": "brief"}}"""

    try:
        response = await async_gemini_query(prompt, ROOT_MODEL, description="Semantic")
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
        data = json.loads(cleaned)
        return NeuronActivation(
            neuron_id="semantic", neuron_type="semantic",
            signal_type=data.get("signal", "excitatory"),
            activation=max(0.0, min(1.0, float(data.get("activation", 0.5)))),
            reasoning=data.get("reasoning", "")[:150]
        )
    except:
        return NeuronActivation("semantic", "semantic", "excitatory", 0.3, "parse error")


async def activate_factcheck_neuron(backstory: str, character: str, evidence: str) -> NeuronActivation:
    """Hunt for factual errors (inhibitory-biased)."""
    prompt = f"""Hunt for factual errors in BACKSTORY vs EVIDENCE.

CHARACTER: {character}
BACKSTORY: {backstory[:500]}
EVIDENCE: {evidence[:3000]}

Look for: wrong dates/names/titles, historical errors, timeline issues.
Return JSON: {{"signal": "excitatory" or "inhibitory", "activation": 0.0-1.0, "error_found": "desc or null"}}"""

    try:
        response = await async_gemini_query(prompt, ROOT_MODEL, description="FactCheck")
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
        data = json.loads(cleaned)
        error = data.get("error_found")
        return NeuronActivation(
            neuron_id="factcheck", neuron_type="factcheck",
            signal_type=data.get("signal", "excitatory"),
            activation=max(0.0, min(1.0, float(data.get("activation", 0.3)))),
            reasoning=error if error else "no errors"
        )
    except:
        return NeuronActivation("factcheck", "factcheck", "excitatory", 0.2, "parse error")


async def activate_context_neuron(backstory: str, character: str, context: str) -> NeuronActivation:
    """Check if backstory fits character portrayal (excitatory-biased)."""
    prompt = f"""Does BACKSTORY fit CHARACTER's portrayal in CONTEXT?

CHARACTER: {character}
BACKSTORY: {backstory[:500]}
CONTEXT: {context[:3000]}

Evaluate: personality match, role match, timeline plausibility.
Return JSON: {{"signal": "excitatory" or "inhibitory", "activation": 0.0-1.0, "reasoning": "brief"}}"""

    try:
        response = await async_gemini_query(prompt, ROOT_MODEL, description="Context")
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
        data = json.loads(cleaned)
        return NeuronActivation(
            neuron_id="context", neuron_type="context",
            signal_type=data.get("signal", "excitatory"),
            activation=max(0.0, min(1.0, float(data.get("activation", 0.5)))),
            reasoning=data.get("reasoning", "")[:150]
        )
    except:
        return NeuronActivation("context", "context", "excitatory", 0.3, "parse error")


# ============================================================================
# Hebbian Fusion
# ============================================================================

def hebbian_fusion(activations: list[NeuronActivation]) -> HebbianFusionResult:
    """Combine neuron signals using Hebbian principles."""
    if not activations:
        return HebbianFusionResult("consistent", 0.5, 0.0, 0.0, 0.0, "No neurons")
    
    excitatory_sum = sum(a.activation for a in activations if a.signal_type == "excitatory")
    inhibitory_sum = sum(a.activation for a in activations if a.signal_type == "inhibitory")
    total = excitatory_sum + inhibitory_sum
    
    if inhibitory_sum == 0:
        verdict = "consistent"
        confidence = min(0.95, 0.5 + excitatory_sum / len(activations) * 0.5)
        agreement = 1.0
    elif excitatory_sum == 0:
        verdict = "contradict"
        confidence = min(0.95, 0.5 + inhibitory_sum / len(activations) * 0.5)
        agreement = 1.0
    else:
        agreement = abs(excitatory_sum - inhibitory_sum) / total if total > 0 else 0
        if excitatory_sum > inhibitory_sum:
            verdict = "consistent"
            confidence = 0.5 + (excitatory_sum - inhibitory_sum) / total * 0.4
        else:
            verdict = "contradict"
            confidence = 0.5 + (inhibitory_sum - excitatory_sum) / total * 0.4
    
    reasons = [a.reasoning for a in activations if a.activation > 0.3]
    reasoning = f"E={excitatory_sum:.2f} I={inhibitory_sum:.2f}. " + (reasons[0][:80] if reasons else "")
    
    return HebbianFusionResult(verdict, round(confidence, 3), excitatory_sum, inhibitory_sum, agreement, reasoning)


# ============================================================================
# Main BDH Fallback
# ============================================================================

async def bdh_fallback_verify(backstory: str, character: str, novel_text: str, 
                               book_name: str = "", sample_id: int = 0) -> tuple[str, float, str]:
    """
    BDH-Inspired Fallback Verification.
    
    Returns: (verdict, confidence, reasoning)
    """
    # Get evidence (cached)
    evidence = get_cached_evidence(book_name, character, novel_text, backstory)
    keyword_evidence = evidence["keyword_evidence"]
    context_evidence = evidence["context_evidence"]
    
    # Activate neurons in parallel
    tasks = []
    if keyword_evidence:
        tasks.append(activate_semantic_neuron(backstory, character, keyword_evidence))
        tasks.append(activate_factcheck_neuron(backstory, character, keyword_evidence))
    if context_evidence:
        tasks.append(activate_context_neuron(backstory, character, context_evidence))
    
    activations = await asyncio.gather(*tasks) if tasks else []
    
    # Hebbian fusion
    fusion = hebbian_fusion(list(activations))
    
    return fusion.verdict, fusion.confidence, f"[BDH] {fusion.reasoning}"


# For debugging - get detailed result
async def bdh_fallback_verify_detailed(backstory: str, character: str, novel_text: str, 
                                        book_name: str = "") -> dict:
    """Returns detailed result for debugging."""
    evidence = get_cached_evidence(book_name, character, novel_text, backstory)
    
    tasks = []
    if evidence["keyword_evidence"]:
        tasks.append(activate_semantic_neuron(backstory, character, evidence["keyword_evidence"]))
        tasks.append(activate_factcheck_neuron(backstory, character, evidence["keyword_evidence"]))
    if evidence["context_evidence"]:
        tasks.append(activate_context_neuron(backstory, character, evidence["context_evidence"]))
    
    activations = await asyncio.gather(*tasks) if tasks else []
    fusion = hebbian_fusion(list(activations))
    
    return {
        "verdict": fusion.verdict,
        "confidence": fusion.confidence,
        "neurons": [asdict(a) for a in activations],
        "fusion": {
            "excitatory": fusion.excitatory_sum,
            "inhibitory": fusion.inhibitory_sum,
            "agreement": fusion.agreement_strength
        },
        "evidence_counts": {
            "keyword_chunks": len(evidence["keyword_chunks"]),
            "context_passages": len(evidence["context_passages"])
        }
    }
