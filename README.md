# BDH-Driven Narrative Consistency Verification: A Neuro-Symbolic Approach to Long-Context Causal Reasoning

**Team 2easy | Track B: BDH-Driven Continuous Narrative Reasoning**
**Kharagpur Data Science Hackathon 2026**

---

## Abstract

We present a novel application of Brain-Derived Heuristics (BDH) to the narrative consistency verification problem—determining whether hypothetical character backstories contradict source novels spanning 100,000+ words. Our approach draws directly from BDH's core mechanisms: excitatory/inhibitory neuron competition, Hebbian learning dynamics, and sparse distributed representations. Unlike conventional transformer-based approaches that suffer from attention dilution and context collapse in long documents, our BDH-inspired architecture maintains semantic coherence through localized computation and signal aggregation patterns that mirror biological neural networks. Through principled iteration across multiple architectural paradigms, we demonstrate that BDH principles translate effectively to narrative reasoning tasks, achieving 70.9% average accuracy while maintaining full interpretability of decisions.

**Keywords**: Baby Dragon Hatchling, Narrative Reasoning, Long-Context NLP, Neuro-Symbolic AI, Causal Inference

---

## 1. Introduction

### 1.1 Problem Formulation

The narrative consistency verification task presents a unique challenge at the intersection of natural language understanding, causal reasoning, and long-context processing. Given:
- A source novel N (100,000-500,000 words)
- A character C mentioned in N  
- A hypothetical backstory B for character C

The task requires binary classification: does B **contradict** N, or is B **consistent** with N?

This seemingly simple formulation masks substantial complexity:

1. **Temporal Scope Disambiguation**: Backstories describe a character's past (before the novel begins), while novels describe their present. The system must distinguish between contradiction and mere temporal separation.

2. **Implicit vs. Explicit Constraints**: Novels rarely state "Character X never visited France." Contradictions emerge from implicit world-building, character capabilities, and causal chains.

3. **Long-Range Dependencies**: A fact established in Chapter 1 may only become relevant to a backstory claim through events in Chapter 47. Traditional attention mechanisms collapse over such distances [Vaswani et al., 2017; Press et al., 2022].

4. **Semantic Ambiguity**: References like "the mate," "his father," or "the captain" may refer to different individuals in backstory vs. novel contexts.

### 1.2 Why BDH?

The Baby Dragon Hatchling (BDH) architecture [referenced in problem statement] offers a fundamentally different computational paradigm than transformer-based approaches:

| Aspect | Transformers | BDH |
|--------|--------------|-----|
| Attention | Global, quadratic | Local, sparse |
| Memory | Fixed context window | Distributed persistent state |
| Reasoning | Pattern correlation | Signal competition |
| Evidence aggregation | Soft attention weights | Hebbian strengthening |
| Interpretability | Attention visualization | Neuron activation tracing |

For narrative consistency verification, BDH's advantages are decisive:

1. **Excitatory/Inhibitory Competition**: Mirrors the epistemic structure of our task—evidence either supports (excitatory) or contradicts (inhibitory) consistency.

2. **Hebbian Learning**: "Neurons that fire together wire together" translates to "Evidence that agrees strengthens the signal"—exactly what multi-source verification requires.

3. **Sparse Activation**: Only relevant constraints "fire" for a given backstory claim, avoiding the attention dilution problem.

4. **Incremental Belief Formation**: BDH naturally handles sequential evidence processing without catastrophic forgetting.

### 1.3 Contributions

1. **Novel BDH Application**: First application of BDH principles to narrative reasoning and long-context causal inference.

2. **Neuro-Symbolic Architecture**: Hybrid system combining symbolic constraint extraction with neural signal competition.

3. **Principled Design Methodology**: Documented iteration through BDH → RLM → MAS paradigms, demonstrating how BDH principles guided each stage.

4. **Interpretable Decisions**: Every verdict traces to specific neuron activations, constraint matches, and evidence quotes.

---

## 2. Related Work

### 2.1 Long-Context Language Models

Recent work has pushed context windows beyond 100K tokens [Anthropic, 2024; Google, 2024], but length ≠ comprehension. Studies show attention entropy increases with context length, leading to "lost in the middle" phenomena [Liu et al., 2023]. Our approach sidesteps this by processing locally and aggregating globally.

### 2.2 Narrative Understanding

Story comprehension benchmarks [Mostafazadeh et al., 2016; Kočiský et al., 2018] typically involve short texts. Long-form narrative reasoning remains underexplored, with most systems relying on summarization—losing the fine-grained details critical for consistency checking.

### 2.3 Neural-Symbolic Integration

Neuro-symbolic approaches [Garcez et al., 2019] combine neural pattern recognition with symbolic reasoning. Our architecture extends this paradigm by mapping BDH's biological mechanisms to symbolic constraint operations.

### 2.4 Multi-Agent Debate

Recent work on multi-agent debate [Du et al., 2023; Liang et al., 2023] shows that competing agents can improve reasoning quality. We formalize this through BDH's excitatory/inhibitory framework.

---

## 3. Methodology

### 3.1 Theoretical Foundation: BDH → Verification Architecture

We establish a formal mapping between BDH mechanisms and verification operations:

```
BDH Mechanism                    Verification Operation
─────────────────────────────────────────────────────────────
Excitatory Neurons (E+)    →    Conservative Agent (assumes consistency)
Inhibitory Neurons (I-)    →    Aggressive Agent (hunts contradictions)
Synaptic Weights           →    Confidence scores on claims
Hebbian Plasticity         →    Agreement strengthens signal
Sparse Coding              →    Character-filtered constraint retrieval
Lateral Inhibition         →    Conflicting evidence competition
Winner-Take-All            →    Supervisor arbitration
Action Potential Threshold →    Severity threshold for verdict
```

This mapping is not metaphorical—it directly informs our implementation decisions.

### 3.2 System Architecture

Our system implements three phases, each grounded in BDH principles:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: SPARSE ENCODING                        │
│                   (BDH Sparse Distributed Representations)          │
│                                                                     │
│   Novel → Overlapping Chunks → Parallel Extraction → Typed Facts   │
│                                                                     │
│   Constraint Types (Monosemantic Synapses):                        │
│   • Events: {description, characters, timing}                       │
│   • Timeline: {character, age/date, context}                        │
│   • Relationships: {char_a, char_b, type, nature}                   │
│   • Capabilities: {character, skill, evidence}                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: SIGNAL COMPETITION                        │
│                (BDH Excitatory/Inhibitory Dynamics)                 │
│                                                                     │
│   ┌─────────────────┐              ┌─────────────────┐             │
│   │  CONSERVATIVE   │              │   AGGRESSIVE    │             │
│   │    AGENT (E+)   │              │   AGENT (I-)    │             │
│   │                 │              │                 │             │
│   │  "Consistent    │              │  "Find ANY      │             │
│   │   unless        │              │   contradiction"│             │
│   │   impossible"   │              │                 │             │
│   │                 │              │                 │             │
│   │  Activation:    │              │  Activation:    │             │
│   │  consistency    │              │  contradiction  │             │
│   │  signal         │              │  signal         │             │
│   └────────┬────────┘              └────────┬────────┘             │
│            │                                │                       │
│            └──────────┬─────────────────────┘                       │
│                       ▼                                             │
│            ┌─────────────────────┐                                  │
│            │  HEBBIAN FUSION     │                                  │
│            │                     │                                  │
│            │  Agreement →        │                                  │
│            │  Strengthened       │                                  │
│            │  confidence         │                                  │
│            │                     │                                  │
│            │  Conflict →         │                                  │
│            │  Supervisor         │                                  │
│            │  arbitration        │                                  │
│            └─────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PHASE 3: BDH FALLBACK VERIFICATION                  │
│               (Multi-Neuron Activation for Sparse Cases)            │
│                                                                     │
│   When extracted constraints ≤ 1:                                   │
│                                                                     │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│   │  SEMANTIC  │  │ FACT-CHECK │  │  CONTEXT   │                   │
│   │  NEURON    │  │  NEURON    │  │  NEURON    │                   │
│   │            │  │            │  │            │                   │
│   │ Alignment  │  │ Error      │  │ Character  │                   │
│   │ analysis   │  │ hunting    │  │ portrayal  │                   │
│   │            │  │ (I- bias)  │  │ (E+ bias)  │                   │
│   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                   │
│         │               │               │                           │
│         └───────────────┼───────────────┘                           │
│                         ▼                                           │
│              Hebbian Fusion: Σ(activation × signal_type)            │
│                         │                                           │
│                         ▼                                           │
│              {verdict, confidence, reasoning}                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Phase 1: Sparse Constraint Encoding

Following BDH's sparse distributed representation principle, we encode novels as typed constraint sets rather than dense embeddings.

**Chunking Strategy**:
- Chunk size: 80,000 characters (respects model context limits)
- Overlap: 20,000 characters (prevents boundary artifacts)
- Max chunks: 8 (cost optimization)
- Processing: 4 parallel workers

**Constraint Types** (analogous to BDH's monosemantic synapses—each carries specific semantic meaning):

1. **Events**: `{description, characters[], timing}`
2. **Timeline Facts**: `{character, age_or_date, context}`
3. **Relationships**: `{char_a, char_b, type, nature}`
4. **Capabilities**: `{character, skill, evidence}`

**Character-Aware Filtering**:
```python
def matches(text: str, character: str) -> bool:
    char_parts = character.lower().replace("/", " ").split()
    for part in char_parts:
        if len(part) > 2 and part in text.lower():
            return True
    return False
```

This handles compound names (e.g., "Tom Ayrton/Ben Joyce") and aliases.

### 3.4 Phase 2: Excitatory/Inhibitory Agent Competition

**Conservative Agent (Excitatory Signal)**:
```
Role: Assume backstory is consistent unless IMPOSSIBLE
Bias: Benefit of the doubt
Firing condition: No explicit contradiction found
Output: Consistency signal strength [0.0-1.0]
```

**Aggressive Agent (Inhibitory Signal)**:
```
Role: Hunt for ANY contradiction
Bias: Skeptical, detail-oriented
Firing condition: Any potential inconsistency detected
Output: Contradiction signal strength [0.0-1.0]
```

**Hebbian Fusion Rule**:
```
if conservative.verdict == aggressive.verdict:
    # Agreement strengthens signal
    final_confidence = max(conservative.confidence, aggressive.confidence) + 0.1
    final_verdict = conservative.verdict
else:
    # Conflict triggers supervisor arbitration
    supervisor_decision = arbitrate(conservative, aggressive)
```

**Supervisor Design** (Critical Innovation):

The supervisor receives ONLY:
- Original backstory
- Conservative argument + confidence
- Aggressive argument + confidence

The supervisor does NOT receive:
- Novel text
- Extracted constraints
- Character mentions

This forces judgment based on **argument quality**, not re-processing of evidence—mimicking BDH's winner-take-all mechanism based on signal strength rather than re-computation.

### 3.5 Phase 3: Multi-Neuron BDH Fallback

When constraint extraction yields ≤1 relevant fact (sparse activation scenario), we activate the BDH fallback with three specialized neurons:

**Neuron 1: Semantic Analysis** (Balanced)
- Input: Backstory + relevant novel chunks
- Output: Overall semantic alignment assessment
- Activation threshold: 0.5

**Neuron 2: Fact-Check** (Inhibitory-biased)
- Input: Backstory claims + character evidence
- Output: Specific error detection
- Activation threshold: 0.4 (more sensitive to contradictions)

**Neuron 3: Context Evaluation** (Excitatory-biased)
- Input: Character portrayal comparison
- Output: Characterization consistency
- Activation threshold: 0.6 (more permissive)

**Hebbian Aggregation**:
```python
excitatory_sum = sum(n.strength for n in neurons if n.signal == "consistent")
inhibitory_sum = sum(n.strength for n in neurons if n.signal == "contradict")

if all_agree:
    hebbian_strength = 0.92 + (0.08 * avg_confidence)
elif majority_agree:
    hebbian_strength = 0.75
else:
    hebbian_strength = 0.5  # Maximum uncertainty

final_verdict = "consistent" if excitatory_sum > inhibitory_sum else "contradict"
```

---

## 4. Design Evolution: Principled Iteration

Our final architecture emerged through systematic exploration, with each iteration preserving BDH principles while addressing discovered limitations.

### 4.1 Iteration 1: Pure BDH Multi-Neuron Pipeline

**Approach**: Direct implementation of BDH with 5 neuron types processing novel chunks.

**Discovery**: While theoretically elegant, the approach struggled with:
- Long-context coherence (neurons firing on distant chunks couldn't coordinate)
- Chunking artifacts (critical information split across boundaries)
- Cost explosion (full BDH processing on 2.5M character novels)

**BDH Insight Preserved**: The multi-neuron, signal-competition paradigm remained central.

### 4.2 Iteration 2: RLM + Threshold Classification

**Approach**: Recursive Language Model for constraint extraction + learned threshold for binary classification.

**Discovery**: The FP/FN oscillation problem:
- Threshold at 1.5 → 80% false positives (everything flagged as contradiction)
- Threshold at 2.5 → 100% false negatives (nothing flagged)
- No stable equilibrium existed

**Root Cause Analysis**: The severity accumulation problem:
```
3 weak signals × 1.0 severity = 3.0 > any_threshold
→ Aggregated noise triggers false positives
```

**BDH Insight Applied**: Single strong signals should dominate, not accumulated weak signals—matching BDH's lateral inhibition where strong activations suppress weak ones.

### 4.3 Iteration 3: Multi-Agent System with Judge

**Approach**: Conservative + Aggressive agents with supervisor arbitration.

**Discovery**: Agent agreement correlates with accuracy:
- Both agree → 78% accuracy
- Disagreement → 62% accuracy (hardest cases)

**BDH Insight Applied**: Excitatory/Inhibitory competition with Hebbian fusion for agreement cases.

### 4.4 Iteration 4: Hybrid BDH + MAS (Final)

**Approach**: 
- MAS for cases with rich constraints
- BDH fallback for sparse constraint cases
- Hebbian fusion throughout

**Key Innovation**: The supervisor-without-context design forces argument-quality-based arbitration, preventing re-computation bias.

---

## 5. Critical Design Decisions

### 5.1 Why Not Fine-Tuning?

Fine-tuning on the training set would:
1. Introduce dataset-specific biases
2. Lose generalization to unseen novels
3. Obscure the reasoning process

Our BDH approach maintains **task-agnostic reasoning**—the same architecture handles any novel without adaptation.

### 5.2 Why Separate Agents Instead of Single Prompt?

Single-prompt approaches suffer from:
1. Mode collapse (system picks one interpretation)
2. Confirmation bias (early evidence anchors conclusions)
3. Lost nuance (no signal competition)

BDH's excitatory/inhibitory separation explicitly models epistemic conflict.

### 5.3 Why Supervisor Without Evidence Access?

Giving the supervisor novel access would:
1. Allow re-processing (defeating the purpose of agents)
2. Introduce recency bias
3. Lose the "argument quality" evaluation

BDH's winner-take-all operates on signal strength, not re-computation.

### 5.4 Why Conservative Default?

Our prompt instructs: "Is there ANY reasonable interpretation where both can be true? If yes → CONSISTENT"

This mirrors BDH's incremental belief formation—extraordinary claims (contradictions) require extraordinary evidence.

---

## 6. Temporal Scope: The Critical Challenge

### 6.1 The Problem

The most pernicious error pattern we encountered:

```
Backstory: "The mate was hanged the night before"
Novel: Shows Tom Austin (the mate) alive and well

WRONG System Output: "CONTRADICTION - The mate cannot be both hanged and alive"
CORRECT Reasoning: The backstory refers to a different mate from the 
                   character's past, not Tom Austin from the novel's present
```

### 6.2 The Solution

We implemented explicit temporal scope prompting:

```
CRITICAL DISTINCTION:
- Backstory describes character's PAST (before novel begins)
- Novel describes character's PRESENT (during the story)
- New backstory information that doesn't CLASH with novel is ALLOWED
- Only mark CONTRADICT if novel EXPLICITLY makes backstory IMPOSSIBLE
```

This directly maps to BDH's temporal integration—past activations inform but don't override present computations.

---

## 7. Results

### 7.1 Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy (Average) | 70.9% |
| Training Accuracy (Maximum) | 74.2% |
| Training Accuracy (Minimum) | 63.7% |
| Total Cost | ₹1,200 INR |
| Average Cost/Sample | ~₹8.5 INR |

### 7.2 Agent Agreement Analysis

| Condition | Accuracy | Samples |
|-----------|----------|---------|
| Both Agents Agree | ~78% | ~65% |
| Agents Disagree (Supervisor Called) | ~62% | ~35% |

This validates the BDH intuition: agreement (Hebbian strengthening) correlates with confidence.

### 7.3 Error Distribution

| Error Type | Percentage | Primary Cause |
|------------|------------|---------------|
| False Negatives | 48.3% | Sparse constraints, missed evidence |
| False Positives | 19.6% | Temporal scope confusion |

### 7.4 BDH Fallback Effectiveness

For samples with ≤1 extracted constraint:
- Without BDH fallback: 52% accuracy (random baseline)
- With BDH fallback: 68% accuracy (+16% improvement)

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Extraction Coverage**: 8 chunks from 2.5M character novels samples ~25% of text
2. **Reference Ambiguity**: Role-based references ("the mate") remain challenging
3. **Implicit Constraints**: World-building rules are underextracted

### 8.2 Future Improvements

1. **Adaptive Chunking**: Use character mention density to guide chunk selection
2. **Coreference Resolution**: Resolve role references before constraint matching
3. **World Rule Extraction**: Dedicated neuron for implicit world-building
4. **Hebbian Persistence**: Maintain constraint weights across verification calls

---

## 9. Conclusion

We have demonstrated that BDH principles—excitatory/inhibitory competition, Hebbian learning, sparse distributed representations—translate effectively to the narrative consistency verification task. Our architecture achieves competitive accuracy while maintaining full interpretability, with every decision traceable to specific neuron activations and constraint matches.

The journey from pure BDH to our hybrid architecture illustrates that biological neural mechanisms provide not just metaphors but actionable design principles for NLP systems. As language models scale, the attention dilution problem will only worsen; BDH-inspired architectures offer a principled alternative.

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases." ICLR.
3. Liu, N., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv.
4. Mostafazadeh, N., et al. (2016). "A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories." NAACL.
5. Kočiský, T., et al. (2018). "The NarrativeQA Reading Comprehension Challenge." TACL.
6. Garcez, A., et al. (2019). "Neural-Symbolic Computing: An Effective Methodology for Principled Integration." JAIR.
7. Du, Y., et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate." arXiv.
8. Liang, T., et al. (2023). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate." arXiv.

---

## Appendix A: Visualization Gallery

- **Figure 1**: BDH Signal Flow Architecture
- **Figure 2**: Neuron Activation Patterns (Contradiction Detection)
- **Figure 3**: Hebbian Fusion Scenarios
- **Figure 4**: BDH vs Transformer Capability Comparison

---

## Appendix B: Code Structure

```
├── constraint_extraction.py   # Phase 1: Sparse encoding
├── ensemble_verification.py   # Phase 2: E/I competition
├── bdh_agent.py              # Phase 3: Multi-neuron fallback
├── visualizations.py         # Analysis generation
├── gemini_client.py          # API with cost tracking
└── config.py                 # Model configuration
```

---

*Team 2easy — IIIT Sonepat*