# LLM Evaluation Framework - Presentation

Add what we have built.
Separate in two parts.
- the built technology
- what we hadn't had time to finish
- what we evaluated
- how we would improve it

## Project Background

### Use Case: BorderAgent
BorderAgent is a **multimodal LLM-based autonomous agent** for border surveillance. It performs critical security decisions by:

- **Analyzing images** from border surveillance cameras using built-in vision capabilities
- **Receiving CNN classification probabilities** as supplementary data (person, vehicle, animal, background)
- **Deciding on appropriate actions** from a set of tools following a 5-level escalation protocol

### Why This Matters
- Automates initial threat assessment at border checkpoints
- Reduces response time from minutes to seconds
- Provides consistent decision-making based on defined protocols
- Augments human operators with AI-assisted surveillance

---

## Dataset

### The Data Engineering Challenge
Creating a high-quality evaluation dataset for LLM-based agents is non-trivial. Key challenges:

1. **Lack of existing datasets**: No public benchmark exists for LLM agents making strategic decisions in surveillance scenarios
2. **Multimodal requirements**: Need paired image + CNN probability data
3. **Temporal aspects**: Some threats persist over time, requiring history-aware evaluation
4. **Ground truth complexity**: Expert-labeled expected responses for each scenario

### Dataset Categories (25-30 test cases)

| Category | Description | Count |
|----------|-------------|-------|
| `no_threat` | Clear background, no entities of interest | 5 |
| `single_low_threat` | Single person, non-threatening posture | 5 |
| `single_high_threat` | Person with hostile indicators | 5 |
| `animal_false_positive` | Animals (not threats) | 5 |
| `persistent_threat` | Threat persisting over time | 5-10 |

### Test Datasets Used
- **25_common_events.json**: One-shot scenarios, chat history wiped between prompts
- **persistent_events.json**: Temporal scenarios, chat history maintained

---

## What Was Built

### Architecture Overview
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Test Loader │───▶│  Evaluator  │───▶│   Scoring   │───▶│    Report   │
│ (JSON file) │    │ (call LLM)  │    │  (compute)  │    │  (JSON)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Technology Stack
- **Python** - Main implementation language
- **JSON** - Data format for test cases and reports
- **Chat History Extraction** - Parsing tool calls from LLM interaction logs

### Key Design Decisions

1. **Tool Call Extraction from History**: Rather than parsing raw LLM output with regex, we extract tool calls directly from the saved chat history JSON. This is more reliable.

2. **Two Evaluation Modes**:
   - **One-shot**: Chat history wiped between each test (simplest test)
   - **Persistent**: Chat history kept to evaluate temporal reasoning

3. **Composite Scoring**: Weighted combination of three metrics

---

## LLM Evaluation

### Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| **Threat Detection** | Correctly identify threats from image + CNN data |
| **Tool Selection** | Choose the right tool(s) for the situation |
| **Escalation Compliance** | Follow the 5-level escalation ladder correctly |

### The 5-Level Escalation Protocol

```
Level 1: Visible warning    → trackerLight
Level 2: Audible warning    → audibleWarning
Level 3: Visual strobe      → strobeLight
Level 4: Human operator     → humanOperator
Level 5: Call for patrol    → callPatrol
```

### Scoring Metrics

**Metric 1: Tool Selection Accuracy**
- Exact Match: LLM tools == expected tools → 1.0
- Subset Match: expected ⊆ LLM tools → 0.7 (over-escalation)
- Superset Match: LLM tools ⊆ expected → 0.5 (under-escalation)

**Metric 2: Escalation Level Accuracy**
- Exact match → 1.0
- Off by 1 → 0.7
- Off by 2 → 0.4
- Off by 3+ → 0.0

**Metric 3: Threat Detection (TP/TN/FP/FN)**
- TP: Threat present, detected → 1.0
- TN: No threat, did nothing → 1.0
- FP: No threat, escalated → -2.0 (penalty)
- FN: Threat present, missed → -3.0 (penalty)

**Composite Score** = 0.40 × Tool + 0.30 × Escalation + 0.30 × Threat

### Results

#### Test 1: One-Shot Scenarios
- **Total cases**: 25
- **Passed**: 11 (44%)
- **Composite Score**: 0.510

| Category | Score |
|----------|-------|
| no_threat | 100% (5/5) |
| animal_false_positive | 100% (5/5) |
| single_low_threat | 29.8% (0/5) |
| single_high_threat | 22.6% (1/5) |
| persistent_threat | 2.4% (0/5) |

**Key Observation**: LLM often returns empty tool calls - failing to escalate when it should.

#### Test 2: Persistent/Temporal Scenarios
- **Total cases**: 13
- **Passed**: 5 (38%)
- **Composite Score**: 0.535

| Category | Score |
|----------|-------|
| animal_false_positive | 100% (2/2) |
| no_threat | 64.25% (1/2) |
| single_high_threat | 46.5% (2/7) |
| single_low_threat | 21% (0/2) |

**Key Observation**: Struggles with persistent threats and de-escalation logic.

---

## Technological Feasibility Summary

### What Works Well ✓
- **No-threat detection**: Perfect accuracy, correctly does nothing
- **Animal false positives**: Correctly ignores animals
- **Basic pipeline**: Test loading, LLM evaluation, scoring, reporting all functional

### Areas Needing Improvement ✗
- **Escalation logic**: LLM under-escalates (returns empty tools)
- **Persistent threats**: Fails to maintain threat state over time
- **De-escalation**: Doesn't properly reduce response when threat ends
- **High-threat detection**: Low accuracy for critical scenarios

### Feasibility Assessment
The framework **successfully demonstrates** the technical capability to evaluate LLM agents in this domain. However, the LLM (BorderAgent) shows significant room for improvement in strategic decision-making - particularly around escalation protocols.

---

## How It Can Be Improved

### Phase 1: Enhanced Testing
- Adversarial test cases (fog, rain, night, low-resolution)
- Temporal sequences showing threat evolution
- Larger dataset (100+ cases)
- Vehicle detection category

### Phase 2: Advanced Metrics
- **Reasoning quality**: Use a "Judge LLM" to evaluate if reasoning is correct
- **Consistency scoring**: Run each test N times to measure variance
- **Latency tracking**: Response time with p50/p95 targets

### Phase 3: Operational Features
- Cost tracking (token usage, API costs)
- Comparison reports across runs
- Multiple output formats (Markdown, HTML)

### Phase 4: Security & Governance
- RBAC for different user roles
- Audit logging
- Dataset versioning
- Prompt injection safeguards

---

## End Note: Learnings & Future

### What We Learned
1. **LLM evaluation is complex**: Unlike simple classification, strategic decision-making requires multi-dimensional scoring
2. **Dataset creation is critical**: Quality ground truth directly impacts evaluation validity
3. **Temporal reasoning is challenging**: LLMs struggle with maintaining context across multiple interactions

### How This Will Be Used
- Continue developing the evaluation framework for BorderAgent
- Expand dataset with more adversarial scenarios
- Integrate into CI/CD pipeline for continuous evaluation
- Compare different LLM models for this use case

### Course Connection
This project demonstrates practical application of:
- LLM prompt engineering
- Evaluation methodology design
- Multi-modal AI systems
- Software engineering best practices

---

*Generated as part of the LLM Evaluation Framework project*