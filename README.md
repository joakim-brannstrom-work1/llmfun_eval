# LLM Strategic Vision Evaluation Framework

Evaluates how well an LLM makes strategic decisions based on a system prompt and vision input. The system prompt describes how the LLM should behave when it finds certain objects in images. This framework measures both **image understanding** and **strategy application**.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              EVALUATION FLOW                 │
                    └─────────────────────────────────────────────┘

    Image ──► LLM (vision + system prompt) ──► Raw Response
                                          │
                                          ├─► Parse: identified objects
                                          ├─► Parse: decision/action
                                          └─► Parse: reasoning/rules cited
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STAGE 1: PERCEPTION METRICS                                 │
    │  ┌──────────────────────────────────────────────────────────┐│
    │  │ Object Precision: % of predicted objects that exist      ││
    │  │ Object Recall:    % of ground truth objects detected     ││
    │  │ Object F1:        Harmonic mean of P and R               ││
    │  │ Exact Match Rate: % samples with perfect object match    ││
    │  │ Per-Object Breakdown: P/R per object type                ││
    │  └──────────────────────────────────────────────────────────┘│
    │                                                              │
    │  STAGE 2: STRATEGY METRICS                                   │
    │  ┌──────────────────────────────────────────────────────────┐│
    │  │ Decision Accuracy: % correct final decisions             ││
    │  │ Rule Recall:      % of rules that should fire, did fire  ││
    │  │ Rule Precision:   % of rules that fired, should fire     ││
    │  │ Hallucinated:     Rules applied that shouldn't be        ││
    │  │ Missed:           Rules that should fire but didn't      ││
    │  └──────────────────────────────────────────────────────────┘│
    │                                                              │
    │  COMBINED: weighted geometric mean of perception × strategy  │
    └──────────────────────────────────────────────────────────────┘
```

## Why Two Stages?

The key insight is that **bad decisions can come from two different failures**:

1. **Perception failure**: The LLM misidentifies objects (e.g., doesn't see a stop sign)
2. **Strategy failure**: The LLM sees the right objects but applies the wrong rule (e.g., sees a stop sign but doesn't stop)

This framework separates these failures so you know **what to fix**:
- Low perception → improve vision capabilities or prompt for better object detection
- Low strategy → improve system prompt rules or reasoning instructions

## Quick Start

### 1. Prepare Your Dataset

Create a ground truth dataset (`dataset.json`):

```json
[
  {
    "sample_id": "test_001",
    "image_path": "images/test_001.jpg",
    "objects_present": ["stop_sign", "intersection", "crosswalk"],
    "expected_decision": "come_to_full_stop",
    "applicable_rules": ["stop_sign_obey", "crosswalk_pedestrian_check"],
    "notes": "Four-way intersection with stop sign"
  },
  ...
]
```

### 2. Collect LLM Responses

Run your LLM on each image with your system prompt. Collect responses into `responses.json`:

```json
[
  {
    "sample_id": "test_001",
    "raw_output": "I see a stop sign and intersection. I need to come to a full stop.",
    "identified_objects": ["stop_sign", "intersection"],
    "decision": "come_to_full_stop",
    "reasoning": "Stop sign requires full stop"
  },
  ...
]
```

**How to collect responses**: You can either:
- Parse the LLM's raw output automatically (extract objects and decisions)
- Use structured output (JSON mode) from the LLM
- Manually annotate for small datasets

### 3. Run Evaluation

```bash
python evaluate.py --dataset dataset.json --responses responses.json --report --output results.json
```

### 4. Read the Report

```
============================================================
  LLM STRATEGIC VISION EVALUATION
============================================================

Samples: 5/5
Perception (avg IoU):  0.8571
Strategy (accuracy):   0.8000
Combined score:        0.6857

--- Perception Metrics ---
  Object Precision:     0.9091
  Object Recall:        0.8571
  Object F1:            0.8824
  Exact Match Rate:     0.4000

  Per-Object Breakdown:
    clear_road               P=1.000  R=1.000
    construction_cone        P=1.000  R=1.000
    crosswalk                P=1.000  R=0.500
    ...

--- Strategy Metrics ---
  Decision Accuracy:    0.8000
  Rule Recall:          0.7500
  Rule Precision:       1.0000
  Hallucinated Rules:   0
  Missed Rules:         3

--- Error Breakdown ---
  perception_fail     : 1
  strategy_fail       : 1
  none                : 3

--- Top Failures ---

  [test_004] error=strategy_fail
    Objects:  GT=['crosswalk', 'pedestrian', 'traffic_light_green']
              Pred=['crosswalk', 'traffic_light_green']
    Decision: GT=yield_to_pedestrian
              Pred=proceed
```

## Understanding the Metrics

### Perception Metrics

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **Precision** | Of all objects the LLM claims to see, how many are real? | >0.9 |
| **Recall** | Of all objects that exist, how many did the LLM detect? | >0.85 |
| **F1** | Harmonic mean of precision and recall | >0.85 |
| **Exact Match** | % of images where object list is perfect | >0.7 |
| **IoU (per sample)** | Jaccard similarity of object sets | >0.7 |

### Strategy Metrics

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **Decision Accuracy** | % of correct final decisions | >0.9 |
| **Rule Recall** | % of rules that should fire, actually fired | >0.85 |
| **Rule Precision** | % of rules that fired, should have fired | >0.9 |
| **Hallucinated** | Rules the LLM invented that don't exist | =0 |
| **Missed** | Rules that should apply but were ignored | =0 |

### Combined Score

```
combined = (perception_iou ^ 0.4) * (strategy_correct ^ 0.6)
```

This is a **geometric mean** that gates strategy on perception. If perception is poor, the combined score drops even if the decision happens to be correct (which could be luck).

## Error Types

Each sample is classified into one error type:

| Error Type | Meaning | Fix |
|------------|---------|-----|
| `perception_fail` | LLM missed too many objects (IoU < 0.5) | Improve vision/prompt |
| `strategy_fail` | LLM saw objects but made wrong decision | Improve rules/prompt |
| `both` | Both perception and strategy failed | Fix both |
| `none` | Correct on both | - |

## Building a Good Evaluation Dataset

### Sample Diversity
Include scenarios that test different combinations:
- **Simple**: One object, one rule
- **Complex**: Multiple objects, competing rules
- **Edge cases**: Objects that look similar but require different actions
- **Ambiguous**: Cases where the correct decision depends on subtle details

### Rule Coverage
Ensure every rule in your system prompt is tested:
```python
# Check coverage
all_rules_in_dataset = set()
for sample in dataset:
    all_rules_in_dataset.update(sample.applicable_rules)

missing_rules = all_system_rules - all_rules_in_dataset
print(f"Untested rules: {missing_rules}")
```

### Minimum Dataset Size
- **Quick check**: 20-50 samples
- **Thorough evaluation**: 100-200 samples
- **Production-ready**: 500+ samples with statistical significance

## Advanced: Custom Scoring

### Weighted Scoring
Adjust weights based on your use case:

```python
# Safety-critical: strategy matters more
score_samples(samples, responses, perc_weight=0.2, strat_weight=0.8)

# Object detection benchmark: perception matters more
score_samples(samples, responses, perc_weight=0.7, strat_weight=0.3)
```

### Custom Object Matching
For fuzzy matching (e.g., "car" vs "parked_car"):

```python
def custom_jaccard(gt_objects, pred_objects):
    # Implement synonym matching, hierarchy, etc.
    matched = set()
    for pred in pred_objects:
        for gt in gt_objects:
            if are_similar(pred, gt):  # Your similarity function
                matched.add((pred, gt))
                break
    # Compute score from matches
    ...
```

## Integration with LLM APIs

### OpenAI Vision API Example

```python
import openai

def get_llm_response(image_path, system_prompt):
    with open(image_path, "rb") as f:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this image and make a decision."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
    return response.choices[0].message.content
```

### Structured Output (Recommended)

Ask the LLM to output JSON for easier parsing:

```
System prompt addition:
"Output your response as JSON:
{
  \"identified_objects\": [\"object1\", \"object2\"],
  \"decision\": \"your_decision\",
  \"reasoning\": \"why you made this decision\",
  \"rules_applied\": [\"rule1\", \"rule2\"]
}"
```

## Interpreting Results

### Scenario 1: High Perception, Low Strategy
```
Perception: 0.92
Strategy:   0.65
```
**Diagnosis**: The LLM sees objects correctly but doesn't apply rules well.
**Fix**: Rewrite system prompt rules. Make them more explicit and unambiguous. Add examples.

### Scenario 2: Low Perception, High Strategy
```
Perception: 0.45
Strategy:   0.85
```
**Diagnosis**: When the LLM sees objects, it makes good decisions. But it misses too many objects.
**Fix**: Use a better vision model. Add object detection instructions to the prompt. Consider a two-stage pipeline (detect objects first, then reason).

### Scenario 3: Both Low
```
Perception: 0.50
Strategy:   0.55
```
**Diagnosis**: Fundamental problems with both vision and reasoning.
**Fix**: Start with a simpler system prompt. Test with fewer rules. Improve both vision and reasoning separately.

## File Structure

```
evaluation_framework/
├── evaluate.py              # Main evaluation script
├── example_dataset.json     # Example ground truth dataset
├── example_responses.json   # Example LLM responses
├── README.md                # This file
└── images/                  # Your test images (not included)
    ├── test_001.jpg
    └── ...
```
