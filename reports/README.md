# A collection of reports from test runs

# Test 1
Test the LLM's strategy decision making for one-shot scenarios.

note: The simplets test. Doesn't this work the more complex will not work.

- The chat history is wiped between each LLM prompt call.

## Dataset `25_common_events.json`
llmfun: git commit f9bd2a1 2026-06-03

- The test data contains more information in the prompt than it should not have compared to in a real case scenario such as:
    - ok: "Analyze this image from a border surveillance camera."
    - extra information in prompt: "A single person is walking near the border marker in a non-threatening posture."

## Result
```
Running evaluation on 25 test cases...
  [1/25] Evaluating test_001...
      Expected: []
      Actual:   []
  [2/25] Evaluating test_002...
      Expected: []
      Actual:   []
  [3/25] Evaluating test_003...
      Expected: []
      Actual:   []
  [4/25] Evaluating test_004...
      Expected: []
      Actual:   []
  [5/25] Evaluating test_005...
      Expected: []
      Actual:   []
  [6/25] Evaluating test_006...
      Expected: ['trackerLight']
      Actual:   []
  [7/25] Evaluating test_007...
      Expected: ['trackerLight']
      Actual:   []
  [8/25] Evaluating test_008...
      Expected: ['trackerLight']
      Actual:   []
  [9/25] Evaluating test_009...
      Expected: ['trackerLight']
      Actual:   []
  [10/25] Evaluating test_010...
      Expected: ['trackerLight']
      Actual:   ['trackerLight', 'audibleWarning']
  [11/25] Evaluating test_011...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   []
  [12/25] Evaluating test_012...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   []
  [13/25] Evaluating test_013...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   ['trackerLight', 'audibleWarning', 'strobeLight']
  [14/25] Evaluating test_014...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   []
  [15/25] Evaluating test_015...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   ['trackerLight', 'audibleWarning', 'strobeLight', 'humanOperator', 'callPatrol']
  [16/25] Evaluating test_016...
      Expected: []
      Actual:   []
  [17/25] Evaluating test_017...
      Expected: []
      Actual:   []
  [18/25] Evaluating test_018...
      Expected: []
      Actual:   []
  [19/25] Evaluating test_019...
      Expected: []
      Actual:   []
  [20/25] Evaluating test_020...
      Expected: []
      Actual:   []
  [21/25] Evaluating test_021...
      Expected: ['audibleWarning']
      Actual:   []
  [22/25] Evaluating test_022...
      Expected: ['strobeLight']
      Actual:   []
  [23/25] Evaluating test_023...
      Expected: ['strobeLight', 'humanOperator']
      Actual:   []
  [24/25] Evaluating test_024...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   []
  [25/25] Evaluating test_025...
      Expected: ['audibleWarning', 'strobeLight']
      Actual:   []

==================================================
EVALUATION RESULTS (MVP)
==================================================
Total test cases: 25
Passed: 11
Failed: 14
Composite score: 0.510

Per-category scores:
  no_threat: 100.00% (5/5)
  single_low_threat: 29.80% (0/5)
  single_high_threat: 22.64% (1/5)
  animal_false_positive: 100.00% (5/5)
  persistent_threat: 2.40% (0/5)
```

# Test 2
Test the LLM's strategy decision making as it would occur on the border by keeping the history and evaluate its reasoning when threat level increase because of persistency.

- The chat history is kept between each LLM prompt call.
- Will trigger chat history compression

Chat history expectations.

- event 11. High threat detected
- event 12. Now attempting to breach border.
No threat escalation because highest is used but it should persist them and reason about the escalation because of persistency.

- event 15. High threat
- event 16. Animal, no threat
Expecting de-escalation and no action.


## Dataset `persistent_events.json`
llmfun: c9511db 2026-06-03

- The test data contains more information in the prompt than it should do in a real case scenario such as: Analyze this image from a border surveillance camera. A single person is walking near the border marker in a non-threatening posture.

The history of events is a stream of events following this pattern:
1. no threat x 2
2. low threat. Single standing
3. low threat. Single traveler
4. medium. Present, not doing anything.
5. high threat. Trying to breach barrier

- no threat, animal
- persistent threat x 2
- no threat, animal
- no threat, animal
- persistent threat x 2

## Result
```
Running evaluation on 13 test cases...
  [1/13] Evaluating test_001...
      Expected: []
      Actual:   []
  [2/13] Evaluating test_002...
      Expected: []
      Actual:   ['trackerLight']
  [3/13] Evaluating test_009...
      Expected: ['trackerLight']
      Actual:   []
  [4/13] Evaluating test_010...
      Expected: ['trackerLight']
      Actual:   []
  [5/13] Evaluating test_011_medium...
      Expected: ['trackerLight', 'strobeLight']
      Actual:   ['trackerLight', 'audibleWarning', 'strobeLight']
  [6/13] Evaluating test_011_high...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   ['trackerLight', 'audibleWarning', 'strobeLight', 'callPatrol']
  [7/13] Evaluating test_016...
      Expected: []
      Actual:   []
  [8/13] Evaluating test_017...
      Expected: []
      Actual:   []
  [9/13] Evaluating test_013_1...
      Expected: ['trackerLight']
      Actual:   []
  [10/13] Evaluating test_013_2...
      Expected: ['strobeLight']
      Actual:   ['trackerLight', 'audibleWarning', 'callPatrol']
  [11/13] Evaluating test_013_2...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   ['strobeLight', 'humanOperator']
  [12/13] Evaluating test_013_3...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   ['trackerLight', 'audibleWarning', 'strobeLight', 'callPatrol', 'humanOperator']
  [13/13] Evaluating test_015...
      Expected: ['humanOperator', 'callPatrol']
      Actual:   []

==================================================
EVALUATION RESULTS (MVP)
==================================================
Total test cases: 13
Passed: 5
Failed: 8
Composite score: 0.535

Per-category scores:
  no_threat: 64.25% (1/2)
  single_low_threat: 21.00% (0/2)
  single_high_threat: 46.50% (2/7)
  animal_false_positive: 100.00% (2/2)

Results saved to: reports/test_2_result.json
```
