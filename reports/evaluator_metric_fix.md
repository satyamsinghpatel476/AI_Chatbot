# Evaluator Metric Fix

## Main fixes

- Removed the composite-final-score path from the research dashboard view.
- Added contamination-specific metric helpers in `evaluator/metrics.py`.
- Safe rejection is no longer counted as contamination when an answer says the relationship is unsupported, unrelated, not suitable, cannot replace robot sensing/control, or lacks robot data such as geometry, pose, motion, obstacle measurements, timestamps, or uncertainty.
- Contamination is reserved for unsupported direct replacement or direct capability claims, such as consumer apps replacing SLAM, LiDAR, localization, PID, perception, sensors, or controllers.
- Cross-domain robustness is now based on relationship correctness, required information, stated limitations, onboard safety preservation, and task fulfillment.
- Intent labels are normalized before summary and confusion-matrix generation.
- Missing classifier output remains `N/A`; System A is not penalized for not exposing an intent classifier.

## Verified example

Answer:

```text
Restaurant ratings do not provide geometry, pose, motion, or obstacle measurements and cannot replace onboard perception.
```

For the mixed-domain question about restaurant ratings and obstacle avoidance, the metric helper returns:

- contamination: `0`
- cross-domain robustness: `1.0`

## Why previous results looked too similar

The earlier scoring emphasized broad answer quality and a composite-style comparison. That made systems look close when they all produced plausible short answers. The new evaluation separates the properties the project is actually about: contamination prevention, false rejection, memory recall, learned knowledge, relationship handling, intent classification, and latency.
