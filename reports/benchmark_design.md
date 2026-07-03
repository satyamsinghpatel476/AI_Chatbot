# Benchmark Design

## Suite sizes

The benchmark generator now writes both the existing filenames and requested `_suite.json` aliases:

- `context_contamination.json` and `context_contamination_suite.json`: 100 cases
- `memory_recall.json` and `memory_recall_suite.json`: 50 save/recall pairs
- `knowledge_growth.json` and `knowledge_growth_suite.json`: 50 teach/recall/paraphrase tests
- `intent_classification.json` and `intent_suite.json`: 120 balanced intent cases
- `cross_domain_robustness.json` and `cross_domain_robustness_suite.json`: 100 relationship cases

## Contamination focus

The context suite now includes:

- incompatible consumer-app and robotics claims
- valid indirect relationships
- conditional relationships
- misleading shared words such as navigation, mapping, tracking, and control
- context switching between robotics and daily-life assistance
- adversarial prompts that ask the assistant to accept a false premise

## Memory and knowledge

Memory recall uses supported personal fact forms such as name, city, favorite app, favorite robot, study area, preferred language, and operating system.

Knowledge growth uses project-specific taught concepts. The unrelated-query probe avoids containing the exact taught concept name, so a correct recall system is not punished for matching the original concept.

## Intent

The intent suite is balanced across:

- robotics
- daily
- personal
- mixed
- general
- unknown

System A reports intent metrics as `N/A` because it does not expose classifier output.

## Ablation

The ablation suite is intentionally harder and includes:

- Uber GPS replacing SLAM
- restaurant ratings improving obstacle avoidance
- social media replacing perception
- food-delivery routes training robot navigation
- Google Maps replacing LiDAR
- ride-hailing drivers improving path planning
- app permissions versus ROS permissions
- ambiguous phone navigation versus robot navigation

The ablation summary reports contamination rate, false rejection rate, cross-domain robustness, latency, memory recall, and knowledge growth.
