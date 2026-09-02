# Literature Grounding

## Project lens used for ranking

This repository implements an app-agnostic Android exploratory-testing agent. It ingests requirements, Figma data, defects, and runtime UI observations into a Neo4j GraphRAG system; plans one test at a time; executes it on a device; builds a persistent UI-state and navigation graph; and uses verdicts, coverage gaps, failures, and strategy memory to choose the next test. Papers are therefore ranked by direct overlap with: (1) Android GUI testing, (2) autonomous exploration and replanning, (3) graph or retrieval memory, (4) reliable evaluation, and (5) visual grounding.

The closest research gap is the combination of an a-priori specification graph (SRS/design) with an a-posteriori runtime UI graph for defect-oriented exploration. None of the main-track papers below combines all three of the repository's central ideas: requirement grounding, a learned runtime app model, and verdict-driven generation of the next exploratory test.

## Papers ranked by relevance

### 1. [VLM-Fuzz: Vision Language Model Assisted Recursive Depth-First Search Exploration for Effective GUI Testing of Android Apps](https://link.springer.com/article/10.1007/s10664-026-10816-4)

**Published in:** *Empirical Software Engineering*, volume 31, article 76, 2026. The journal is listed as Q1 in the Software category in the [SCImago journal ranking](https://www.scimagojr.com/journalrank.php?category=1712&country=&ord=desc&order=rpd).

**Short review:**

- VLM-Fuzz is a fully automated Android GUI-testing approach built around recursive depth-first exploration of GUI states.
- It combines fast heuristics, Android Manifest analysis, runtime UI hierarchy data, and on-demand VLM reasoning for visually complex screens.
- On 59 benchmark apps it improves class, method, and line coverage over the strongest baseline by 9.0%, 3.7%, and 2.1%, respectively.
- The journal version also evaluates 80 recent Google Play apps and reports 52 unique crashes across 12 apps, giving it unusually strong real-world evidence.

**Relevance to this project:** This is the closest paper overall because both systems autonomously explore Android GUI states and try to avoid unproductive repetition. VLM-Fuzz provides a strong comparison baseline for this repository's Live App Model, state identity, navigation memory, and livelock handling. The repository's main distinction is specification-grounded test generation and cross-run verdict/coverage learning, which VLM-Fuzz does not provide.

### 2. [GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_GUI-Xplore_Empowering_Generalizable_GUI_Agents_with_One_Exploration_CVPR_2025_paper.html)

**Published in:** CVPR 2025 main conference, pages 19477-19486.

**Short review:**

- GUI-Xplore studies how an agent can explore an unfamiliar application once and reuse the acquired knowledge across later tasks.
- Its dataset supplies exploration videos and five hierarchical tasks spanning action recall, page analysis, app usage, and application-level understanding.
- Xplore-Agent combines action-aware GUI modeling with graph-guided environment reasoning rather than treating each screenshot independently.
- The paper reports a 10% improvement over prior methods in unfamiliar environments, while acknowledging substantial remaining headroom.

**Relevance to this project:** This is the closest main-track paper to the repository's persistent UIState graph and reusable NavTree. Its graph-guided reasoning strongly supports the design choice to preserve an app model across campaigns. Unlike this project, it targets general task execution rather than requirement coverage, exploratory test selection, and defect discovery.

### 3. [MobileUse: A Hierarchical Reflection-Driven GUI Agent for Autonomous Mobile Operation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3994410d63ec68ce9a66011a34c9a2c4-Abstract-Conference.html)

**Published in:** NeurIPS 2025 main conference track.

**Short review:**

- MobileUse is an autonomous mobile GUI agent designed for long-horizon execution, error recovery, and cold-start operation in unfamiliar apps.
- Its hierarchical reflection architecture monitors individual actions, intermediate progress, and overall task completion at different time scales.
- Reflection-on-Demand limits unnecessary reasoning cost, while proactive exploration builds environment knowledge before difficult execution.
- It reports task success rates of 62.9% on AndroidWorld and 44.2% on AndroidLab and releases a physical-device toolkit.

**Relevance to this project:** MobileUse closely matches the repository's planner-executor loop, recovery logic, and need to improve autonomy on multi-step Android workflows. Its temporal reflection design is a strong model for separating action failure, route failure, and test-objective failure. This project adds SRS/Figma grounding, defect-focused objectives, and persistent graph memory across tests.

### 4. [Retrieval-augmented GUI Agents with Generative Guidelines](https://aclanthology.org/2025.emnlp-main.902/)

**Published in:** EMNLP 2025 main conference, pages 17866-17875.

**Short review:**

- RAG-GUI retrieves web tutorials at inference time and converts them into guidelines for a VLM-based GUI agent.
- It is designed as a model-agnostic plug-in rather than a replacement for the underlying perception and action model.
- Training combines supervised fine-tuning with self-guided rejection-sampling fine-tuning to improve the usefulness of retrieved knowledge.
- Across three tasks and two model sizes, it improves over inference baselines by 2.6% to 13.3%.

**Relevance to this project:** This paper is the clearest external support for retrieval-augmented GUI control. The repository goes further by retrieving structured requirements, validation rules, design elements, live UI states, defects, and previous outcomes from a graph. A useful research comparison would isolate plain prompting, text RAG, and the repository's graph-native multi-source retrieval.

### 5. [AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents](https://openreview.net/forum?id=il5yUQsrjC)

**Published in:** ICLR 2025 conference.

**Short review:**

- AndroidWorld provides 116 programmatic tasks across 20 real Android apps in a reproducible emulator environment.
- Tasks are dynamically parameterized, so evaluation is not limited to memorized wording or a fixed set of static demonstrations.
- Each task defines initialization, success checking, and teardown by inspecting durable Android system or application state.
- The best baseline in the paper completes 30.6% of tasks, and robustness tests show that small task variations materially affect results.

**Relevance to this project:** AndroidWorld directly addresses the repository's stated lack of ground truth. Its programmatic state checks are a better oracle than asking the executor or an LLM to judge its own result. Adapting this approach would make autonomy, bug discovery, requirement coverage, and recovery improvements experimentally defensible.

### 6. [DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1704ddd0bb89f159dfe609b32c889995-Abstract-Conference.html)

**Published in:** NeurIPS 2024 main conference track.

**Short review:**

- DigiRL argues that static demonstrations do not capture the stochastic and dynamic behavior encountered while controlling real GUIs.
- It constructs a scalable Android learning environment with a VLM-based evaluator and applies offline followed by offline-to-online reinforcement learning.
- The method improves its 1.5B model from 17.7% to 67.2% success on Android-in-the-Wild without additional human demonstrations.
- Its results show that interactive outcome feedback can matter far more than imitation of fixed trajectories alone.

**Relevance to this project:** DigiRL supports the repository's central idea that execution outcomes should change future behavior. The current repository performs adaptation through GraphRAG memory, strategy scores, coverage, and prompts rather than updating model weights. DigiRL is therefore both a conceptual baseline and a possible future policy-learning extension.

### 7. [UI-Hawk: Unleashing the Screen Stream Understanding for Mobile GUI Agents](https://aclanthology.org/2025.emnlp-main.920/)

**Published in:** EMNLP 2025 main conference, pages 18217-18236.

**Short review:**

- UI-Hawk identifies a weakness in agents that see only the current screenshot and a plain-text action history.
- It adds a history-aware visual encoder that processes the sequence of screens encountered during mobile navigation.
- Training progresses through GUI grounding, referring, screen question answering, and screen summarization before advanced stream understanding.
- The accompanying FunUI benchmark and navigation experiments show that visual screen history improves GUI task performance.

**Relevance to this project:** The repository already stores screenshots, UIState transitions, ordered execution paths, and action outcomes, making UI-Hawk highly applicable. Its findings support presenting selected visual history rather than only textual route summaries to the planner or executor. The main research question here is how to compress that history without exceeding the prompt budget.

### 8. [GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning](https://openreview.net/pdf?id=XVm8KOO3Ri)

**Published in:** NeurIPS 2025 main conference track.

**Short review:**

- GUI Exploration Lab is a configurable simulator whose screens, icons, and inter-screen navigation graphs are fully observable for training and evaluation.
- It separates memorizing basic GUI knowledge, generalizing single actions, and learning exploration through interactive multi-turn trial and error.
- Experiments use a staged pipeline of supervised fine-tuning, single-turn reinforcement learning, and multi-turn reinforcement learning.
- The work provides controlled evidence that multi-turn training develops stronger navigation and exploration strategies.

**Relevance to this project:** This is a useful blueprint for upgrading the repository's simulator into a controlled evaluation environment with known navigation graphs. It also offers a principled path beyond heuristic coverage directives. The repository would still need testing-specific rewards such as defect exposure, requirement coverage, and route cost rather than task success alone.

### 9. [Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](https://proceedings.mlr.press/v267/xu25ae.html)

**Published in:** ICML 2025, Proceedings of Machine Learning Research 267, pages 69772-69805.

**Short review:**

- Aguvis is a screenshot-only autonomous GUI framework with a standardized cross-platform action space and explicit structured reasoning.
- Its two-stage training pipeline deliberately separates basic GUI grounding from higher-level planning and reasoning.
- The dataset includes multimodal grounding and reasoning annotations for web, desktop, and mobile interaction.
- It reports strong offline and real-world online results without requiring a closed-source model to perform the interaction loop.

**Relevance to this project:** Aguvis offers a clean perception-planning separation that mirrors this repository's high-level planner and low-level device executor. It could be evaluated as an alternative executor when accessibility trees are sparse or unreliable. It does not provide testing objectives, requirement oracles, or the repository's persistent knowledge and verdict loop.

### 10. [AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://aclanthology.org/2024.acl-long.850/)

**Published in:** ACL 2024 main conference, Volume 1: Long Papers, pages 16022-16076; Best Resource Paper Award.

**Short review:**

- AppWorld provides a controllable environment of nine realistic applications, 457 APIs, and 750 complex agent tasks.
- Tasks require iterative control flow and interaction rather than a short, predetermined sequence of calls.
- Evaluation uses database-state unit tests, accepting different valid solution paths while detecting unintended collateral changes.
- The strongest reported model solves about 49% of normal tasks and 30% of challenge tasks, leaving a demanding benchmark.

**Relevance to this project:** AppWorld is not a GUI-testing system, but its oracle design is directly valuable. The repository should similarly judge final app state and unintended side effects independently of the agent's narrative verdict. This is especially relevant for destructive actions, role boundaries, and tests that appear to pass visually while corrupting hidden state.

### 11. [OS-ATLAS: A Foundation Action Model for Generalist GUI Agents](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0faa4bc5f522076947a030273629d4fe-Abstract-Conference.html)

**Published in:** ICLR 2025 conference.

**Short review:**

- OS-ATLAS develops an open foundation action model for GUI grounding and out-of-distribution interaction.
- Its data-generation toolkit spans Android, web, Windows, Linux, and macOS and produces a corpus of more than 13 million GUI elements.
- The model is evaluated across mobile, web, and desktop benchmarks, emphasizing transfer to unseen interfaces and resolutions.
- It can also serve as a grounding module beneath a separate planner, keeping perception and decision making modular.

**Relevance to this project:** OS-ATLAS is most relevant to the executor's element-selection reliability and the project's app-agnostic goal. It could complement accessibility-tree actions with visual grounding when resource IDs or semantic nodes are absent. It does not address exploratory-test choice, defect attribution, or requirement coverage.

### 12. [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://aclanthology.org/2024.acl-long.505/)

**Published in:** ACL 2024 main conference, Volume 1: Long Papers, pages 9313-9332.

**Short review:**

- SeeClick is a screenshot-only GUI agent centered on locating interface elements from natural-language instructions.
- It introduces automated GUI-grounding data construction and grounding-focused pre-training for mobile, desktop, and web screens.
- The paper also introduces ScreenSpot, a realistic benchmark for instruction-to-screen-element grounding.
- Results across three downstream benchmarks show a strong relationship between grounding accuracy and complete agent performance.

**Relevance to this project:** SeeClick targets the low-level action failures that currently reduce autonomy when controls are hard to identify. It is a relevant executor baseline and supports explicit grounding metrics in addition to end-to-end test success. Its scope ends before test planning, graph memory, or verdict-driven exploration.

### 13. [Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs](https://eccv.ecva.net/virtual/2024/poster/749)

**Published in:** ECCV 2024 main conference.

**Short review:**

- Ferret-UI specializes a multimodal language model for mobile interface understanding, referring, grounding, and reasoning.
- Training covers elementary tasks such as OCR, icon recognition, widget classification, and element finding, followed by advanced UI reasoning.
- Its any-resolution design accommodates the tall and varied aspect ratios common to Android and iPhone screens.
- A 14-task benchmark shows strong gains over open models and competitive performance with GPT-4V on mobile UI understanding.

**Relevance to this project:** Ferret-UI is an enabling perception model for screens whose accessibility hierarchy is thin, malformed, or missing. It could improve state labeling, element descriptions, and screenshot-based recovery. It is not an autonomous testing framework and has no coverage, memory, or defect-learning mechanism.

### 14. [ShowUI: One Vision-Language-Action Model for GUI Visual Agent](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.html)

**Published in:** CVPR 2025 main conference, pages 19498-19508.

**Short review:**

- ShowUI is a compact 2B vision-language-action model for grounding and navigation across web, mobile, and online environments.
- It models a screenshot as a UI-connected graph and uses that structure to discard redundant visual tokens.
- Interleaved vision-language-action streaming represents multi-turn action history inside a unified model interface.
- The paper reports 75.1% zero-shot grounding accuracy, 33% fewer visual tokens, and a 1.4x training speedup.

**Relevance to this project:** ShowUI is relevant where device latency and prompt size constrain repeated screenshot reasoning. Its UI-graph token selection and interleaved history could make visual recovery more efficient. The model solves perception and action generation rather than exploratory testing or GraphRAG-based prioritization.

### 15. [OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use](https://aclanthology.org/2025.acl-long.369/)

**Published in:** ACL 2025 main conference, Volume 1: Long Papers, pages 7436-7465.

**Short review:**

- This survey unifies computer, phone, and browser agents under an operating-system-agent framework.
- It organizes the field around environment interfaces, observations, actions, understanding, planning, grounding, memory, and self-evolution.
- It compares both domain-specific foundation models and modular agent frameworks rather than treating GUI agents as one homogeneous method.
- Its benchmark and evaluation taxonomy is a useful map of the rapidly changing literature and open reliability problems.

**Relevance to this project:** This is the best recent main-track survey for positioning the architecture as a mobile OS agent with explicit testing responsibilities. It is less directly actionable than the empirical papers above, but useful for terminology, related-work organization, and identifying standard benchmarks. The repository's distinctive contribution should be framed as requirement-grounded exploratory testing, not general phone automation.

## Synthesis for this project

The literature supports the repository's major architectural choices but also clarifies the strongest research claim. `VLM-Fuzz` is the closest Android-testing baseline; `GUI-Xplore` and `UI-Hawk` support persistent environment and history modeling; `MobileUse` supports hierarchical recovery; `RAG-GUI` supports external knowledge retrieval; and `DigiRL` supports learning from interactive outcomes. `AndroidWorld` and `AppWorld` show how to replace self-reported verdicts with reproducible state-based oracles.

A defensible novelty statement is therefore: **requirement-grounded agentic exploratory testing that steers a runtime UI-state graph using gaps between the specification graph and observed application behavior**. To validate that statement, the most important experiment is an ablation with four arms: executor-only, runtime graph only, specification GraphRAG only, and the combined dual-graph agent. Each arm should use the same device-step budget and be compared on independently checked defect discovery, requirement coverage, screen coverage, autonomy, and unintended side effects.
