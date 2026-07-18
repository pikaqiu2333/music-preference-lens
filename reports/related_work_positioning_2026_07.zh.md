# 相关研究与项目定位（2026-07）

## 最相关的一手工作

1. [Plan-and-Solve Prompting](https://aclanthology.org/2023.acl-long.147/)（ACL 2023）：先规划再执行可改善部分推理任务，但没有研究开放目录推荐、实体幻觉或解释忠实性。
2. [Order Matters in Hallucination](https://arxiv.org/abs/2408.05093)（2024/2025）：直接比较 answer-first 与 logic-first；与输出顺序实验最接近，但主要是选择题和 TruthfulQA。
3. [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388)（NeurIPS 2023）：模型可能受未说明偏置影响选择，再生成貌似独立的合理化理由。
4. [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)（2023）：通过截断、改写、错误和 filler 干预理由，测最终答案是否真的依赖理由。
5. [On Faithfulness and Coherence of Language Explanations for Recommendation Systems](https://arxiv.org/abs/2209.05409)（2022）：推荐解释可以流畅但脆弱，不能直接视为评分或推荐的真实依据。
6. [Discovering Latent Knowledge in Language Models Without Supervision](https://openreview.net/forum?id=ETKGuby0hcs)（ICLR 2023）：隐藏状态可能包含与最终文本不同的可恢复知识。
7. [How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/abs/2305.06569)（2023）：比较多种 Item ID，明确区分自然语言名称生成与合法物品集合选择。
8. [Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval](https://arxiv.org/abs/2503.24193)（2025）：直接从自然语言生成 track ID；未研究自由生成错误之前的内部关系信号。
9. [Music Recommendation with Large Language Models](https://arxiv.org/abs/2511.16478)（2025）：把幻觉、知识截止和非确定性列为 LLM 音乐推荐的核心风险，但属于综述与评测框架。
10. [Eliminating Out-of-Domain Recommendations in LLM-based Recommender Systems](https://aclanthology.org/2026.findings-acl.310/)（Findings ACL 2026）：RecLM 统一比较检索、受限生成和离散 item token，并在其实验中将 `OOD@10` 降为 0。
11. [Logit Space Constrained Fine-Tuning for Mitigating Hallucinations in LLM-Based Recommender Systems](https://aclanthology.org/2025.emnlp-main.1491/)（EMNLP 2025）：通过 logit-space 约束微调降低推荐幻觉，没有分析单次错误的内部机制。
12. [Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations](https://aclanthology.org/2024.findings-emnlp.466/)（Findings EMNLP 2024）：在 subject–relation 查询中区分低层 MLP 的知识不足和高层 attention 的答案提取失败。
13. [FactCheckmate](https://aclanthology.org/2025.findings-emnlp.663/)（Findings EMNLP 2025）：从解码前隐藏状态预测通用事实幻觉，并通过表示干预提高事实性；任务主要是 QA，不是开放式推荐实体关系。
14. [HIDE and Seek](https://arxiv.org/abs/2506.17748)（2025）：用输入与输出隐藏表示的解耦程度做单次生成幻觉检测；同样没有处理音乐目录中的 title-artist 关系。
15. [Do Language Models Know When They’re Hallucinating References?](https://aclanthology.org/2024.findings-eacl.62/)（Findings EACL 2024）：用一致性查询区分真实与虚构引用，提供了与歌名—歌手相似的“易核验模型生物”。
16. [MechELK](https://arxiv.org/abs/2605.28825)（2026）：把 SAE、激活 patching、因果探针和表示工程组合成 latent knowledge 的定位、验证与提取流程。
17. [Relational Linearity is a Predictor of Hallucinations](https://arxiv.org/abs/2601.11429)（2026）：报告关系线性与合成实体幻觉率的强相关，提示关系表示方式会影响模型自我评估知识的难度。
18. [The Self-Correction Illusion](https://arxiv.org/abs/2606.05976)（2026）：保持错误内容不变，仅改变 chat role 就能大幅改变纠错行为，说明“模型改口”不能直接视为“模型原先意识到错误”。
19. [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)（2023）：在没有外部反馈时，内在自纠错常常无效甚至退化。
20. [When Can LLMs Actually Correct Their Own Mistakes?](https://aclanthology.org/2024.tacl-1.78/)（TACL 2024）：批判性综述显示可靠外部反馈和天然可验证任务更容易产生稳定纠错。
21. [FaithLM](https://aclanthology.org/2026.eacl-long.177/)（EACL 2026）：把解释忠实性定义为反事实干预性质，而不是语言是否流畅。
22. [Towards Faithful Natural Language Explanations](https://aclanthology.org/2025.emnlp-main.529/)（EMNLP 2025）：用 activation patching 比较解释和答案的因果归因，为理由机制实验提供直接方法参照。
23. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)（2026）：J-space 选择性参与灵活推理，常规自动处理可能绕过；因此歌名-歌手检索不一定天然适合用 J-lens 捕捉。
24. [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213)（NeurIPS 2023）：因果定位结果不必然预测最佳知识编辑位置，提醒我们不能把“某层有信号”直接写成“该层可修复错误”。
25. [GrowLoop](https://arxiv.org/abs/2605.28882)（2026）：用少量人工种子让 agent 持续演化主观对话评价 rubric；它适合处理“推荐理由是否令人信服”这类主观评价，但不能替代 title-artist 的外部目录事实核验。
26. [On Large Language Models' Hallucination with Regard to Known Facts](https://aclanthology.org/2024.naacl-long.60/)（NAACL 2024）：对同一事实三元组构造答对/答错提示配对，并用跨层输出动态预测幻觉；它已经覆盖“模型有可恢复知识仍答错”，但检测回溯已生成答案 token，且没有实体 disjoint 测试。
27. [Two Pathways to Truthfulness](https://aclanthology.org/2026.acl-long.1173/)（ACL 2026）：把事实性信号拆成 question-anchored 与 answer-anchored 路径并做 attention knockout/token patching；检测使用完整问答，因此尚未覆盖 title 完成、artist 尚未生成的关系完成前时刻。
28. [Are the Hidden States Hiding Something?](https://aclanthology.org/2025.acl-long.304/)（ACL 2025）：显示合成真假模板上的 probe 可能无法迁移到模型自然生成输出，是我们必须使用自然冲突、严格 holdout 和表面基线的关键反证。
29. [Simple Factuality Probes](https://aclanthology.org/2025.findings-emnlp.880/)（Findings EMNLP 2025）：把隐藏状态事实性探针扩展到长文本生成，但没有分析目录关系的中间完成时刻。
30. [INSIDE](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)（ICLR 2024）：说明内部状态可用于低监督幻觉检测，但可解码性仍不等于模型会使用该信号。
31. [WeMusic-Agent](https://arxiv.org/abs/2512.16108)（2025 预印本）：直接生成 `song_name` 与 `singer_name`，使用目录事实奖励和工具调用边界学习；与业务场景很接近，但没有公开内部关系机制。
32. [MuChator](https://arxiv.org/abs/2605.27103)（2026 预印本）：将目录事实性纳入 hybrid reward 并报告工业部署收益；闭源数据和作者报告的线上结果不能独立复核。
33. [Read Between the Tracks](https://aclanthology.org/2026.nlp4musa-1.7/)（NLP4MusA 2026）：即使给定歌曲候选，开放文本输出仍可能越出候选集合，说明 prompt 列表不等于解码约束。
34. [ERBench](https://papers.nips.cc/paper_files/paper/2024/hash/5ef9853a6cdea40ae3e301a6d8dc32b5-Abstract-Datasets_and_Benchmarks_Track.html)（NeurIPS 2024）：已用关系数据库约束自动验证实体关系并包含 Music DB，因此目录关系自动核验本身不是本项目的新意。
35. [How Do LLMs Understand Relevance?](https://doi.org/10.1145/3774942)（ACM TOIS 2026）：已把 activation patching 用到检索相关性机制；我们的范围必须收窄到生成式 title-artist 关系错误，而不是笼统的推荐机制可解释性。
36. [Towards Trustworthy LLM-Based Recommendation via Rationale Integration](https://arxiv.org/abs/2601.02364)（2026）：支持 rationale-first 可改善候选集推荐，但理由是在已知目标 item 后生成，未证明反事实忠实性。
37. [Implicit Reasoning for LLM-based Generative Recommendation](https://arxiv.org/abs/2606.14142)（2026 预印本）：显式理由对删除和噪声敏感，并可能与 semantic ID 决策不一致，是 reason-first 必然有益或忠实的直接反证。

本轮 AutoResearchClaw 运行、召回缺陷和更完整的证据矩阵见[文献调研审计](autoresearchclaw_literature_audit_2026_07.zh.md)。
使用 Academic Research Suite 重做的结构化综述、纳入/排除协议和反方审查见[定向结构化文献综述](academic_research_suite_literature_review_2026_07.zh.md)。

## 与现有工作的重叠

已有研究分别覆盖：

- 先规划可能改善推理；
- 输出顺序可能改变可靠性；
- 理由可能事后合理化；
- 推荐解释可能不忠实；
- 检索、受限生成和 Item ID 已可在给定目录上严格消除 OOD 输出；
- subject–relation 幻觉可能来自知识不足或答案提取失败；
- 隐藏状态可以支持通用幻觉检测或知识提取；
- 纠错行为会受到反馈来源和 chat role 影响；
- 机制定位不自动等于有效干预。

因此，我们不能把“reason-first”“推荐解释不忠实”“Song ID 减少实体幻觉”或“隐藏状态能分类事实错误”单独当成新贡献。

## 可能的新意

当前项目把开放式音乐推荐拆成四个独立可验证层次：

1. 目录合法性：完整 title-artist 是否是 catalog 中的真实关系。
2. 内部预警：错误 pair 完整输出前是否已有跨实体可泛化的关系有效性信号。
3. 因果类型：错误来自知识缺失、关系绑定失败，还是有正确候选但对象提取失败。
4. 解释忠实性：理由是否基于可核验属性，并且是否因果影响实体选择。

最值得继续验证的主假设是：

> 在开放式音乐推荐中，模型生成目录不支持的 title-artist 关系之前，隐藏状态可能已经包含可预测的关系错误信号；但只有 held-out 泛化和定向因果干预才能把该信号解释为模型可用的 latent relation knowledge。

理由顺序保留为次级交互假设：

> rationale-first 在熟悉、可满足的场景中可能像语义查询扩展；在可行集合很窄或模型知识不足时，理由会形成提前承诺，随后诱导占位符或概念型虚构实体。

本轮检索尚未找到同时验证“自由开放音乐推荐、完整 title-artist 关系、title 后且 artist 前的隐藏状态、实体/提示 disjoint 泛化和选择性因果干预”的工作。这是当前最窄、最可辩护的研究空缺；不能写成未经系统综述支持的绝对“首次”。Jiang et al. 已研究同一已知事实答对/答错的跨层动态，Two Pathways 已研究完整回答中的两条事实性路径，ERBench 已做关系自动核验，Text2Tracks/RecLM/WeMusic-Agent/MuChator 已提供强目录约束方案。因此当前新意只能来自**自然关系完成时序 + 严格泛化 + 因果区分**的组合，而不是其中任一单点。

Phase 2 又进一步缩小了当前贡献。我们已经完成从自由生成、双目录证据、离线重放到预注册机制准入门槛的流程，但 80 个歌单最终只有 7 个唯一严格冲突，低于预注册的 30 个。因此这一阶段是**可复现的样本产出边界与负结果**，不是新的机制检测器。后续若要形成更强的论文主张，需要重新注册“发现集扩大候选池 + 全新 held-out 确认集”的设计，不能在这 7 个案例上继续补实验后声称确认。

## 最小可发表问题

### 1. 完整关系的错误信号能否跨实体泛化？

继续让模型自由生成新歌单，用独立目录标注完整 pair；先取得至少 30 个唯一 strict conflict title cluster 及匹配 exact controls，再在 artist-disjoint、title-disjoint 和 prompt-disjoint holdout 上比较隐藏状态与 logit、entropy、流行度、词长和词法向量基线。

### 2. 错误属于缺知识、绑定失败还是提取失败？

对 title 和 artist 都存在但 binding 不成立的案例，测试正确关系能否由候选打分恢复，再用预注册 activation patching、随机层和标签打乱对照检验因果作用。

### 3. 推荐理由是否因果忠实？

固定实体，删除、替换、反转或交换理由，再测完整 pair 与自由选择；同时比较 reason-first、item-first 和 no-reason。Song ID、retrieve-then-rank 和受限生成只作为产品可靠性强基线。

## 表述边界

- 报告 pilot 的原始计数，不称 reason-first 显著降低幻觉。
- 严格无人声两组 0/10 是地板效应，不能比较顺序优劣。
- 五个占位符是待复现失败模式，不足以单独建立因果。
- Space Oddity 支持跨提示不一致与错误理由，不证明模型主观知道或故意撒谎。
- 没有反事实干预时，只称理由与推荐表面一致。
- Song ID 保证输出属于给定目录，不保证属性、相关性或解释正确。
- title 已经生成后的测量只能称 `pre-artist` 或 `pre-relation-completion`，不能称全局 `pre-output`。
- 线性可解码性只称“关系有效性信号”，不能称意识、自知或欺骗。
