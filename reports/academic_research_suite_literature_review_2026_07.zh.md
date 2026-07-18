# Academic Research Suite 定向结构化文献综述（2026-07-14）

## 摘要

本轮使用 `academic-research-skills-codex` 的 `deep-research / lit-review` 工作流，重新审查“开放式音乐推荐中的 title-artist 幻觉是否可在模型内部提前识别并因果干预”。检索和全文核验后的结论比此前更窄，也更可发表：

1. **目录合法性不是本项目的新问题。** Track ID、检索、受限生成、item token、工具调用和目录事实奖励已经能大幅降低甚至在特定基准中消除目录外输出。
2. **通用事实幻觉的隐藏状态检测也不是新问题。** 已有工作能从输入或答案隐藏状态预测幻觉，并区分知识写入不足与答案提取失败。
3. **仍未发现直接覆盖本项目完整交叉点的工作：**自然开放式音乐推荐、完整 title-artist 关系、title 已生成但 artist 尚未生成的中间时刻、跨 title/artist/prompt 的严格泛化，以及选择性因果干预。
4. 因而最稳妥的科学问题不是“模型是否意识到自己在撒谎”，而是：

> 在自由生成音乐推荐时，模型写完 title、尚未写 artist 的隐藏状态中，是否存在跨实体泛化的关系有效性信号；该信号反映知识缺失、关系绑定失败还是答案提取失败，并且能否被选择性因果干预？

当前 80 个歌单只得到 7 个唯一严格关系冲突，不足以支撑确认性机制结论。下一步首先是冻结标注规则并扩大自然错误发现集；达到至少 30 个严格冲突 title cluster 后，才进入预注册的 held-out probe，probe 过门槛后才做干预。

## 1. 综述协议

执行环境为本机安装的 `academic-research-skills-codex` 0.1.18（内含 Academic Research Suite upstream v3.16.0），采用其 `deep-research / lit-review`、来源核验、证据矩阵、综合与 Devil's Advocate 路径。

### 1.1 研究问题

本轮问题按 PICOC 式要素拆解：

| 要素 | 定义 |
|---|---|
| 对象 | 能自由生成自然语言音乐推荐的 decoder-only LLM |
| 现象 | 输出的 title、artist 各自可能真实，但完整关系不被目录支持 |
| 关键时刻 | title 完成后、artist 首 token 生成前 |
| 比较 | exact binding、strict binding conflict、nonexistent title、表面特征基线 |
| 结果 | held-out AUROC/AUPRC、关系 margin、定向 patch 效果和副作用 |
| 场景 | 开放式多歌曲推荐，不给模型候选列表或 Song ID |

### 1.2 操作性定义

- **关系有效性：**标准化后的 `(title, artist)` 由冻结目录快照和独立证据支持；title-only 或 artist-only 命中不算完整正确。
- **模型“知道”：**只在可观测层面定义，至少需要替代提示或闭集候选能稳定恢复正确关系。
- **内部预警：**在 artist 输出前，隐藏状态对完整关系标签的预测超过 logit、entropy、长度、词频、流行度和词法向量基线，并在实体与提示 disjoint 切分上泛化。
- **因果可用：**预注册干预能选择性提高正确 artist 的关系 margin，同时不主要靠破坏流畅性、改变输出长度或偏向热门实体。
- **不采用的表述：**意识、撒谎、主观确信和故意虚构。线性可解码性不能证明这些心理属性。

### 1.3 检索范围与方法

检索日期为 2026-07-14。来源包括 ACL Anthology、arXiv、ICLR/OpenReview、NeurIPS Proceedings、ACM/DOI 页面、Spotify Research，以及核心论文的前向与后向引文链。查询簇包括：

- `LLM music recommendation hallucination title artist`
- `generative recommendation out-of-domain constrained generation item token`
- `music recommendation track ID generative retrieval`
- `hallucination hidden states pre decoding factuality probe`
- `known facts hallucination inference dynamics logit lens`
- `mechanistic interpretability recommender relevance`
- `relation binding hallucination activation patching`
- `rationale first recommendation faithfulness`
- `self correction hallucination internal knowledge`

纳入标准：与音乐/推荐实体幻觉、隐藏状态事实性、关系机制、解释忠实性或自纠错直接相关的一手研究；优先 2020–2026 年，保留必要奠基工作。排除标准：只提供泛泛综述、仅讨论音频生成或视觉幻觉、只做 RAG 而不触及本问题、没有可核验一手页面，或只与通用 agent 工作流相关。

这是**定向结构化文献综述**，不是 PRISMA 系统综述：各检索服务没有提供可冻结、去重且可复现的总命中集，本轮也没有双人独立筛选。机器可读的纳入、排除与核验级别见 [`runs/academic_research_suite_literature_matrix_20260714.json`](../runs/academic_research_suite_literature_matrix_20260714.json)。

## 2. 最接近的既有工作

### 2.1 通用事实幻觉：已经非常接近，但实验时刻和泛化不同

[Jiang et al.（NAACL 2024）](https://aclanthology.org/2024.naacl-long.60/)构造查询同一 subject-relation-object 知识、却因提示不同而答对或答错的配对案例，并用 Logit/Tuned Lens 追踪输出 token 跨层动态。论文报告其动态特征分类器约 88% 准确率。这直接否定了“首次从隐藏状态发现模型已知关系仍会答错”这一主张。

但它与本项目仍有三处关键区别：检测特征回溯的是已经生成的首个答案 token；训练测试是随机向量切分，而不是 title、artist 或 relation disjoint；任务是单跳对象补全，不是开放式多项目、multi-token title-artist 生成。因此，我们只能主张更严格的**任务、时刻和泛化组合**，不能主张发现了这一现象本身。

[Mechanistic Understanding and Mitigation（Findings EMNLP 2024）](https://aclanthology.org/2024.findings-emnlp.466/)提出两类机制：较低层 MLP 没有充分注入 subject 属性的知识富集失败，以及较高层 attention 没能选出正确对象的答案提取失败。这个区分应直接成为我们的分析框架：strict title-artist conflict 不能一律叫“模型明明知道但说错”，必须先测正确 artist 是否可恢复。

[Two Pathways to Truthfulness（ACL 2026）](https://aclanthology.org/2026.acl-long.1173/)进一步把事实性信号拆成依赖 question-answer 信息流的 Q-Anchored 路径和由已生成答案自带证据形成的 A-Anchored 路径，并用 attention knockout 与 token patching 做因果区分。这是本项目目前最强的机制竞争工作。其检测输入包含问题和完整生成答案，且未报告实体 disjoint 测试；我们的差异只能落在**relation completion 前的位置、自然音乐错误和严格实体泛化**上。论文中的 `self-awareness` 是路径类别可解码的操作性标签，不应扩展为主观意识。

### 2.2 隐藏状态探针：可行，但现实泛化是核心风险

[FactCheckmate（Findings EMNLP 2025）](https://aclanthology.org/2025.findings-emnlp.663/)表明解码前输入隐藏状态可用于通用 QA 幻觉预测，并报告表示干预能提高事实性。[Simple Factuality Probes（Findings EMNLP 2025）](https://aclanthology.org/2025.findings-emnlp.880/)则把简单探针扩展到长文本生成。这些工作支持“内部预警”具有可行性，但没有验证 title-artist 关系绑定。

最重要的反证来自 [Are the Hidden States Hiding Something?（ACL 2025）](https://aclanthology.org/2025.acl-long.304/)：在人工合成真假模板上很好的事实性探针，迁移到模型自然生成的 TriviaQA、TruthfulQA 和 SQuAD 输出时可能接近随机。作者指出模板、随机替换、分布偏移和模型本身的知识状态都会制造虚假可分性。因此，本项目不能用手工交换歌手名的合成负样本证明自然幻觉可检测；合成 pair 只能做机制校准，主结果必须来自模型自由生成并经目录核验的错误。

[INSIDE（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)和[虚构引用检测（Findings EACL 2024）](https://aclanthology.org/2024.findings-eacl.62/)也说明内部表示和一致性行为能暴露幻觉风险，但二者同样不能替代关系级、输出前和严格 holdout 证据。

### 2.3 推荐和音乐系统：产品侧已能强力解决目录错误

[Text2Tracks](https://arxiv.org/abs/2503.24193)直接生成 track ID。论文报告 semantic ID 相对 artist-name/track-title 标识在 Hits@10 上提高 48%，且解码步数约少 7.5 倍。它清楚地说明：如果产品目标只是返回可播放歌曲，直接生成目录标识通常比自由写歌名更合适。

[RecLM（Findings ACL 2026）](https://aclanthology.org/2026.findings-acl.310/)在统一框架中比较 embedding retrieval、基于 prefix tree 的受限生成和离散 item token；其三个变体在三个基准上均达到 `OOD@10 = 0`。这意味着“消灭目录外推荐”不能作为我们的创新主张，受限生成应是产品可靠性强基线。

[WeMusic-Agent](https://arxiv.org/abs/2512.16108)让模型输出 `song_name` 与 `singer_name`，用约 300 万实体的目录事实奖励训练，并学习在内部知识不足时调用检索工具。[MuChator](https://arxiv.org/abs/2605.27103)同样把目录存在性纳入奖励，并报告接近饱和的离线事实性和线上收益。这些工业预印本进一步削弱“直接自由生成也能靠提示词上线”的现实意义，但它们没有公开分析 title 写完后、artist 尚未写出时的内部关系状态。其闭源数据和线上数字应视为作者报告，不能当作独立复现结论。

[Read Between the Tracks（NLP4MusA 2026）](https://aclanthology.org/2026.nlp4musa-1.7/)还观察到：即便提示中提供 40 首候选，开放式输出仍会产生候选外推荐。这支持将“提示约束”和“解码约束”分开，前者不能保证目录合法性。

[ERBench（NeurIPS 2024 Datasets and Benchmarks）](https://papers.nips.cc/paper_files/paper/2024/hash/5ef9853a6cdea40ae3e301a6d8dc32b5-Abstract-Datasets_and_Benchmarks_Track.html)已用关系数据库约束自动验证实体关系，并包含 Music DB。因此，“把歌名和歌手写成关系并自动核验”本身也不是新贡献；本项目的新意必须来自自然推荐过程中的内部时序和因果机制。

### 2.4 推荐可解释性：理由可能参与错误构造，而非解释既有选择

[Towards Trustworthy LLM-Based Recommendation via Rationale Integration](https://arxiv.org/abs/2601.02364)显示 rationale-first 训练/推理可以改善候选集推荐，但其训练理由由 GPT-4o 在已知目标 item 后生成，忠实性评价也没有做反事实因果验证。它支持理由有用，不证明理由忠实。

[Implicit Reasoning for LLM-based Generative Recommendation](https://arxiv.org/abs/2606.14142)提供相反方向的证据：显式 CoT 对措辞删除、增噪和类别信息移除很敏感，并可能引入语义 ID 与文本理由不匹配。结合[Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388)和 [FaithLM](https://aclanthology.org/2026.eacl-long.177/)，我们的既有 reason-first 结果更合理的解释是：理由改变了后续实体搜索或提前承诺，但它不稳定地证明了歌曲级忠实解释。

[How Do LLMs Understand Relevance?](https://doi.org/10.1145/3774942)已经把 activation patching 用到检索相关性判断，说明“推荐/检索领域做机制可解释性”也不是空白。我们需要把贡献限定在**生成式目录关系错误**，而非笼统声称首个 recommender mechanistic interpretability 工作。

## 3. 证据综合

### 3.1 已有证据共同支持什么

1. LLM 可能拥有某条事实的可恢复知识，却因提示和推理动态输出错误对象。
2. 隐藏状态中常能读出事实性信号，但探针非常容易利用模板、表面身份和数据分布捷径。
3. 非事实输出至少要区分知识没有写入/富集、关系绑定失败和最终对象提取失败。
4. 推荐系统可通过 ID、检索、受限生成、工具和目录奖励在产品层控制合法动作空间。
5. 流畅理由可能是决策依据、语义引导或事后合理化；只有反事实干预才能区分。

### 3.2 现有证据没有证明什么

- 线性 probe 成功不等于模型自己会使用该信号。
- 模型在复查提示后改口不等于首次输出时已经知道错误。
- title 和 artist 各自真实不等于 pair 真实。
- 某层被定位不等于编辑该层就能可靠修复。
- Song ID 的目录合法性不等于推荐相关性、属性正确或推荐理由忠实。
- 一个音乐领域探针超过随机不等于产生了领域独有的新机制。

## 4. 新意判定

### 4.1 不成立的主张

- 首次研究 LLM 推荐幻觉。
- 首次用隐藏状态检测模型已知事实的错误输出。
- 首次把音乐目录写成实体关系并自动验证。
- 首次发现 rationale-first 会影响后续答案。
- 从根本上解决大模型推荐幻觉。

### 4.2 当前可辩护的候选贡献

在本轮定向检索范围内，尚未发现工作同时做到：

1. 模型不接收候选列表，自由生成自然音乐歌单；
2. 标签针对完整 title-artist binding，而不是 item OOD、偏好矛盾或孤立实体；
3. 读取 title 结束后、artist 生成前的中间表示；
4. 在 title、artist、prompt-template disjoint 切分上验证；
5. 用匹配 exact/conflict 对和 sham patch 证明选择性因果效应。

因此候选论文贡献应写成：

> 我们把自然音乐推荐中的目录错误建模为一个分阶段的关系完成问题，并测试在关系完成前是否存在跨实体泛化且可因果使用的有效性信号。

这是一项**组合式和测量式新意**，不是全新理论。它只有在严格 holdout 和干预都通过后才足以支撑论文主张；若 probe 失败，同样可形成有价值的负结果，说明通用事实性探针不能迁移到自然 title-artist 生成。

## 5. Devil's Advocate 审查

1. **领域替换风险：**如果只是把通用 QA 的 probe 换成歌曲，审稿人会认为贡献不足。必须利用 multi-token、开放集合和 relation-completion 时序提出新的可证伪问题。
2. **产品价值风险：**工业系统已用 ID 和目录约束解决合法性。论文应诚实定位为基础机制研究，并明确推荐上线仍应优先采用约束方案。
3. **样本稀缺风险：**当前 7/80 的唯一严格冲突太少。通过 long-tail 提示刻意增产错误可能引入选择偏差，发现集和确认集必须分离。
4. **预训练知识不可见：**目录证明歌曲存在，不证明被测模型在预训练中见过它。必须用行为可恢复性和流行度/频率分层建立模型知识边界。
5. **时刻命名风险：**title 已经生成，因此不能称 `pre-output`；准确名称是 `pre-artist` 或 `pre-relation-completion`。
6. **捷径风险：**probe 可能只学到热门歌手、token 长度或 title 词形。必须加入 surface-only 模型、流行度匹配和双重实体 disjoint。
7. **干预副作用：**patch 可能只是把分布推向热门 artist，或破坏语法。需要 relation margin、非目标 token KL、长度和流畅性副作用指标。
8. **意识过度解释：**可解码性只支持“存在统计信号”，不支持心理意义上的自知或欺骗。
9. **工业证据不透明：**WeMusic-Agent 和 MuChator 很接近业务问题，但数据、目录和部分评估不可复现；既不能忽略，也不能把作者报告当成已验证事实。
10. **多重尝试风险：**旧 7 个案例已经被多轮分析，不可继续作为确认集。所有确认性阈值、层位和指标必须在新数据生成前冻结。

## 6. 下一阶段预注册草案

### Stage A：自然错误发现与冻结

- 使用自由生成，不给歌曲列表或 Song ID。
- 覆盖热门、长尾、跨年代、地区、相似艺人、强属性约束和组合场景；固定模型、模板、temperature 和 seeds。
- 对输出分成 `exact_binding`、`binding_conflict`、`nonexistent_title`、`nonexistent_artist`、`ambiguous_or_unverified`。
- 两个独立目录源自动核验；alias、cover、live、remaster、featured artist 和翻译名由盲法人工复核。
- discovery 目标：至少 30 个唯一 strict conflict title cluster，且每个有流行度和 token 长度匹配的 exact control。
- 达不到门槛即报告自然错误稀缺性并停止确认性机制结论。

### Stage B：冻结的 position-wise probe

- 位置：prompt end、reason end（如有）、title end / pre-artist、artist first token、full pair end。
- 标签：primary 为 exact vs strict binding conflict；nonexistent title 作为不同错误类型单独分析。
- 切分：nested group split，外层 title+artist disjoint，附加 prompt-template disjoint；禁止同一实体或别名跨 train/test。
- 基线：entropy、top-1 margin、序列 logprob、title 长度、token 数、流行度、词法 embedding、catalog-only classifier、随机标签和随机层。
- 指标：AUROC、AUPRC、balanced accuracy、95% cluster bootstrap CI；不只报告最佳层。
- 准入：pre-artist hidden-state probe 的外层 AUROC 置信区间下界超过最佳 surface baseline，且至少两个模型/种子方向一致。

### Stage C：知识边界与因果区分

- 对每个 strict conflict 遮住 artist，做闭集正确候选恢复和开放式事实问答，区分不可恢复与可恢复案例。
- 在匹配 exact/conflict 对上做 residual、attention 与 MLP patch；主指标为正确 artist 相对错误 artist 的 token-level/sequence-level margin。
- 对照：随机层、随机供体、实体不匹配供体、sham patch、标签打乱。
- 副作用：非目标 token KL、格式合规率、输出长度、重复率和热门 artist 偏移。
- 只有可恢复案例中出现选择性、跨实体复现的干预，才称为 binding/extraction failure 的证据。

### Stage D：理由作为次级因子

- 因子为 reason-first、item-first、no-reason；保持解码预算相同。
- reason swap、删除、反转和属性冲突提示用于检验歌曲级忠实性。
- 若理由只改变场景分布而不稳定改变具体 relation margin，则结论限定为 semantic steering 或 commitment，不称忠实解释。

## 7. 阶段性决策

**继续研究有意义，但必须换一个更严格的目标。** 下一步不是继续在旧案例上画更多层图，也不是做一个泛化 benchmark；而是建立足够的自然 conflict discovery pool，冻结新的 entity-disjoint 确认集，再回答 pre-artist 信号能否超越表面捷径。这个结果无论正负，都比“模型好像知道自己错了”更可靠，也更接近可发表研究。

## 8. 透明度声明

本轮由 Codex 在人工给定研究问题下，按照 `academic-research-skills-codex` 的定向综述、证据分级、反方审查和来源核验流程执行。模型用于检索辅助、结构化提取和初稿综合；关键数字和结论回到 ACL Anthology、arXiv、OpenReview/会议页、ACM DOI 或作者研究页核验。未做双人筛选、引用网络穷尽检索或正式 meta-analysis。
