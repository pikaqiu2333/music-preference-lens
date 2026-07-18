# AutoResearchClaw 文献调研启用与审计（2026-07-14）

## 一句话结论

AutoResearchClaw 已经真正运行，而不只是安装了一个同名 skill。它完成了选题、问题拆解、检索策略和文献收集，但没有完成自动筛选与综合。因此，本次结果可以作为**可审计的广泛召回**，不能冒充系统综述。

精准补检后的研究判断是：

- “怎样不让推荐系统输出目录外歌曲”已经有很强的 ID、检索和受限生成方案，单做这个产品问题的新意较弱。
- “模型在输出错误的 `title–artist` 关系前，内部是否已有可识别的错误信号；为什么这个信号没有控制最终输出”仍然是有价值的问题。
- 研究单位必须是完整的歌名—歌手关系。只研究歌手会把实体熟悉度、关系绑定和最终选择混在一起。
- “模型改口了”不等于“模型原本知道自己错了”；“探针能读出来”也不等于模型自身能使用该信息。

## 运行回执

机器可读回执见 [`runs/autoresearchclaw_literature_review_receipt_20260714.json`](../runs/autoresearchclaw_literature_review_receipt_20260714.json)。

| 项目 | 结果 |
|---|---|
| 工具 | ResearchClaw 0.5.0，识别到 22 个 skills |
| Run ID | `rc-20260714-020432-4b4cf6` |
| 已完成 | Stage 1–4：选题、拆解、检索策略、文献收集 |
| 未完成 | Stage 5：文献筛选；后续知识抽取和综合未执行 |
| 候选池 | 442 条 JSONL，440 条 BibTeX |
| 失败原因 | Codex ACP 使用额度到达上限；本地提示可在 2026-07-20 17:50 后再试 |
| 产物状态 | 原始产物和 checkpoint 已保留，关键文件 SHA-256 已写入回执 |

## 为什么不能直接采用 442 条结果

自动生成的查询过于宽泛，例如 `mechanistic interpretability` 和 `self-correction hallucinated`。候选池前部出现了医学系统综述、时序预测、地震等无关论文。Semantic Scholar 和 arXiv 又在收集期间返回 HTTP 429，最终主要依赖 OpenAlex。

我们用 13 篇事前指定的核心论文标题做了最小召回审计，只找到 1 篇：*Large Language Models have Intrinsic Self-Correction Ability*。这说明：

1. 442 不是 442 篇相关论文，只是 442 条待筛候选。
2. 当前查询既有大量误召回，也漏掉了 MechELK、FactCheckmate、HIDE、Text2Tracks 和 J-space 等关键工作。
3. 即使 ACP 额度恢复，也不应原样把 Stage 5 跑到底；应先替换查询或注入人工核验的种子集。

## 精准补检证据

以下只记录一手论文页、会议页、OpenReview 或作者研究页。`摘要核验`表示本轮只据官方摘要定位，不表示已逐页复核实验细节。

### 音乐推荐与目录约束

| 工作 | 核心证据 | 对本项目的含义 | 核验 |
|---|---|---|---|
| [Music Recommendation with Large Language Models](https://arxiv.org/abs/2511.16478) | 明确把幻觉、知识截止、非确定性和不透明训练数据列为 LLM 音乐推荐风险，并把任务定义为生成可解析目录项。 | 证明问题真实存在，但它是综述与评测框架，不回答内部机制。 | arXiv HTML/摘要 |
| [Text2Tracks](https://arxiv.org/abs/2503.24193) | 直接从自然语言生成 track ID；semantic ID 优于以歌名为标识的策略。 | ID 路线减少实体解析和目录外输出，但不证明自由生成模型知道自己的错误。 | arXiv HTML/摘要 |
| [Eliminating Out-of-Domain Recommendations in LLM-based Recommender Systems](https://aclanthology.org/2026.findings-acl.310/) | RecLM 比较检索、受限歌名生成和离散 item token；在其基准中三种变体均达到 `OOD@10 = 0`。 | “目录合法性”已有强产品基线；我们的新意不能写成“首次消除虚构项目”。 | ACL 正式论文页 |
| [TalkPlay-Tools](https://arxiv.org/abs/2510.01698) | 用 tool calling 编排过滤、稀疏/稠密检索和 semantic-ID 生成。 | 工业系统可把 LLM 作为规划器而不是事实数据库。 | arXiv 摘要 |
| [How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/abs/2305.06569) | 系统比较 item ID 表示。 | 合法动作空间与自然语言实体生成是两个不同问题。 | arXiv 摘要 |
| [Logit Space Constrained Fine-Tuning](https://aclanthology.org/2025.emnlp-main.1491/) | 用正负指令对和 logit-space 约束降低推荐幻觉。 | 表明推荐幻觉也可在训练目标层面处理，但没有回答单次自由生成的内部自知。 | ACL 正式论文页 |
| [On Faithfulness and Coherence of Language Explanations for Recommendation Systems](https://arxiv.org/abs/2209.05409) | 推荐解释可连贯却对输入扰动不稳定。 | 推荐语“听起来有道理”不能当作选歌依据的忠实证据。 | arXiv 摘要 |
| [ReasoningRec](https://aclanthology.org/2025.findings-naacl.454/) | 用大模型生成偏好解释并微调推荐模型。 | 更接近“用理由提高推荐”，不验证理由是否反映真实内部因果。 | ACL 正式论文页 |

### 隐藏状态、关系知识与因果机制

| 工作 | 核心证据 | 对本项目的含义 | 核验 |
|---|---|---|---|
| [Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations](https://aclanthology.org/2024.findings-emnlp.466/) | 在 subject–relation 查询中区分低层 MLP 的知识不足和高层 attention 的答案提取失败。 | 与 `title–artist` 最接近：错误可能来自没有关系知识，也可能来自有知识但选错对象。 | ACL 正式论文页 |
| [FactCheckmate](https://aclanthology.org/2025.findings-emnlp.663/) | 用解码前输入隐藏状态预测 QA 幻觉，并通过隐藏状态干预提高事实性。 | 直接支持“输出前预警”可行，但尚未覆盖开放音乐实体关系。 | ACL 正式论文页 |
| [HIDE and Seek](https://arxiv.org/abs/2506.17748) | 单次生成中，用输入与输出隐藏表示的解耦检测幻觉。 | 可作为无需训练或低训练成本的检测基线，但信号出现在生成过程中，不一定早于实体决策。 | arXiv HTML/摘要 |
| [Do Language Models Know When They’re Hallucinating References?](https://aclanthology.org/2024.findings-eacl.62/) | 通过对同一引用反复询问作者等属性，用一致性检查区分真实与虚构引用。 | 歌名—歌手也可当“易核验模型生物”；但行为不一致仍不是主观意识证据。 | ACL 正式论文页 |
| [Discovering Latent Knowledge in Language Models Without Supervision](https://openreview.net/forum?id=ETKGuby0hcs) | CCS 从无标签隐藏状态中恢复与表面回答不同的真假信号。 | 说明“模型说什么”和“激活中可读出什么”可以分离；仍需排除探针制造信息。 | ICLR/OpenReview |
| [MechELK](https://arxiv.org/abs/2605.28825) | 组合 SAE、activation patching、因果探针和表示干预来定位、验证并引出 latent knowledge。 | 提供方法模板，但我们必须用 held-out 关系和随机干预证明不是词频或实体熟悉度。 | arXiv 摘要 |
| [Relational Linearity is a Predictor of Hallucinations](https://arxiv.org/abs/2601.11429) | 在合成实体的六类关系中，关系线性与幻觉率高度相关。 | 强化“关系本身”而非孤立歌手是分析单位，也提示不同关系的自我评估难度可能不同。 | arXiv 摘要 |
| [Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html) | J-space 与可报告、可调节和灵活推理有关，常规自动加工可能绕过它。 | 歌名检索若是自动联想，J-lens 没信号并不等于模型无知识；它只能作为次级解释框架。 | 作者完整研究页 |
| [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213) | 因果定位与最佳编辑位置并不等价。 | “某层能读到关系”不能直接推出“改这一层就能修复输出”。 | arXiv 摘要 |
| [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) | patching 的指标和 corruption 选择会显著改变定位结果。 | 后续必须预注册 patch 指标、对照和位置，不能看完热图再挑层。 | arXiv 摘要 |

### 推荐理由与自我纠错

| 工作 | 核心证据 | 对本项目的含义 | 核验 |
|---|---|---|---|
| [FaithLM](https://aclanthology.org/2026.eacl-long.177/) | 把解释忠实性定义为干预性质：与解释相反的提示应改变预测。 | 可把我们的 reason-swap 从“文本一致性”升级为预注册的反事实干预。 | ACL 正式论文页 |
| [Towards Faithful Natural Language Explanations](https://aclanthology.org/2025.emnlp-main.529/) | 用 activation patching 比较解释与答案的因果归因。 | 直接连接我们已有的理由顺序实验与机制 patching。 | ACL 正式论文页 |
| [Are self-explanations from LLMs faithful?](https://aclanthology.org/2024.findings-acl.19/) | 自解释忠实性依赖模型、任务和解释形式。 | 不能把推荐语统一当成可靠解释。 | ACL 正式论文页 |
| [Language Models Don’t Always Say What They Think](https://arxiv.org/abs/2305.04388) | 模型会在受未说明偏置影响后生成看似合理的解释。 | 与“言之有理的错误歌名—歌手组合”高度相符。 | arXiv 摘要 |
| [Order Matters in Hallucination](https://arxiv.org/abs/2408.05093) | answer-first 与 logic-first 会产生不同幻觉表现。 | reason-first 的顺序效应不是独立新意；音乐关系和目录核验才是领域贡献。 | arXiv 摘要 |
| [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) | 无外部反馈的自纠错常无效，甚至降低表现。 | 不能只问“再检查一次”就声称模型有内在错误意识。 | arXiv 摘要 |
| [Large Language Models have Intrinsic Self-Correction Ability](https://arxiv.org/abs/2406.15673) | 在零温度和作者定义的公平提示下报告了相反结果。 | 自纠错结论依赖提示和解码设置，必须做成对照实验。 | arXiv 摘要 |
| [Large Language Models Can Self-Correct with Key Condition Verification](https://aclanthology.org/2024.emnlp-main.714/) | ProCo 通过遮蔽关键条件并重新预测来验证答案。 | 对歌名—歌手可改写成“遮住歌手，模型能否稳定恢复”，但这是验证任务，不是自由推荐本身。 | ACL 正式论文页 |
| [When Can LLMs Actually Correct Their Own Mistakes?](https://aclanthology.org/2024.tacl-1.78/) | 批判性综述认为可靠外部反馈和适合验证的任务更容易成功。 | 目录核验与 intrinsic correction 必须分栏报告。 | TACL/ACL 正式页 |
| [The Self-Correction Illusion](https://arxiv.org/abs/2606.05976) | 保持错误内容不变，只改变 chat role 就能大幅改变纠错率。 | “改口”可能来自角色模板，不是对初次错误的稳定内部认识。 | arXiv 摘要 |

## 对当前项目的真正影响

### 1. 项目仍有意义，但论文问题需要收窄

最稳妥的问题不是“怎样从根本上消灭推荐幻觉”，而是：

> 在开放式音乐推荐中，当模型即将生成目录不支持的 `title–artist` 关系时，隐藏状态是否包含可跨实体泛化的关系有效性信号；该信号反映知识缺失、关系绑定失败，还是最终对象提取失败？

本轮没有找到同时覆盖“开放音乐推荐 + 完整歌名—歌手关系 + 输出前隐藏状态 + held-out 因果纠错”的一手工作。这个交叉点仍可形成贡献，但只能表述为**本轮检索尚未发现直接覆盖**，不能写成绝对的“首次”。

### 2. 我们之前一直研究歌手，不够准确

只看歌手会得到“模型认识 Taylor Swift”这种实体熟悉度结论，却回答不了“这首歌是否真是 Taylor Swift 的”。正确标签至少要分成：

1. title 不存在；
2. artist 不存在；
3. 两者都存在但 binding 不成立；
4. 完整关系有目录支持。

后续主分析必须以完整 pair 为单位，title-only 和 artist-only 只做诊断对照。

### 3. Song ID 没有让研究失去意义

Song ID、检索和受限解码能很好解决线上系统的合法输出问题，却不回答自由生成模型内部发生了什么。两条路线应明确分开：

- 产品结论：要可靠上线，优先使用目录检索、ID 或受限生成。
- 科学结论：研究内部预警、关系绑定和错误提取，解释为何模型有时“似乎知道”却仍输出错误。

前者是必须比较的强基线，后者才是本项目可能的新贡献。

## 下一阶段研究流程

### Stage A：先解决自然错误样本产出

- 继续让模型自由生成新歌单，不给候选列表。
- 用 long-tail、跨年代、强属性和相似艺人等场景扩大 discovery pool。
- 用固定目录快照标注完整 pair，并人工复核别名、cover、feat、live 和翻译标题。
- discovery 只用于找失败模式；在看到机制结果前冻结全新的 entity-disjoint holdout。
- 若唯一严格关系冲突仍达不到预注册门槛，就停止机制显著性主张，不拿 7 个旧案例反复试到“有结果”。

### Stage B：输出前关系错误检测

- 主要位置：请求末端、title 结束后且 artist 尚未生成、完整 pair 结束后。
- 主要比较：hidden-state probe 对比 logit、entropy、流行度、词长和实体熟悉度基线。
- 主要切分：artist-disjoint、title-disjoint、prompt-template-disjoint。
- 主要表述：只称“可预测的内部信号”，不称“意识”或“主观知道”。

### Stage C：区分缺知识、绑定失败和提取失败

- 对两实体都真实但 pair 错误的案例，测试正确 artist/title 是否可通过候选打分恢复。
- 只有在 held-out probe 通过后才做 activation patching。
- 加入随机层、随机样本、标签打乱、流行度匹配和 token 长度匹配对照。
- 只有定向干预能选择性恢复正确关系，才讨论 latent relation knowledge 或 binding failure。

### Stage D：最后再处理理由

- reason-first、item-first 和 no-reason 保持输出预算与解码参数一致。
- 使用 FaithLM 式 contrary-hint、reason swap 和内部 activation patching 同时测行为与机制。
- 如果理由只改变场景级选歌方向，不稳定指向具体 pair，就把它写成 semantic steering，而不是 faithful explanation。

## 当前决策

1. 继续研究，但主线改为完整关系的内部错误信号与因果区分。
2. 不把新数据集包装成 benchmark；目录标注只是实验测量工具。
3. 不再把“模型改口”“模型自信较低”或“线性探针能分类”单独解释为自知。
4. Phase 2 的 7 个冲突继续保留为样本产出边界，不用于追补确认性结论。
5. AutoResearchClaw 恢复后只做独立复核：先修订查询，再从筛选阶段运行；不得用它替代人工的一手引用核验。

## 全文复核队列

正式写论文前，优先逐页核对以下工作的方法、数据划分、失败案例和限制：RecLM、Text2Tracks、FactCheckmate、Mechanistic Understanding、Relational Linearity、FaithLM、activation-patching NLE、Self-Correction Illusion。其余记录目前足以支持研究定位，不足以支持精确数字对比。
