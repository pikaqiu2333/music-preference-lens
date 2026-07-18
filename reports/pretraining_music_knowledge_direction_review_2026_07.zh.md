# 从训练证据到可调用知识：音乐关系预训练研究方向审查

日期：2026-07-14

## 一句话结论

DeepSeek 公布了许多模型权重和技术细节，但没有公布足以追溯具体歌曲知识来源的完整预训练语料、逐样本顺序和端到端训练记录，因此不适合作为这项研究的主模型。

新的可发表方向不是继续问“模型是否意识到自己在幻觉”，而是研究：

> 在总暴露次数相同的条件下，独立来源、表达多样性、重复簇和冲突信息，如何决定一条 title-artist 关系从训练文本进入模型表示，并最终变成无需候选也能稳定调用的知识？

这比此前围绕 11 个自然错误案例继续做 probe 更扎实。它直接解释长尾音乐知识为何缺失、为何候选比较看似知道而自由生成仍会说错，也保留了可解释性的核心：把训练证据、内部表示和输出行为连成可检验的链条。

## 1. DeepSeek 是否“全公布”

答案是：**主力模型大多公布了权重，但没有全栈公布训练过程。** “开放权重”和“可审计预训练”是两件不同的事。

| 系列 | 已公开的主要内容 | 对本研究缺失的关键内容 | 可承担的研究角色 |
|---|---|---|---|
| DeepSeek LLM 7B/67B | Base/Chat 权重、技术报告、若干训练指标、官方中间 checkpoint | 原始语料、文档 ID、完整混合比例、逐样本顺序、端到端 trainer 和原始日志 | 可观察行为在训练中何时出现，不能定位由哪些歌曲文档导致 |
| DeepSeek V2/V3 | Base/Chat 权重、架构与训练细节、部分底层系统和推理代码 | 精确语料、样本顺序、完整过滤配方、密集中间 checkpoint、完整训练代码 | 能力对照，不适合数据到知识追踪 |
| DeepSeek R1 | R1-Zero、R1 和蒸馏权重，后训练阶段和数据规模概述 | V3 原始语料、完整 RL/SFT 样本、rollout、训练日志和中间轨迹 | 后训练行为研究；不适合追溯歌曲知识来源 |
| DeepSeek V4 | Flash/Pro 的 Base 与后训练权重、技术与推理材料 | 仍未公开逐文档语料、样本顺序和完整中间训练轨迹 | 大模型能力确认；成本和审计链均不适合主实验 |

DeepSeek LLM 官方仓库明确提供了 7B/67B 的 AWS S3 中间 checkpoint，也只描述语料由互联网、数学、代码、书籍和自采文本构成。V3 报告披露了 14.8T token、架构和训练成本，R1 仓库披露了两段 RL、两段 SFT 及蒸馏关系，V4 也发布了 Base 权重；这些都没有补齐“某个训练步实际看到哪篇歌曲文档”的证据链。

因此，DeepSeek 可以回答“模型在某个 checkpoint 会不会答”，不能严谨回答“它看过哪些歌曲材料，这些材料是否让它学会”。主要依据见 [DeepSeek LLM 官方仓库](https://github.com/deepseek-ai/DeepSeek-LLM)、[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)、[DeepSeek-R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)、[DeepSeek-V4 官方发布](https://api-docs.deepseek.com/news/news260424)和[官方权重集合](https://huggingface.co/collections/deepseek-ai/deepseek-v4)。

## 2. 预训练语料中会有哪些歌曲内容

对一首真实歌曲，网页语料中可能出现的并不是一条干净的 `(title, artist)` 三元组，而是以下混合物：

1. 百科和艺人介绍：明确写出发行、演唱、创作、收录专辑等关系。
2. 专辑 tracklist、唱片目录和结构化元数据：关系很明确，但常有导航模板和镜像重复。
3. 乐评、采访、新闻、榜单、获奖记录和歌单文章：表达丰富，关系可能跨句出现。
4. 歌词页、歌词引用和歌曲背景说明：可能包含强词面记忆，也受版权清理影响。
5. 论坛、社交媒体和用户评论：语言自然，但事实质量和实体消歧较弱。
6. 数据库导出、代码和知识图谱文本：格式规则，容易重复，却未必支持自然问答调用。
7. 同名歌、翻唱、remix、featured artist、作曲者与表演者：同一个标题可能合法对应多个对象。
8. 错误转载和无关同现：歌名与热门艺人出现在同一页，不代表页面断言二者有关系。

所以，不能只统计 `title` 和 `artist` 同页出现多少次。至少应记录：

```text
(title_id, artist_id, relation_type, assertion_status,
 source_id, duplicate_cluster, expression_type,
 token_position, training_step, conflict_target)
```

其中 `assertion_status` 要区分真实断言、否定、引用错误、导航列表和无关同现；`relation_type` 要区分主艺人、合作、翻唱、remix、作者与表演者。MusicBrainz 的 MBID 可作为实体锚点，但目录本身不能证明某段文本进入了模型训练集。

## 3. 必须分开的五层证据

本方向最重要的方法学纪律是：

```text
开放语料中出现
  != 属于该模型的精确训练序列
  != 对参数更新产生因果影响
  != 在隐藏状态中可被解码
  != 在无候选提示下可稳定调用
```

此前 Phase 3 的 4/11 正 pairwise margin、0/11 candidate-free recoverability 正好说明最后两层不能互换：两个候选中“相对更不差”，不代表模型能主动检索正确艺人。后续任何 probe、LRE 或 patching 都必须以独立的候选无关行为门为前提。

## 4. 已有研究做到哪里

已有工作证明了若干局部规律，但还没有形成我们需要的完整闭环。

| 已有结论 | 代表工作 | 留下的问题 |
|---|---|---|
| 相关训练文档越多，长尾事实平均越容易回答 | [Kandpal et al., ICML 2023](https://proceedings.mlr.press/v202/kandpal23a.html) | 同文档共现不等于关系断言，也没有处理多对多音乐关系 |
| 事实由多次小概率增益逐步获得，之后还会遗忘；重复数据遗忘更快 | [Chang et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6fdf57c71bc1f1ee29014b8dc52e723f-Abstract-Conference.html) | 主要依赖受控虚构事实，未验证真实网页中的来源多样性和冲突 |
| 释义和句序多样性可帮助“存了但提不出”的知识变得可调用 | [Allen-Zhu & Li, 2023](https://openreview.net/forum?id=5x788rqbcj) | 合成单值传记不能直接外推到真实音乐页面 |
| 高频错误共现对象可能压过低频正确对象 | [Kang & Choi, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.518/) | 微调实验不能直接等同于大规模预训练动力学 |
| 事实共现频率与线性关系表示显著相关 | [Merullo et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b35c6f99df526a798433fabff039bd50-Abstract-Conference.html) | 线性可解码不是知识存在的必要条件，也不是自由回忆的充分条件 |
| 找到显式事实的文档与找到真正影响输出的文档不是同一任务 | [Chang et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/65798a76cc176c29b6bfefe84b0a03ff-Abstract-Conference.html) | 大规模 influence 仍不是严格的单文档反事实证明 |
| 提示变化会显著改变可观察知识 | [Jiang et al., TACL 2020](https://aclanthology.org/2020.tacl-1.28/) | 单一问法只能给出知识下界，需跨模板稳定性定义 |

现有证据共同支持：频率重要，但关系特异性、表达多样性、重复方式、出现顺序、遗忘、冲突和模型容量都会改变最终召回；不存在跨模型通用的“出现 N 次就学会”阈值。

## 5. 真正可辩护的研究空缺

单做“歌曲共现次数是否预测回答正确”会与长尾知识和频率研究高度重叠，论文新意不够。音乐只有在利用其真实结构时才不是换皮：

1. title-artist 天生是多对多关系，包含同名、翻唱、合作、remix 和 credit 角色差异。
2. 音乐网页包含大量镜像目录与模板重复，可自然区分“重复次数”和“独立来源多样性”。
3. 错误艺人往往不是随机对象，而是受标题歧义、热门实体先验和邻近上下文影响的竞争对象。
4. 推荐系统最终需要主动生成或绑定实体，因此 forced choice 与 candidate-free access 的差距有直接产品含义。

据此，建议冻结主研究问题：

> **RQ：**在控制总关系暴露量后，独立来源多样性、表达多样性和冲突比例，如何影响多对多 title-artist 关系的获取、内部表示、遗忘和候选无关调用？

三个可证伪假设：

- **H1 多样性效应：**相同总暴露次数下，跨来源、跨表达的证据比同一文档重复更早形成稳定的 candidate-free recall。
- **H2 冲突效应：**错误或多义关系会扩大“pairwise margin 为正、自由回忆失败”的差距，并延后稳定获取时间。
- **H3 表示到调用的转变：**关系信号可能先变得可解码，随后才成为跨模板稳定、可因果使用的输出知识；二者不是同一个时刻。

## 6. 模型选择

### 主试验：Pythia-1B-deduped

[Pythia](https://github.com/EleutherAI/pythia) 为每个模型提供 154 个 checkpoint，所有规模使用相同数据和相同顺序，并发布预分词数据及重建 dataloader 的方法。它的架构和语料较旧，但最适合低成本验证“哪段训练序列之后，关系行为发生了什么变化”。

### 确认与因果继续训练：OLMo-1B

[OLMo](https://arxiv.org/abs/2402.00838) 发布 Dolma、数据顺序工具、每 1000 步 checkpoint、优化器状态、训练配置和日志。[Dolma](https://aclanthology.org/2024.acl-long.840/) 也公开了网页、论文、代码、公共领域图书、社交媒体和百科等来源。它比 Pythia 更适合复现自然观察并进行保持原训练流的受控关系注入。

### 现代能力复现：OLMo 2 1B，必要时 OLMo 3 7B

[OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B) 公开数据、配置、日志和原始 OLMo-format checkpoint。需要注意，方便直接加载的 `OLMo-2-0425-1B-early-training` 是后续按同配置重跑的轨迹，官方明确说明其不与原始训练完全相同；它不能与原始 final checkpoint 拼成一条精确因果轨迹。现代模型只承担外推确认，不承担主来源结论。

### 只作能力对照：DeepSeek、Qwen

DeepSeek 和 Qwen 可以检验更强模型是否也表现出 forced-choice/free-recall 落差，但因没有精确语料和顺序，不能用于“哪类训练证据导致知识形成”的核心结论。

## 7. 推荐的研究流程

### 阶段 A：自然语料可行性试验

1. 从 MusicBrainz 抽取 120 个英语关系：40 个单值主艺人、40 个合法多值关系、40 个同名或竞争关系。
2. 在 Pythia 精确训练语料中检索 title、artist、别名和关系表达，保留 shard、row、hash、token offset，不发布完整版权文本。
3. 自动聚类镜像重复，识别来源域和表达模板；只对候选片段做小规模分层人工核验。
4. 对每条关系形成累计暴露曲线：有效断言数、独立来源数、表达多样性、重复集中度、冲突比例和最近一次暴露位置。
5. 在事件前后 checkpoint 测量目标序列 log probability、无候选贪心补全、跨模板稳定性和反向 artist-to-title 回忆。

这一阶段不训练 detector，也不做 intervention。它只回答样本、语料映射和行为信号是否足以支撑正式研究。

### 阶段 B：受控继续预训练

若自然试验通过，再在 OLMo/Pythia 小模型上进行 `2 x 2 x 2` 因子实验：

- 低/高关系暴露次数；
- 单一文本重复/多来源多表达；
- 无冲突/固定比例竞争关系。

使用来自真实 MusicBrainz 关系图的匿名实体，生成元数据卡、乐评、采访和榜单等不同文体；保留真实的一对多结构，但替换名称以避免已有知识泄漏。至少 3 个 seed，并在注入后继续喂入无关原训练数据，测量 washout 和遗忘。主指标是未见提示模板上的稳定 candidate-free exact accuracy，而不是 forced-choice accuracy。

### 阶段 C：可解释性闭环

只有 H1/H2 的行为结果稳定后才进入：

1. 逐 checkpoint 测量层级 target-logit trajectory 和关系线性表示质量。
2. 使用 title、artist、关系和提示模板 disjoint 切分，排除词面与流行度捷径。
3. 检验表示信号是否时间上先于稳定自由回忆。
4. 用 LRE/activation intervention 验证该表示是否能选择性改变正确艺人的生成概率，并设置随机层、打乱关系和同长度对象 sham control。
5. 若只能解码、不能因果改变行为，结论必须停在“表示可读”，不能写成“模型会使用”。

## 8. 首轮停止条件

为避免再次围绕少量案例无限追加实验，预先采用以下边界：

1. 如果精确训练语料中无法获得至少 80 条经核验关系，并覆盖重复主导、来源多样和冲突三类，停止自然轨迹主张。
2. 如果关系证据自动标注在 200 条分层人工审计上的 precision 低于 0.90，先修数据管线，不运行 GPU。
3. 如果 candidate-free 指标在著名正对照上不能达到 80%，更换提示或模型，但不得用 forced choice 代替。
4. 如果多样性效应只出现在见过的问法、pairwise margin 或训练后立即测量，且在未见模板/washout 后消失，H1 失败。
5. 如果自然观察和受控继续预训练的效应方向不一致，不写因果结论。
6. 如果没有行为效应，不做大规模 probe、SAE 或 patching。

受控主效应可暂以未见模板上的绝对准确率差 `>= 10` 个百分点作为继续门；正式功效分析应在无标签 smoke 后、查看主结果前冻结。

## 9. 与此前实验的关系

此前工作不是作废，而是被降级为动机和测量经验：

- Phase 3 的 `0/11` 表明自然目录冲突大多是知识缺失或关系不稳定，不能直接研究“明知故错”。
- `4/11` 正 pairwise margin 但自由恢复失败，提出了新的核心现象：**表示/相对偏好与主动调用之间存在缺口。**
- 已有 MusicBrainz 规范化、别名匹配、无候选提示、artifact hash 和 HF Jobs 流程可以复用。
- 既有 Qwen 隐藏状态结果不能作为新研究的确认性证据，也不能与新样本混作训练集。

## 10. 研究价值与诚实边界

这项研究有论文价值，但不是因为“音乐推荐会幻觉”这一现象新，而是因为它有机会提供：

1. 一个按精确训练步索引、区分真实关系断言与表面共现的音乐暴露数据集；
2. 频次、来源多样性、重复与冲突对知识获取和遗忘的分离结果；
3. 从训练证据到表示，再到候选无关调用的时间链；
4. 自然语料观察与受控继续预训练相互验证的因果证据。

它不能承诺“从根本上解决幻觉”。即使找到稳定机制，生产推荐仍应优先使用 Song ID、目录约束、检索和工具验证。研究贡献是说明模型何时形成了可调用关系、何时只有脆弱信号，以及怎样的数据配方更可能减少长尾实体错误。

对产品经理背景而言，这个方向并不吃亏：最关键的工作之一正是定义音乐关系、冲突类型、证据质量和可失败的产品行为。真正需要补齐的是统计设计和训练工程，而不是假装单靠 probe 就能提出大理论。

## 11. 当前建议

**进入阶段 A，不立刻做大规模训练。** 先完成 120 条关系的语料可检索性 smoke、语义证据 schema 和 12 个 Pythia checkpoint 的最小行为曲线。只有数据产率、标注精度和 candidate-free 正对照同时过门，才启动受控继续预训练。

对应的冻结协议见 [`docs/pretraining_music_knowledge_feasibility_protocol.md`](../docs/pretraining_music_knowledge_feasibility_protocol.md)，机器可读证据矩阵见 [`runs/pretraining_music_knowledge_research_matrix_20260714.json`](../runs/pretraining_music_knowledge_research_matrix_20260714.json)。
