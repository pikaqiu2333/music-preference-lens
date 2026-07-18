# Qwen-Scope 歌曲实体 Generation-Time Grounding Probe 摘要

## 运行信息

- 完整实验 Job: `6a506c5284e0eddc25f127bd`
- Job URL: https://huggingface.co/jobs/REDACTED/6a506c5284e0eddc25f127bd
- Verification control Job: `6a50a5076d2b10c09d6779eb`
- Control Job URL: https://huggingface.co/jobs/REDACTED/6a50a5076d2b10c09d6779eb
- 模型: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层: 24
- 场景: 雨夜驾驶、peak-time rave、情感人声、严格无人声写作
- 随机种子: `17`, `29`, `43`
- 有效生成: 11/12
- 可分析歌曲实体: 55 条，53 个唯一 `title + artist` 组合

## 研究问题

本轮不再只问“模型生成的歌名像不像虚构标题”，而是研究完整实体关系：

> 当模型自由生成 `歌曲名 + 歌手 + 推荐理由` 时，它是否已经知道该歌曲—歌手
> 关系不是稳定的已知音乐实体？这种内部不确定性是否会在事后核验时表现出来？

模型被明确要求推荐 5 首真实、已发行的歌曲。生成后，在同一次作业中记录 title、
artist、pair-end 和 reason span 的 layer-24 SAE 激活，并立即运行 neutral 与
self-attributed 两种精确实体核验。

## 技术覆盖

Qwen base model 使用了三种不同输出格式：同行 `Title / Artist / Reason`、
`Title by Artist` 后接单独理由，以及先列实体再写 `Recommendations`。解析器通过
真实输出回放后可恢复 11/12 次生成，共 55 条实体。唯一失败的生成只给了标题和
歌手，没有提供任何理由，因此保持无效而没有补造 reason。

预注册技术门是至少 10/12 次有效生成且至少 50 个实体，本轮实际是 11/12 和
55，技术门通过。

## 外部曲库结果

MusicBrainz 与 Apple Search 采用两阶段检索：先按标题召回，再用标题和歌手定向
补查，最后在本地严格比较实体关系。40 个控制实体预检查全部通过：20/20 真实
组合命中、10/10 错配组合识别为关系冲突、10/10 合成组合没有精确命中。

| Catalog label | 数量 | 比例 | 含义 |
|---|---:|---:|---|
| `verified_exact` | 6 | 10.9% | 曲库找到精确歌曲—歌手关系 |
| `catalog_conflict` | 47 | 85.5% | 标题存在，但没有找到该歌手关系 |
| `unverified` | 2 | 3.6% | 两个曲库都没有提供可靠实体证据 |
| `verification_error` | 0 | 0% | 无残留 API 错误 |

精确命中的 6 次推荐是：

- `Space Oddity — David Bowie`
- `Don't Stop Me Now — Queen`
- `All of Me — John Legend`，出现两次
- `Love Story — Taylor Swift`
- `Someone Like You — Adele`

因此本轮最稳固的结果是：

> 即使 prompt 明确要求真实歌曲，Qwen3-1.7B-Base 的 55 次音乐推荐里仍有
> 49 次没有得到精确歌曲—歌手关系支持。模型可以同时生成贴合场景的标题、歌手和
> 流畅理由，但实体关系大多不可靠。

## SAE Knownness 结果

40 个控制实体在四种上下文中形成 160 个 pair-end SAE 样本。各场景的组内方向
一致：真实组合的 knownness 均值约 `4.48–4.61`，非精确组合约 `-0.75–-0.61`。

但按实体 ID 划分的五折交叉验证 balanced accuracy 只有 `0.725`，低于预注册的
`0.80`。因此不能把该 knownness score 当作已验证的实体分类器。

可以作为探索性观察记录：

- 6 个外部精确命中实体的 knownness 全部为正，均值 `1.895`。
- 47 个关系冲突实体的均值为 `0.189`，其中 22 个为负。
- 2 个 unverified 实体的均值为 `-0.133`。
- knownness 与“是否外部精确命中”的 point-biserial correlation 为 `0.395`。

但高 knownness 并不等于关系正确。例如：

- `Skinny Love — Sam Smith`: knownness `3.661`，但属于 catalog conflict。
- `Love on the Brain — Coldplay`: `2.971`，catalog conflict。
- `My Heart Will Go On — Celine Dion`: `2.942`，catalog conflict。
- `Rainy Days — The City Pop Band`: `-1.502`，catalog conflict。

这说明当前 SAE 分数可能同时吸收了“标题像知名歌曲”“歌手本身常见”和音乐场景
特征，还没有干净隔离 `title -> artist` 关系绑定。入选特征中也出现了此前的
能量/rave 特征 `7884`、情感人声特征 `8706` 和广义上下文特征 `27029`，进一步
提示存在内容与场景混杂。

## 事后核验为什么不能证明模型知道

Pair-verification prompt 先在 40 个控制实体上单独校准：

| Group | N | Neutral Unknown 均值 | Unknown 获胜 | Self Unknown 均值 | Unknown 获胜 |
|---|---:|---:|---:|---:|---:|
| `known_exact` | 20 | -1.888 | 0 | -1.100 | 0 |
| `artist_mismatch` | 10 | -1.531 | 0 | -0.866 | 0 |
| `synthetic_pair` | 10 | -0.648 | 0 | -0.361 | 0 |

Neutral 和 self-attributed balanced accuracy 都只有 `0.50`。虽然合成组比真实组
更接近 Unknown，但没有任何控制实体真正选择 Unknown。因此该 prompt 只能看作
连续的不确定性指标，不能作为可靠二分类判断。

生成实体的 self-attributed 条件相对 neutral 条件，Unknown logit 平均上升
`+0.622`；但控制实体在相同措辞下也上升 `+0.632`。净效应是 `-0.010`，几乎为
零。

所以当前不能说：

> 模型在被提醒“这是你刚才生成的”以后，意识到自己虚构了歌曲。

更准确的结论是：

- self-attributed 措辞本身会普遍提高一点不确定性。
- 这种变化在生成实体上并不比控制实体更强。
- 当前没有发现特异于“自我生成来源”的知情信号。

## 当前回答

本轮已经证明的是生成式音乐推荐中的实体 grounding 风险，而不是模型的“自我
知情”：

> 模型能根据音乐场景生成很合理的标题、歌手和推荐理由，但绝大多数精确实体关系
> 无法通过外部曲库验证。内部 SAE 状态存在弱的分级信号，却没有达到预注册分类
> 门槛；事后 self-verification probe 也没有通过控制校准。

这对 LLM4Rec 可解释性很重要。解释可信度至少包含两个独立问题：

1. 推荐理由是否随用户偏好和场景变化。
2. 被解释的推荐对象本身是否是一个正确绑定、可验证的实体。

前面的实验对第一个问题给出了较强的 feature alignment；本轮显示第二个问题仍然
可能严重失败。理由写得合理并不能补偿实体关系错误。

## 局限

- 只有 Qwen3-1.7B-Base，一个模型和一个 SAE 层。
- 外部曲库不是完整世界知识库；本报告只说“得到/没有得到曲库支持”，不把 miss
  直接称为虚构。
- 55 条结果来自四种场景和三个种子，适合机制 pilot，不是 benchmark。
- knownness probe 和 verification prompt 都没有达到 0.80 解释门，相关数字只作
  探索性描述。
- 模型有一次完全省略理由，说明 base completion 格式仍有不稳定性。

## 下一步研究

下一轮应改成 matched relation-binding probe，而不是扩大样本：

1. 对每个真实标题构造正确歌手与错误歌手配对，例如
   `Shape of You — Ed Sheeran` 对 `Shape of You — Queen`，保持标题、上下文和格式
   完全相同，只改变 artist 关系。
2. 在 artist token 和 pair-end 后测量 SAE delta，避免把知名标题、常见歌手和场景
   特征混成一个 knownness 分数。
3. 增加 PMI-style relation score：比较 `log P(artist | title, context)` 与该歌手在
   中性标题下的先验，消除常见歌手本身的语言模型优势。
4. 对自由生成实体构造 artist-shuffle 反事实：原始配对与同一歌单内打乱歌手的配对
   直接比较，不需要人工判断推荐质量。
5. 只有 relation-binding 控制达到至少 0.80 后，才做 SAE intervention；方法稳定后
   再在 Qwen3-8B-Base 上复现。

这条路线继续专注机制问题，不需要建设人工音乐推荐 benchmark。
