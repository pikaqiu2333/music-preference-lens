# 推荐理由输出顺序实验（大白话版）

## 一句话结论

先写理由会强烈改变模型接下来想到的歌，但它不是稳定的“防幻觉技巧”。

在模型熟悉的情绪人声场景里，reason-first 的真实 title-artist 组合更多；在严格无人声这种强约束、长尾场景里，它没有产生任何真实组合，反而先写出理想属性，再编造符合这些属性的歌名和歌手。

## 实验设置

我们保持模型、用户需求、随机种子和采样参数不变，只交换每条推荐的输出顺序：

- pair-first：Title → Artist → Reason。
- reason-first：Reason → Title → Artist。

共测试两个场景、两个种子和两种顺序，得到 8 份歌单、40 条推荐。

原 Job 的解析器只支持字段各占一行，因此最初误报为技术失败。修正行内字段解析后，8/8 份生成和 40/40 条推荐都可提取，实际字段顺序符合率为 100%。原始文本和旧失败汇总均保留，模型输出没有被重写。

## 结果一：顺序改变了模型的搜索轨迹

四组同场景、同种子的 pair-first 与 reason-first 歌单中，完整 title-artist 组合的重合数全部为 0。

这说明理由放在前面并非只改变展示格式。理由 token 会成为后续 title 和 artist 的上下文，实际改变模型接下来搜索和生成实体的路径。

## 结果二：熟悉场景可能受益

情绪人声场景：

- pair-first：1/10 个组合被 catalog 验证为真实。
- reason-first：4/10 个组合被验证为真实。

reason-first 生成了 Shallow - Lady Gaga & Bradley Cooper、Blinding Lights - The Weeknd、Someone Like You - Adele 和 I Will Always Love You - Whitney Houston。

一种合理解释是：先写“强烈情绪、故事歌词、戏剧性副歌”等理由，可能把模型带向训练数据中关联度较高的知名歌曲实体。当前只有两个种子，不能声称它稳定提高 4 倍。

## 结果三：强约束场景反而更危险

严格无人声场景中：

- pair-first：0/10 个真实组合。
- reason-first：0/10 个真实组合。
- reason-first 另有 5/10 条使用 [Artist Name] 占位。

典型输出包括：

- Hypnotic Synthwave - [Artist Name]
- Silent Melody - [Artist Name]
- Instrumental Melody - [Artist Name]
- White Noise Melody - Nature's Symphony

这里的生成轨迹很像：

1. 先根据需求写出“无人声、白噪声、专注”等理想理由。
2. 再生成一个语义上听起来符合理由的歌名。
3. 模型没有成功从真实音乐实体记忆中找到对应歌曲，于是使用泛化名称或占位歌手。

这正是“理由作为规划”和“理由诱导幻觉”同时存在的例子。

## Catalog 结果

| 顺序 | 总数 | 真实组合 | Catalog conflict | Unverified | 占位歌手 |
|---|---:|---:|---:|---:|---:|
| pair-first | 20 | 1 | 19 | 0 | 0 |
| reason-first | 20 | 4 | 14 | 2 | 5 |

占位歌手即使因为同名歌曲被 catalog 标成 conflict，也仍单独视为明确无效输出。Unverified 只表示两个 catalog 都没有找到，不能单独证明歌曲不存在。

## 当前回答

可见理由在 pair-first 中是选完歌后补写的，容易事后合理化。

reason-first 则让理由真正成为后续选歌的因果上下文，但它只规定了“应该是什么样的歌”，并不保证模型能检索到真实 title-artist 实体。因此：

> 先给理由可以增强规划，也可能增强“言之有理的虚构”。

输出顺序本身不能从根本上解决推荐幻觉。产品上仍需要实体约束、检索或 Song ID 落库。

## 下一步

下一步做小规模 reason swap：

- 固定同一个 title-artist。
- 分别在它前面放原理由、其他歌曲理由、相反需求理由和中性理由。
- 测量完整 title-artist 概率是否随理由改变。
- 分开比较真实组合、配错歌手组合和概念化虚构组合。

这能判断模型是被理由中的真实属性引导，还是仅仅被表面词汇带向一个听起来合理的名字。

