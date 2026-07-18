# 真实歌名+歌手的字段级失败定位报告

## 结论

完整 title-artist pair 的两个失败并不是同一种错误：

- `Space Oddity - David Bowie`：title 与 artist 都被匹配理由压低，属于整实体方向错误；
- `Awake - Tycho`：title 被强烈压低，但 artist 反而被正确支持，属于明显的字段内部冲突。

这说明生成时预警至少需要两种信号：字段之间是否互相矛盾，以及自由生成路径是否与独立关系验证路径矛盾。只看 title/artist 分歧会漏掉 `Space Oddity`，只看完整 pair 又看不出 `Awake` 内部已经出现了相反证据。

## 字段效应

数值是匹配理由减相反理由的平均 token log probability。正数表示匹配理由支持真实字段，负数表示反而压低它。

| 真实歌曲 | Title 效应 | Artist 效应 | 完整 Pair | 预注册标签 |
| --- | ---: | ---: | ---: | --- |
| Blinding Lights - The Weeknd | +0.430 | -0.027 | +0.202 | artist only, weak boundary |
| Space Oddity - David Bowie | **-0.201** | **-0.065** | **-0.147** | both fields |
| River Flows in You - Yiruma | +0.071 | +0.552 | +0.251 | neither field |
| Awake - Tycho | **-1.034** | **+0.292** | **-0.150** | title only |

`Blinding Lights` 的 artist 负值小于事先规定的 `0.03` 弱效应门槛，不能解释成实质性的歌手失败；它的正确 pair 结果主要由强 title 信号贡献。`River Flows in You` 则主要由 artist 信号贡献。

## 冻结 Head 的作用

上一轮从独立候选判断中冻结的 `head 0/1/8/9` 仍然主要影响 title：

- 在 `Space Oddity` 中，四个 head 一起搬运会重现 `155%` 的错误 title 效应，但对 artist 的恢复接近零；
- 在 `Awake` 中，它们重现 `31%` 的错误 title 效应，同时重现 `24%` 的正确 artist 效应；
- 因此同一组 head 可以把全局 vocality 条件送入两个字段，但字段与具体实体的结合方向仍可能不同。

`Awake` 尤其重要：模型内部不是“什么都不知道”，而是同一时刻存在互相竞争的证据。title 路径把真实歌曲推远，artist 路径却把真实歌手拉近。这是可以进一步研究为输出前冲突信号的最小实例。

## 对迁移实验的约束

后续自由生成迁移不训练分类器，也不扩大成 benchmark。冻结两类描述性预警：

1. **字段冲突**：title 与 artist 的关系支持方向相反，且两边都超过弱效应门槛；
2. **跨路径冲突**：自由生成上下文支持已输出 pair，但独立的 title-to-artist 关系验证更支持另一个目录歌手，或相反。

已有自由生成档案包含 6 个目录 exact 和大量 catalog-conflict pair。迁移集将使用全部 6 个 exact，并按目录双源一致、上下文配额和预先规定的支持度排序选 6 个 conflict；不根据模型预警分数挑样本。

## 技术检查

- Hugging Face Job: `6a5232f0effc02a91cbd9881`
- Run ID: `20260711T121410Z_pilot`
- 总耗时：194 秒（排队 126 秒，运行 68 秒；探针约 20 秒）
- Pair 加权重建最大误差：`3.18e-7`
- 16-head 与完整 attention 对照最大误差：`0`
- 全部技术门槛：通过
