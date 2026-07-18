# Phase 2 Granite Primary 目录核验结果（v2）

## 一句话结论

Granite 在 290 个自由生成的歌名-歌手结果中，经过更保守的双目录规则后，得到 5 个互不重复的严格冲突标题和 22 个严格正确标题。这个数量足以说明“严格可复核的关系错误确实会出现”，但远远不够做预注册的确认性机制与纠错实验，因此必须运行已经冻结的 extension，不能直接进入 GPU 机制分析。

## 结果

- 输入关系：290 条，来自 60 个冻结的自由生成歌单。
- normalized title：223 个。
- strict conflict：5 条，去重后仍为 5 个标题 cluster。
- strict exact：37 条，去重后为 22 个标题 cluster。
- ambiguous：9 条。
- excluded：239 条。
- 网络 error：0 条；旧版遗留的 6 条错误已补齐。
- 原始请求证据：957 条，全部保留。

这里的 `excluded` 很多，不代表 239 条都是错误或正确。它只表示现有目录响应不足以满足我们特别严格的确认性定义，例如结果窗口可能截断、两个目录没有形成唯一共同 reference，或定向查询提示可能存在别名。

## 为什么旧版 8 个 conflict 变成 5 个

旧版结果只作为审计材料。v2 在公开修订后重新读取原始响应，并增加两条保守门槛：搜索窗口可能截断时不能把“没看到”当成不存在；定向 MusicBrainz 查询返回 canonical artist 时按潜在别名排除。因此 strict conflict 只会减少，不会因为修订而新增。

## 完整性检查

42 条 strict 行全部只使用自己链接的 raw responses 离线重跑 verifier。重算后的 label、reference artist、窗口完整性、alias audit 和 request IDs 与保存结果一致。290 条记录的协议、模型 revision、生成记录 SHA-256 和 record ID 也全部通过。

Primary raw evidence 为 235,389,061 bytes，SHA-256 为 `4ee997f...6564`。压缩快照为 33,996,356 bytes，解压后的 SHA-256 与原文件完全一致。v2 verifier SHA-256 为 `c9659b9...9867`，其方法代码在运行前已经发布到公开 Gist 历史 `6444e8d...`。

## 能说什么，不能说什么

可以说：在冻结 Granite、冻结提示和自由生成音乐推荐中，存在少量能被两套目录严格复核的 title-artist 关系冲突。

不能说：5/223 是模型真实幻觉率。大量行因证据门槛被排除，目录检索也不是现实世界的穷尽真相；本阶段更不能说明模型内部是否“知道自己错了”。

## 冻结决策

Primary 已有 290 个 parsed events，超过 240 门槛；但只有 5 个严格冲突标题，低于 40 门槛。根据预注册的 `either below` 规则，必须运行早已冻结的 20 个 extension 歌单。Primary 与 extension 合并后，如果严格冲突标题仍少于 30，确认性机制与纠错实验停止，并把低样本产出作为边界结果公开；不更换模型、不追加第三批样本、不放松目录规则。
