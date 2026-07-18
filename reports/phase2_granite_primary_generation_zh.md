# Phase 2 Granite 正式生成阶段报告

## 当前结论

预注册的 primary generation 已经完成并通过技术 gate。60 个歌单全部有输出，
共解析出 290 条 title-artist-reason 记录，高于 240 条门槛。现在只能说明正式
数据生成成功，不能据此判断有多少歌曲关系正确或错误。

## 结果概览

- 模型：`ibm-granite/granite-4.1-3b-base`
- 固定 revision：`dacb9cb9157bec98e99b09f285c92a4d58405c96`
- 场景：12 个冻结场景
- 随机种子：5 个冻结种子
- 预期与实际歌单数：60 / 60
- 非空输出：60 / 60
- 解析事件：290 条
- 不同标准化歌名：223 个
- 有重复事件的歌名 cluster：25 个
- hook 端点最大误差：`0.0`
- GPU 运行时间：803 秒

其中两个生成结果没有通过冻结 parser，已按原样保留为零条解析事件；没有补写、
手工修复或重跑这两个 seed。这样避免只修“不好看”的样本。

## 数据完整性

远端输出被压缩成 3 个编号分块。三个分块均已取回、顺序连续，并解码为 60 条
原始 completion。恢复后的 artifact、解析后的 290 条 catalog 输入、正式 bundle
和 exact runner 都记录了 SHA-256。远端脚本哈希与提交前公开 Gist 中的文件一致。

## 下一道 gate

解析事件数已经超过 240，因此不会因“解析量不足”触发 extension。但协议使用
逻辑或：如果双目录核验后少于 40 个不同的严格 conflict title cluster，仍必须运行
额外 20 个注册歌单。这个判断必须等待 MusicBrainz 与 Apple 的完整证据归档。

Job: https://huggingface.co/jobs/REDACTED/6a54cc73e4a4e82c0b591404
