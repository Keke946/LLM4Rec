> Historical material from the supplied archive; superseded by README.md, docs/PROJECT_STATUS.md and reports/reproduction/REPRODUCTION_REPORT.md. Historical metrics and improvements are not current verified claims.

# LLM4Rec 面试准备

## 开场介绍

LLM4Rec 是一个面向短视频推荐场景的多阶段混合推荐系统项目。项目主线使用 KuaiRec 真实公开数据，完整覆盖数据适配、时间切分、多路召回、候选融合、排序重排、离线评估和失败样本诊断。

需要强调的是：LLM 不直接生成推荐 item ID，也不决定推荐列表。推荐结果始终由本地召回、排序和业务重排链路产生；LLM 只用于 query rewrite、用户画像摘要、推荐解释和安全兜底。

## 60 秒版本

> 我做了一个基于 KuaiRec 的短视频推荐系统项目，重点不是单一模型，而是把推荐系统离线链路做完整。数据层把 KuaiRec、MovieLens、MIND 适配成统一 schema，并按用户时间顺序做 train/val/test 切分；召回层实现 Trend、ItemCF、BPR、LightGCN、TwoTower、SASRec、ContentFaiss、Metadata 等多路召回，再用 source-weighted RRF 做候选融合；排序层构建 candidate-level 样本，训练 GBDT、LambdaRank、DeepFM、DIN 等 ranker；评估层区分 candidate_pool_k 和 final_top_k，输出 Recall、NDCG、MRR、Coverage、TailRatio、candidate_positive_recall 和 failure cases。最后我把 LLM 放在解释层，并加候选集校验，避免幻觉 item 进入推荐结果。

## 简历 bullet

> 构建 LLM4Rec 多阶段混合推荐系统，基于 KuaiRec 真实短视频数据完成数据适配、时间切分、多路召回、RRF 候选融合、Learning-to-Rank 精排、曝光感知负采样、业务重排和离线诊断；实现 Trend、ItemCF、BPR、LightGCN、TwoTower、SASRec、ContentFaiss、Metadata 等召回器，以及 GBDT、LambdaRank、DeepFM、DIN 等排序模型；通过 candidate_positive_recall、positive_injection_rate、cold/tail bucket 和 failure case taxonomy 定位召回覆盖与排序退化问题；实验中 GBDT top50 head rerank 相比 retrieval-only 实现 Recall@10 约 115% 相对提升、NDCG@10 约 168% 相对提升、MRR@10 约 236% 相对提升，并保留深层候选召回能力。

## 重点追问

### 1. 这个项目是不是 toy？

不是纯 toy，但也不能说成生产级平台。它具备真实数据 adapter、统一 schema、时间切分、防泄漏、多路召回、候选融合、排序模型、业务重排、评估矩阵、失败样本诊断、测试和报告。限制是没有线上 A/B、feature store、模型注册、监控和分布式训练。最准确的定位是“可复现实验型推荐系统工程项目”。

### 2. 为什么要做多路召回？

不同召回器覆盖不同信号。Trend 适合短视频热点和近期行为，ItemCF/BPR 覆盖协同行为，LightGCN 建模图传播，TwoTower 支持向量召回，SASRec 建模行为序列，ContentFaiss/Metadata 给冷启动和长尾 item 提供侧信息路径。多路召回后用 RRF 融合，并把 source_rank、source_count、retrieval_score 继续传给 ranker。

### 3. 为什么 ranker 不直接全量重排？

full rerank 可能破坏强召回在深层位置的排序，尤其当 ranker 训练样本和线上候选分布不完全一致时。所以项目同时做了 retrieval_only、ranker_only、retrieval_ranker_fusion、ranker_top_50、ranker_top_100 的 ablation。最终 head rerank 更稳：只重排头部候选，提升 Top10 质量，同时保留 Recall@100。

### 4. candidate_positive_recall 是什么？

它衡量 split positives 在排序前是否已经被召回。如果正样本主要靠 injected positives 补进训练集，ranker 即使学得很好，也不能证明召回系统有效。因此项目显式记录 candidate_positive_recall 和 positive_injection_rate，用来审计候选池质量。

### 5. 负采样为什么重要？

推荐系统中的负样本不是简单的“没点过”。曝光但低 watch ratio 的样本更接近真实负反馈，retrieved hard negative 更接近排序难例，但难负样本过强也可能导致模型退化。因此项目对 random、exposed low-watch、retrieved hard、mixed negative 做了消融。

### 6. LLM 在项目里有什么作用？

LLM 不是推荐模型，不生成 item ID。它只负责 query rewrite、profile summary、tool calling 和推荐解释。解释层必须基于本地候选列表和 metadata，如果出现候选集外 item 或 unsupported claim，就回退模板解释。

## 不能乱说的地方

- 不说线上 CTR/watch-time/留存提升。
- 不说生产级大规模推荐平台。
- 不说 SOTA。
- 不说没有实验支撑的 Recall 90%。
- 不说 LLM 直接生成推荐列表。
- 不把 KuaiRec 子集实验说成全量线上业务验证。

## 可以主动强调的地方

- 这个项目体现的是推荐系统离线工程闭环。
- 有真实数据、有时间切分、有防泄漏、有多路召回、有排序和诊断。
- 指标不只看最终 Recall@K，还拆解候选池召回、长尾/冷启动、失败原因和 ranker 对不同深度的影响。
- 话术上要稳：强包装技术复杂度，不夸不存在的线上收益。
