> Historical material from the supplied archive; superseded by README.md, docs/PROJECT_STATUS.md and reports/reproduction/REPRODUCTION_REPORT.md. Historical metrics and improvements are not current verified claims.

# LLM4Rec Interview Q&A

## 1. 这个项目一句话怎么介绍？

LLM4Rec 是一个真实数据驱动的多阶段混合推荐系统项目，覆盖数据适配、特征构建、多路召回、候选融合、深度排序、序列召回、业务重排、离线评估、实验矩阵、FastAPI 服务和 LLM 解释安全防护。项目主线是 KuaiRec/KuaiRand 视频推荐，MovieLens 只做 smoke test，MIND 作为新闻/NLP 推荐路径。

## 2. 这个项目是不是 toy？

不是纯 toy，但也不能说成生产级平台。它已经具备推荐系统核心闭环：真实数据 adapter、统一 schema、时间切分、防泄漏、多路召回、DeepFM/DIN 排序、SASRec 序列模型、业务 rerank、candidate pool 级评估、artifact 管理、测试和报告。限制是还没有线上 A/B、feature store、模型注册、监控和分布式训练。所以面试里应定位为“可复现实验型推荐系统工程项目”。

## 3. 项目用了哪些模型？

召回层：

- Popular：热门正反馈 baseline。
- ContentFaiss：SentenceTransformer + FAISS 内容召回，失败时回退 TF-IDF。
- ItemCF：基于 item 共现的协同过滤。
- BPR-MF：pairwise BPR loss 的矩阵分解召回。
- TwoTower：用户塔 = user embedding + pooled history embedding，item tower = item embedding，支持导出 item embedding 和可选 FAISS index。
- LightGCN：user-item 二部图归一化传播，BPR loss 训练。
- SASRec：序列推荐基线，使用 item embedding、position embedding 和 Transformer self-attention 做 next-item prediction。

排序层：

- EmbeddingMLP：稀疏特征 embedding 后接 MLP。
- DeepFM：一阶线性项、FM 二阶交叉项、deep 高阶交互。
- DIN：候选 item 对用户历史行为做 attention。

LLM 不是推荐模型，只做 query rewrite、profile summary、tool calling 和已选 item 的解释。

## 4. 为什么要加 SASRec？

短视频推荐强依赖 session 和行为顺序。DIN 能做候选相关历史 attention，但它本质仍是排序模型；SASRec 用 self-attention 建模用户历史序列中的转移关系，更适合作为 sequential retriever / sequence baseline。面试中可以说：DIN 解决候选与历史相关性，SASRec 解决 next-item 序列预测。

## 5. 为什么 ranker 不能直接覆盖 retrieval score？

弱 ranker 直接用 raw logit 覆盖召回分数会破坏强召回排序。项目提供三种模式：

```text
retrieval_only
ranker_only
retrieval_ranker_fusion
```

默认融合公式：

```text
final_score = alpha * normalized_retrieval_score + beta * sigmoid(ranker_logit)
```

默认 `alpha=0.7, beta=0.3`，并在报告里保留 retrieval-only / ranker-only ablation，不隐藏 ranker degradation。

## 6. 排序模型用了哪些特征？

稀疏特征包括 user、item、author、category、tag、duration bucket、retrieval source。序列特征是用户历史正反馈 item 序列。稠密特征包括 retrieval_score、source_rank、source_count、item_popularity、user_activity_level、history_len。这样 ranker 不会丢失召回阶段的信息。

## 7. 业务 reranker 是什么？

业务 reranker 是模型排序之后的轻量规则层，支持：

- diversity：提升类目多样性，限制同类目/同作者过度集中。
- diversity_tail：在 diversity 基础上给长尾 item 温和 boost。
- none：不做业务重排。

报告会对比 NDCG@10、Coverage、Diversity、TailRatio 的 before/after。

## 8. 如何避免数据泄漏？

项目按用户时间顺序切分 train/val/test，test 只代表未来行为。MIND 用 `user_id + impression_id` 做 group-aware split，避免同一次曝光的 clicked/non-clicked 行被拆到不同 split。test 评估默认过滤 train+val 正反馈，因为它们在 test 时刻已经是历史已看内容。

## 9. candidate_pool_k 和 final_top_k 为什么要分开？

如果只生成 top10，却计算 Recall@50/100，指标是错的。项目先生成 `candidate_pool_k=500` 的候选池，ranker/reranker 在候选池上处理，metrics 按 cutoffs 计算，`final_top_k` 只表示最终展示数量。

## 10. LLM 在项目里怎么可量化？

LLM 不生成 item_id。当前可量化实验是 explanation safety：检测 hallucinated item ID rate、unsupported claim rate、template fallback rate。若 LLM 输出候选集外 item_id，系统回退模板解释。后续也可以做 raw query vs rule rewrite vs LLM rewrite 的 ContentFaiss Recall 对比。

## 11. retriever 为什么不直接 pickle 整个对象？

直接 pickle 带 pandas DataFrame 的对象有跨版本风险，也会保存大对象。现在 retriever artifact 是目录式：

```text
config.json
item_ids.json
train_seen_by_user.json
state.pkl / state.pt / state.npz
```

Popular/ItemCF 保存最小字典，BPR/TwoTower/LightGCN/SASRec 保存 `state_dict` 和 ID 映射。加载失败时 evaluator/recommender 会 warning 并从 train data refit。

## 12. 这个项目还能怎么继续生产化？

可以补 feature store、ANN 检索服务、模型注册、在线 A/B、监控告警、分布式训练、更细的冷启动/长尾分桶分析，以及真实 KuaiRec 全量实验结果。
