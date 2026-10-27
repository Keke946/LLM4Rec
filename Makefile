.PHONY: test smoke-movielens eval-movielens eval-kuairec probe-kuairec-gbdt train-kuairec-lambdarank eval-kuairec-lambdarank ablate-kuairec-ltr report-kuairec-v2 diagnose-kuairec validate-kuairec health serve

PYTHON ?= python
MOVIELENS_RETRIEVERS := popular,trend,itemcf,bpr,lightgcn,two_tower,sasrec,content_faiss,metadata
KUAIREC_RETRIEVERS := popular,trend,itemcf,bpr,lightgcn,two_tower,sasrec,content_faiss,metadata
KUAIREC_RAW_DIR ?= data/raw/kuairec_official/extracted/KuaiRec 2.0/data
KUAIREC_SOURCE_WEIGHTS := trend=1.6,sasrec=1.1,lightgcn=0.9,itemcf=0.8,bpr=0.7,two_tower=0.7,content_faiss=0.5,popular=0.4,metadata=0.25
KUAIREC_TREND_ARGS := --trend_half_life_days 2.0 --trend_category_weight 0.4 --trend_tag_weight 0.2 --trend_popularity_floor 0.02 --trend_candidate_cache_k 2000
KUAIREC_LTR_FEATURES := data/features/kuairec_trend1500_ltr
KUAIREC_LTR_MODELS := models/kuairec_trend1500_ltr

test:
	$(PYTHON) -m unittest discover -s tests

smoke-movielens:
	$(PYTHON) scripts/prepare_dataset.py --dataset movielens --raw_dir examples/sample_data/movielens --output_dir data/processed/movielens
	$(PYTHON) scripts/build_features.py --dataset movielens --processed_dir data/processed/movielens --output_dir data/features/movielens --max_history_len 20
	$(PYTHON) scripts/build_ranker_candidates.py --dataset movielens --processed_dir data/processed/movielens --features_dir data/features/movielens --retriever_dir models/movielens/retrievers --retrievers $(MOVIELENS_RETRIEVERS) --candidate_pool_k 500 --max_history_len 20 --epochs 1 --batch_size 512 --embedding_dim 16
	$(PYTHON) scripts/train_ranker.py --dataset movielens --model embedding_mlp --features_dir data/features/movielens --output_dir models/movielens/embedding_mlp --epochs 1 --batch_size 512 --embedding_dim 16 --max_history_len 20 --patience 1
	$(PYTHON) scripts/train_ranker.py --dataset movielens --model deepfm --features_dir data/features/movielens --output_dir models/movielens/deepfm --epochs 1 --batch_size 512 --embedding_dim 16 --max_history_len 20 --patience 1
	$(PYTHON) scripts/train_ranker.py --dataset movielens --model din --features_dir data/features/movielens --output_dir models/movielens/din --epochs 1 --batch_size 512 --embedding_dim 16 --max_history_len 20 --patience 1
	$(PYTHON) scripts/run_experiment_matrix.py --dataset movielens --config configs/movielens.yaml --retrievers popular,itemcf --rankers none,deepfm --rerank_modes retrieval_ranker_fusion --rerankers none --filter_seen_splits train
	$(PYTHON) scripts/diagnose_candidate_recall.py --dataset movielens --processed_dir data/processed/movielens --features_dir data/features/movielens --output_dir reports/movielens --split test

eval-movielens:
	$(PYTHON) scripts/prepare_dataset.py --dataset movielens --raw_dir data/raw/movielens --output_dir data/processed/movielens
	$(PYTHON) scripts/build_features.py --dataset movielens --processed_dir data/processed/movielens --output_dir data/features/movielens --max_history_len 20
	$(PYTHON) scripts/build_ranker_candidates.py --dataset movielens --processed_dir data/processed/movielens --features_dir data/features/movielens --retriever_dir models/movielens/retrievers --retrievers $(MOVIELENS_RETRIEVERS) --candidate_pool_k 500 --max_history_len 20
	$(PYTHON) scripts/train_ranker.py --dataset movielens --model embedding_mlp --features_dir data/features/movielens --output_dir models/movielens/embedding_mlp --epochs 3 --batch_size 512 --embedding_dim 16 --max_history_len 20 --patience 2
	$(PYTHON) scripts/train_ranker.py --dataset movielens --model deepfm --features_dir data/features/movielens --output_dir models/movielens/deepfm --epochs 3 --batch_size 512 --embedding_dim 16 --max_history_len 20 --patience 2
	$(PYTHON) scripts/train_ranker.py --dataset movielens --model din --features_dir data/features/movielens --output_dir models/movielens/din --epochs 3 --batch_size 512 --embedding_dim 16 --max_history_len 20 --patience 2
	$(PYTHON) scripts/run_experiment_matrix.py --dataset movielens --config configs/movielens.yaml
	$(PYTHON) scripts/diagnose_candidate_recall.py --dataset movielens --processed_dir data/processed/movielens --features_dir data/features/movielens --output_dir reports/movielens --split test

eval-kuairec:
	$(PYTHON) scripts/prepare_dataset.py --dataset kuairec --raw_dir "$(KUAIREC_RAW_DIR)" --output_dir data/processed/kuairec --kuairec_matrix big --positive_watch_ratio_threshold 1.0 --sample_users 1000 --max_interactions 300000 --max_interactions_strategy random --max_interactions_seed 42 --no_sample_fallback --min_users 1000 --min_items 5000 --min_interactions 100000
	$(PYTHON) scripts/build_features.py --dataset kuairec --processed_dir data/processed/kuairec --output_dir data/features/kuairec --max_history_len 50
	$(PYTHON) scripts/build_ranker_candidates.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec --retriever_dir models/kuairec/retrievers --retrievers $(KUAIREC_RETRIEVERS) --candidate_pool_k 500 --max_history_len 50 --epochs 10 --batch_size 1024 --embedding_dim 32 --source_weights $(KUAIREC_SOURCE_WEIGHTS) $(KUAIREC_TREND_ARGS)
	$(PYTHON) scripts/train_ranker.py --dataset kuairec --model embedding_mlp --features_dir data/features/kuairec --output_dir models/kuairec/embedding_mlp --epochs 10 --batch_size 1024 --embedding_dim 32 --max_history_len 50 --patience 2
	$(PYTHON) scripts/train_ranker.py --dataset kuairec --model deepfm --features_dir data/features/kuairec --output_dir models/kuairec/deepfm --epochs 10 --batch_size 1024 --embedding_dim 32 --max_history_len 50 --patience 2
	$(PYTHON) scripts/train_ranker.py --dataset kuairec --model din --features_dir data/features/kuairec --output_dir models/kuairec/din --epochs 10 --batch_size 1024 --embedding_dim 32 --max_history_len 50 --patience 2
	$(PYTHON) scripts/run_experiment_matrix.py --dataset kuairec --config configs/kuairec.yaml
	$(PYTHON) scripts/diagnose_candidate_recall.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec --output_dir reports/kuairec --split test

probe-kuairec-gbdt:
	$(PYTHON) scripts/build_ranker_candidates.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec --output_dir data/features/kuairec_trend1500_ltr --retriever_dir models/kuairec/retrievers --retrievers trend --candidate_pool_k 1500 --retriever_top_k 1500 --splits val,test --max_history_len 50 $(KUAIREC_TREND_ARGS)
	$(PYTHON) scripts/build_temporal_ranker_train.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec --output_dir data/features/kuairec_trend1500_ltr --retrievers trend --candidate_pool_k 1500 --retriever_top_k 1500 --holdout_quantile 0.8 --max_history_len 50 $(KUAIREC_TREND_ARGS)
	$(PYTHON) scripts/train_gbdt_ranker.py --dataset kuairec --features_dir data/features/kuairec_trend1500_ltr --output_dir models/kuairec_trend1500_ltr/gbdt --train_split train --val_split val --max_iter 120 --learning_rate 0.07 --max_leaf_nodes 31 --l2_regularization 0.05 --max_train_rows 700000 --max_val_rows 500000
	$(PYTHON) scripts/run_experiment_matrix.py --dataset kuairec --config configs/kuairec.yaml --features_dir data/features/kuairec_trend1500_ltr --ranker_root models/kuairec_trend1500_ltr --output_dir reports/kuairec_trend1500_headrerank_probe --retrievers trend --rankers none,gbdt --rerank_modes ranker_top_50,ranker_top_100 --rerankers none --candidate_pool_k 1500 --cutoffs 5,10,20,50,100 --final_top_k 10 --skip_auto_train_rankers
	$(PYTHON) scripts/diagnose_candidate_recall.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec_trend1500_ltr --output_dir reports/kuairec_trend1500_ltr --split test

train-kuairec-lambdarank:
	$(PYTHON) scripts/train_lambdarank_ranker.py --dataset kuairec --features_dir $(KUAIREC_LTR_FEATURES) --output_dir $(KUAIREC_LTR_MODELS)/lambdarank --train_split train --val_split val --n_estimators 120 --learning_rate 0.05 --num_leaves 31 --max_train_rows 700000 --max_val_rows 500000

eval-kuairec-lambdarank:
	$(PYTHON) scripts/run_experiment_matrix.py --dataset kuairec --config configs/kuairec.yaml --features_dir $(KUAIREC_LTR_FEATURES) --ranker_root $(KUAIREC_LTR_MODELS) --output_dir reports/kuairec_lambdarank_eval --retrievers trend --rankers none,lambdarank --rerank_modes ranker_only,ranker_top_50,ranker_top_100 --rerankers none --candidate_pool_k 1500 --cutoffs 5,10,20,50,100 --final_top_k 100 --skip_auto_train_rankers

ablate-kuairec-ltr:
	$(PYTHON) scripts/run_ltr_ablation.py --dataset kuairec --config configs/kuairec_ltr_ablation.yaml

report-kuairec-v2:
	$(PYTHON) scripts/run_ltr_ablation.py --dataset kuairec --config configs/kuairec_ltr_ablation.yaml --skip_training

diagnose-kuairec:
	$(PYTHON) scripts/diagnose_candidate_recall.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec --output_dir reports/kuairec --split test

validate-kuairec:
	$(PYTHON) scripts/validate_experiment_artifacts.py --dataset kuairec --processed_dir data/processed/kuairec --features_dir data/features/kuairec --reports_dir reports/kuairec --require_real --min_recall100 0.04

health:
	$(PYTHON) -m compileall -q llm4rec scripts run_chat.py Recall_and_Rank.py test_env.py
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) scripts/check_secrets.py

serve:
	$(PYTHON) scripts/serve_api.py

.PHONY: download-kuairec reproduce-kuairec reproduce-kuairec-historical verify-retrievers
download-kuairec:
	$(PYTHON) scripts/download_kuairec.py

reproduce-kuairec:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 $(PYTHON) scripts/reproduce_kuairec.py --protocol strict

reproduce-kuairec-historical:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 $(PYTHON) scripts/reproduce_kuairec.py --protocol historical

verify-retrievers:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 $(PYTHON) scripts/verify_retrievers.py
