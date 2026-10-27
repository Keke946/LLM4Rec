from __future__ import annotations

import argparse

from dotenv import load_dotenv

from llm4rec.recommender import LLM4RecRecommender
from tools.recommend_tool import recommend_videos


def print_recommendations(results: list[dict]) -> None:
    if not results:
        print("No suitable recommendations found. Try another query.")
        return

    for idx, item in enumerate(results, start=1):
        if "video_id" in item:
            tags = ", ".join(item.get("tags", []))
            print(f"{idx}. {item['title']} ({item['video_id']})")
            print(f"   score={item['score']:.4f} | tags={tags}")
            print(f"   reason={item['reason']}")
        else:
            sources = ",".join(item.get("retrieval_sources", []))
            print(f"{item.get('rank', idx)}. {item['title']} ({item['item_id']})")
            print(f"   score={item['score']:.6f} | sources={sources}")
            print(f"   explanation={item['explanation']}")
            if "debug" in item:
                print(f"   debug={item['debug']}")


def run_offline(k: int) -> None:
    print("LLM4Rec local demo mode. Type exit/quit/q to stop.")
    while True:
        user_input = input("\ninterest> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        print_recommendations(recommend_videos(user_input, k=k))


def run_dataset_offline(
    dataset: str,
    k: int,
    retrievers: list[str],
    ranker: str,
    debug: bool,
    candidate_pool_k: int,
    rerank_mode: str,
    retrieval_weight: float,
    ranker_weight: float,
    reranker: str,
) -> None:
    recommender = LLM4RecRecommender(
        dataset=dataset,
        processed_dir=f"data/processed/{dataset}",
        retriever_dir=f"models/{dataset}/retrievers",
    )
    print(f"LLM4Rec dataset mode: {dataset}. Type exit/quit/q to stop.")
    while True:
        user_id = input("\nuser_id> ").strip()
        if user_id.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        query = input("query optional> ").strip() or None
        results = recommender.recommend(
            user_id=user_id,
            query=query,
            top_k=k,
            candidate_pool_k=candidate_pool_k,
            retrievers=retrievers,
            ranker=ranker,
            debug=debug,
            rerank_mode=rerank_mode,
            retrieval_weight=retrieval_weight,
            ranker_weight=ranker_weight,
            reranker=reranker,
        )
        print_recommendations(results)


def run_agent() -> None:
    from agents.recommender_agent import get_recommender_agent

    agent = get_recommender_agent()
    print("LLM4Rec agent mode. Type exit/quit/q to stop.")
    while True:
        user_input = input("\ninterest> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        response = agent.invoke({"input": user_input})
        print(response.get("output", response))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM4Rec recommender.")
    parser.add_argument("--agent", action="store_true", help="Use the LangChain/OpenAI agent.")
    parser.add_argument("--offline", action="store_true", help="Use the local recommender only.")
    parser.add_argument("--dataset", default=None, help="Processed dataset name.")
    parser.add_argument("--retrievers", default="popular,itemcf,content_faiss", help="Comma-separated retrievers.")
    parser.add_argument("--ranker", default="deepfm", help="Ranker name for dataset mode.")
    parser.add_argument("--candidate_pool_k", type=int, default=200, help="Candidate pool before ranker rerank.")
    parser.add_argument(
        "--rerank_mode",
        default="retrieval_ranker_fusion",
        choices=["retrieval_only", "ranker_only", "retrieval_ranker_fusion"],
        help="How to combine retrieval and ranker scores.",
    )
    parser.add_argument("--retrieval_weight", type=float, default=0.7, help="Retrieval score weight.")
    parser.add_argument("--ranker_weight", type=float, default=0.3, help="Ranker probability weight.")
    parser.add_argument(
        "--reranker",
        default="none",
        choices=["none", "diversity", "diversity_tail"],
        help="Optional business reranker.",
    )
    parser.add_argument("--debug", action="store_true", help="Print retrieval/ranking debug fields.")
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations.")
    args = parser.parse_args()

    load_dotenv()
    if args.dataset:
        run_dataset_offline(
            dataset=args.dataset,
            k=args.k,
            retrievers=[name.strip() for name in args.retrievers.split(",") if name.strip()],
            ranker=args.ranker,
            debug=args.debug,
            candidate_pool_k=args.candidate_pool_k,
            rerank_mode=args.rerank_mode,
            retrieval_weight=args.retrieval_weight,
            ranker_weight=args.ranker_weight,
            reranker=args.reranker,
        )
        return
    if args.agent and not args.offline:
        run_agent()
    else:
        run_offline(k=args.k)


if __name__ == "__main__":
    main()
