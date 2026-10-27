from __future__ import annotations

from llm4rec.recommender import LLM4RecRecommender


def main() -> None:
    recommender = LLM4RecRecommender()
    results = recommender.recommend(["football", "sports", "goals"], k=5)
    for item in results:
        print(
            f"{item['video_id']}: {item['title']} | "
            f"tags={item['tags']} | score={item['score']:.4f}"
        )
        print(f"  reason: {item['reason']}")


if __name__ == "__main__":
    main()
