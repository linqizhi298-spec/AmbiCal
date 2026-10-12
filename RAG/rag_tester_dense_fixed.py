"""Dedicated entry point for Dense/Semantic retrieval with a fixed k."""

import logging

from rag_tester import RagTester as BaseRagTester


class RagTester(BaseRagTester):
    """RAG tester restricted to Dense/Semantic + fixed-k configurations."""

    def __init__(self, config_path: str = "config.json") -> None:
        super().__init__(config_path=config_path)
        if self.adaptive_k_enabled:
            raise ValueError(
                "rag_tester_dense_fixed requires adaptive_k_enabled=false "
                "or the option to be absent."
            )
        if self.retrieval_algorithms_to_test != ["embedding"]:
            raise ValueError(
                "rag_tester_dense_fixed requires exactly one 'embedding' "
                "retrieval algorithm."
            )
        if not isinstance(self.num_retrieved_docs, int) or self.num_retrieved_docs < 1:
            raise ValueError("Fixed k must be a positive integer.")


def start_rag_tests(config_path: str = "config.json") -> bool:
    """Run the dedicated Dense/Semantic fixed-k experiment entry point."""
    logging.info(
        "--- Starting Dense fixed-k tests "
        f"(config: {config_path}) ---"
    )
    tester = RagTester(config_path=config_path)
    tester.run_tests()
    logging.info("--- Dense fixed-k tests completed successfully ---")
    return True


if __name__ == "__main__":
    start_rag_tests()
