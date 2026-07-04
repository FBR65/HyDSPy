import sys
import logging
from dotenv import load_dotenv

from src.bootstrap import build  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> int:
    load_dotenv()

    logger.info("Initialising components ...")
    components = build(load_env=False)
    if not components.initialised:
        logger.error("Components were not fully initialised.")
        return 1

    sample_document = """
    The combination of Hypothetical Document Embeddings (HyDE) and DSPy is very
    effective in modern Retrieval-Augmented Generation (RAG) systems. HyDE
    improves semantic search by generating hypothetical documents that are
    closer to the actual query than the original question. DSPy optimises the
    prompts and the entire pipeline to achieve better results.

    HyDE works as follows: when a user asks a question, HyDE first generates a
    hypothetical document that could contain the answer. This hypothetical
    document is then used to search the vector database, which yields better
    matches than the original question.

    DSPy then optimises the entire pipeline through automatic prompt
    engineering. It learns from example data which prompts deliver the best
    performance and adapts them automatically.
    """

    logger.info("Processing document ...")
    processed_chunks = components.document_processor.process_document(
        "sample_doc_1", sample_document
    )
    logger.info("Document processed. %d chunks created.", len(processed_chunks))

    question = "How does HyDE work with DSPy?"

    logger.info("Answering question: %s", question)
    result = components.hyde_rag(question)
    print(f"Answer: {result.answer}")
    print(f"Context: {result.context}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
