import logging
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.batch_retrieval_grader import batch_retrieval_grader
from graph.state import GraphState

logger = logging.getLogger("agentic_rag.grade")


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the user question.
    Uses batch grading to reduce API calls (grade all documents in one call).
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): Filtered out irrelevant documents and updated use_web_search state.
    """
    print("---GRADE DOCUMENTS---")
    logger.info("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            "documents": [],
            "use_web_search": True,
            "question": question,
            "user_id": state.get("user_id"),
            "chat_history": state.get("chat_history", []),
        }

    # Use batch grading for 2+ documents to reduce API calls
    # For single document, use individual grader (simpler and faster)
    if len(documents) >= 2:
        print(f"---BATCH GRADING {len(documents)} DOCUMENTS---")
        
        # Format documents with indices
        documents_text = "\n\n".join([
            f"Document {idx}:\n{doc.page_content[:1000]}"  # Limit length to avoid token limits
            for idx, doc in enumerate(documents)
        ])
        
        try:
            result = batch_retrieval_grader.invoke({
                "question": question,
                "documents": documents_text
            })
            
            relevant_indices = set(result.relevant_document_indices)
            print(f"---BATCH GRADING RESULT: {len(relevant_indices)}/{len(documents)} documents relevant---")
            print(f"---REASONING: {result.reasoning[:100]}...---")
            
            filtered_documents = [
                doc for idx, doc in enumerate(documents)
                if idx in relevant_indices
            ]
            
            # If no documents are relevant, trigger web search
            use_web_search = len(filtered_documents) == 0
            
            # Log individual results
            for idx, doc in enumerate(documents):
                if idx in relevant_indices:
                    print(f"---DOCUMENT {idx} IS RELEVANT---")
                else:
                    print(f"---DOCUMENT {idx} IS NOT RELEVANT---")
            
        except Exception as e:
            print(f"---BATCH GRADING ERROR: {e}, FALLING BACK TO INDIVIDUAL GRADING---")
            # Fallback to individual grading on error
            filtered_documents = []
            use_web_search = False
            for doc in documents:
                result = retrieval_grader.invoke(
                    {"question": question, "document": doc.page_content}
                )
                grade = result.binary_score
                if grade.lower() == "yes":
                    print("---DOCUMENT IS RELEVANT---")
                    filtered_documents.append(doc)
                else:
                    print("---DOCUMENT IS NOT RELEVANT---")
                    use_web_search = True
    else:
        # Single document: use individual grader
        print("---GRADING SINGLE DOCUMENT---")
        doc = documents[0]
        result = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = result.binary_score
        if grade.lower() == "yes":
            print("---DOCUMENT IS RELEVANT---")
            filtered_documents = [doc]
            use_web_search = False
        else:
            print("---DOCUMENT IS NOT RELEVANT---")
            filtered_documents = []
            use_web_search = True
    
    if not filtered_documents:
        use_web_search = True

    return {
        "documents": filtered_documents,
        "use_web_search": use_web_search,
        "question": question,
        "user_id": state.get("user_id"),
        "lesson_id": state.get("lesson_id"),  # Preserve lesson_id
        "is_platform_question": state.get("is_platform_question", False),  # Preserve platform question flag
        "chat_history": state.get("chat_history", []),
    }
