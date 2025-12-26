import logging

from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH, GREETING, REJECT
from graph.chains import hallucination_grader, answer_grader, question_router, combined_grader
from graph.nodes import (
    generate, 
    grade_documents, 
    retrieve, 
    web_search, 
    greeting, 
    _is_greeting,
    reject_unrelated_question,
    _is_unrelated_question_simple,
)

logger = logging.getLogger("agentic_rag.graph")


# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    logger.info("---ASSESS GRADED DOCUMENTS---")

    if state["use_web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, GO TO WEB---")
        logger.info("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, GO TO WEB---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        logger.info("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState):
    """
    Combined grader: Check both hallucination and answer relevance in one API call.
    Reduces from 2 calls to 1 call.
    Prevents infinite loops by limiting regeneration attempts.
    """
    print("---CHECK HALLUCINATIONS AND ANSWER RELEVANCE (COMBINED)---")
    logger.info("---CHECK HALLUCINATIONS AND ANSWER RELEVANCE (COMBINED)---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    regeneration_count = state.get("regeneration_count") or 0
    web_search_count = state.get("web_search_count") or 0
    
    # Prevent infinite loops: if already tried web search multiple times, accept answer
    if web_search_count >= 2:
        msg = f"---DECISION: WEB SEARCH LIMIT REACHED ({web_search_count} attempts), ACCEPTING ANSWER---"
        print(msg)
        logger.info(msg)
        return "useful"  # Accept the answer to prevent infinite loop
    
    # Prevent infinite loops: if no documents or already regenerated too many times, fallback
    if not documents:
        msg = "---DECISION: NO DOCUMENTS AVAILABLE, FALLBACK TO WEB SEARCH---"
        print(msg)
        logger.info(msg)
        return "not_useful"  # Fallback to web search
    
    # Increased limit and more aggressive fallback to prevent recursion
    if regeneration_count >= 3:
        msg = f"---DECISION: REGENERATION LIMIT REACHED ({regeneration_count} attempts), ACCEPTING ANSWER---"
        print(msg)
        logger.info(msg)
        return "useful"  # Accept the answer to prevent infinite loop
    
    # If already regenerated once and still not grounded, go to web search instead
    if regeneration_count >= 1 and not generation:
        msg = f"---DECISION: ALREADY REGENERATED ONCE ({regeneration_count} attempts), FALLBACK TO WEB SEARCH---"
        print(msg)
        logger.info(msg)
        return "not_useful"  # Fallback to web search instead of regenerating again
    
    documents_text = "\n\n-----\n\n".join([doc.page_content for doc in documents])

    try:
        # Use combined grader to check both in one call
        result = combined_grader.invoke({
            "question": question,
            "documents": documents_text,
            "generation": generation
        })
        
        is_grounded = result.is_grounded
        addresses_question = result.addresses_question
        
        msg1 = f"---COMBINED GRADING RESULT: grounded={is_grounded}, addresses_question={addresses_question}---"
        msg2 = f"---REASONING: {result.reasoning[:100]}...---"
        print(msg1)
        print(msg2)
        logger.info(msg1)
        logger.info(msg2)
        
        if is_grounded and addresses_question:
            msg = "---DECISION: ANSWER IS GROUNDED AND ADDRESSES QUESTION---"
            print(msg)
            logger.info(msg)
            return "useful"
        elif not is_grounded:
            msg = "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---"
            print(msg)
            logger.info(msg)
            # If no documents or already regenerated once, fallback to web search instead of regenerating
            if not documents or regeneration_count >= 1:
                msg2 = "---DECISION: FALLBACK TO WEB SEARCH INSTEAD OF REGENERATING---"
                print(msg2)
                logger.info(msg2)
                return "not_useful"
            # Only allow regeneration if we haven't tried yet
            if regeneration_count == 0:
                return "not_supported"
            else:
                msg2 = "---DECISION: REGENERATION ALREADY ATTEMPTED, FALLBACK TO WEB SEARCH---"
                print(msg2)
                logger.info(msg2)
                return "not_useful"
        else:
            msg = "---DECISION: ANSWER DOES NOT ADDRESS THE USER QUESTION---"
            print(msg)
            logger.info(msg)
            # If already tried web search once, accept answer to prevent loop
            if web_search_count >= 1:
                msg2 = f"---DECISION: ALREADY TRIED WEB SEARCH ({web_search_count} attempts), ACCEPTING ANSWER---"
                print(msg2)
                logger.info(msg2)
                return "useful"
            return "not_useful"
            
    except Exception as e:
        msg = f"---COMBINED GRADER ERROR: {e}, FALLING BACK TO SEPARATE GRADERS---"
        print(msg)
        logger.warning(msg, exc_info=True)
        # Fallback to separate graders on error
        score = hallucination_grader.invoke(
            {"documents": documents_text, "generation": generation}
        )
        if hallucination_grade := score.binary_score:
            msg1 = "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---"
            msg2 = "---CHECK ANSWER---"
            print(msg1)
            print(msg2)
            logger.info(msg1)
            logger.info(msg2)
            score = answer_grader.invoke({"question": question, "generation": generation})
            if answer_grade := score.binary_score:
                msg = "---DECISION: ANSWER ADDRESSES THE USER QUESTION---"
                print(msg)
                logger.info(msg)
                return "useful"
            else:
                msg = "---DECISION: ANSWER DOES NOT ADDRESS THE USER QUESTION---"
                print(msg)
                logger.info(msg)
                return "not_useful"
        else:
            msg = "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---"
            print(msg)
            logger.info(msg)
            # If no documents or already regenerated, fallback to web search
            if not documents or regeneration_count >= 1:
                msg2 = "---DECISION: FALLBACK TO WEB SEARCH INSTEAD OF REGENERATING---"
                print(msg2)
                logger.info(msg2)
                return "not_useful"
            return "not_supported"


def route_question(state: GraphState):
    print("---ROUTE QUESTION---")
    logger.info("---ROUTE QUESTION---")
    question = state["question"]
    
    # Check greeting/chit-chat FIRST (avoid unnecessary resource consumption)
    if _is_greeting(question):
        msg = "---DECISION: ROUTE QUESTION TO GREETING (CHIT-CHAT)---"
        print(msg)
        logger.info(msg)
        return GREETING
    
    # Early rejection: Check if question is obviously unrelated (fast pattern check)
    if _is_unrelated_question_simple(question):
        msg = "---DECISION: QUESTION IS UNRELATED TO COURSES (PATTERN CHECK)---"
        print(msg)
        logger.info(msg)
        return REJECT

    try:
        source = question_router.invoke({"question": question})
    except Exception as e:
        msg = f"---ERROR: ROUTER FAILED ({type(e).__name__}: {e}), DEFAULTING TO RAG---"
        print(msg)
        logger.warning(msg, exc_info=True)
        return RETRIEVE

    # Check if source is None, default to vectorstore
    if source is None:
        msg = "---WARNING: ROUTER RETURNED NONE, DEFAULTING TO RAG---"
        print(msg)
        logger.warning(msg)
        return RETRIEVE

    if source.datasource == WEBSEARCH:
        # Additional validation before allowing web search
        # This provides a second layer of protection
        msg = "---DECISION: ROUTER SUGGESTED WEB SEARCH, WILL VALIDATE IN WEB_SEARCH NODE---"
        print(msg)
        logger.info(msg)
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        msg = "---DECISION: ROUTE QUESTION TO RAG---"
        print(msg)
        logger.info(msg)
        return RETRIEVE
    
    # Fallback: if datasource is not web_search or vectorstore
    msg = "---WARNING: UNKNOWN DATASOURCE, DEFAULTING TO RAG---"
    print(msg)
    logger.warning(msg)
    return RETRIEVE


flow = StateGraph(state_schema=GraphState)

flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(GENERATE, generate)
flow.add_node(WEBSEARCH, web_search)
flow.add_node(GREETING, greeting)
flow.add_node(REJECT, reject_unrelated_question)

flow.set_conditional_entry_point(
    route_question, 
    path_map={
        RETRIEVE: RETRIEVE, 
        WEBSEARCH: WEBSEARCH, 
        GREETING: GREETING,
        REJECT: REJECT
    }
)

flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

flow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
flow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={"useful": END, "not_useful": WEBSEARCH, "not_supported": GENERATE},
)

flow.add_edge(WEBSEARCH, GENERATE)
# Note: GENERATE already has conditional edge above, don't add fixed edge
flow.add_edge(GREETING, END)  # Greeting node ends immediately, no need to continue
flow.add_edge(REJECT, END)  # Reject node ends immediately

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
