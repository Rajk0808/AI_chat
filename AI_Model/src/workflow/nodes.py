import sys
import time
from pathlib import Path



# Fix imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.exceptions import CustomException
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Try importing RAG - if it fails, continue with mock
try:
    from src.rag.rag_pipline import RAGPipeline, should_use_rag
    RAG_AVAILABLE = True
except Exception as e:
    logger.warning(f"RAG not available: {e}")
    RAG_AVAILABLE = False



# ============================================================================
# NODE 1: Input Processing Node
# ============================================================================

def input_processing_node(state: Dict) -> Dict:
    """
    Node 1: Validate and Process user Inputs 
    
    Responsibilities:
    - Validate the user query and IDs
    - Extract MetaData
    - Initialize TimeStamp
    - Set default values
    """
    logger.info("NODE 1: Input Processing")
    
    try:
        # Access state as dictionary
        query = state.get("query", "")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        
        # Validate query
        if not query or query.strip() == "":
            logger.warning("Invalid/empty query provided")
            state["error"] = ["Invalid Query"]
            state["strategy"] = "error"
            return state
        
        # Truncate long queries
        if len(query) > 2000:
            logger.warning(f"Query too long ({len(query)} chars), truncating...")
            state["error"] = ["Query too long"]
            state["query"] = query[:2000]
        
        # Initialize metadata
        state["start_time"] = time.time()
        if "error" not in state or state["error"] is None:
            state["error"] = []
        state["fallback_used"] = False
        
        logger.info(f'Processing query from user {user_id} in session {session_id}')
        logger.info(f'Query: {state["query"][:50]}...')
        
        return state
        
    except Exception as e:
        logger.error(f"Error in input_processing_node: {e}")
        state["error"] = [f"Input processing error: {str(e)}"]
        state["strategy"] = "error"
        return state


# ============================================================================
# NODE 2: Decision Router Node
# ============================================================================

def decision_router_node(state: Dict) -> Dict:
    """
    Node 2: Decides which technique to be used based on state inputs

    Responsibilities:
    - Analyze the Query types
    - Decide on RAG usage
    - Select Appropriate Model 
    - Choose Execution Strategy
    """
    logger.info("NODE 2: Decision Router")
    
    try:
        query = state.get("query", "").lower()
        
        
        use_rag = should_use_rag(query)
        
        state["use_rag"] = use_rag
        logger.info(f"RAG decision: use_rag = {use_rag}")
        
        # Check if fine-tuned model is available (for now, using base model)
        ft_model = None  # Placeholder
        
        if ft_model and hasattr(ft_model, 'performance_score') and ft_model.performance_score >= 0.9:
            state["model_to_use"] = str(ft_model.id)
            logger.info(f'Using fine-tuned model: {ft_model.id}')
        else:
            state["model_to_use"] = 'gpt-4-turbo'
            logger.info('Using base model: gpt-4-turbo')
        
        # Determine Execution Strategy
        if use_rag:
            state["strategy"] = "rag"
        else:
            state["strategy"] = "prompt_only"
        
        logger.info(f"Strategy: {state['strategy']}")
        return state
        
    except Exception as e:
        logger.error(f"Error in decision_router_node: {e}")
        state["error"] = [f"Router error: {str(e)}"]
        state["strategy"] = "error"
        return state


# ============================================================================
# NODE 3: RAG Retrieval
# ============================================================================

def rag_retrieval_node(state: Dict) -> Dict:
    """
    Node 3: Retrieve relevant documents from vector store

    Responsibilities:
    - Connect to vector store
    - Perform similarity search
    - Update state with retrieved documents
    """
    logger.info("NODE 3: RAG Retrieval")
    
    try:
        use_rag = state.get("use_rag", False)
        
        if not use_rag:
            logger.info('RAG not required, skipping document retrieval')
            state["context"] = ""
            state["retrieved_docs"] = []
            return state
        
        if not RAG_AVAILABLE:
            logger.warning("RAG pipeline not available, skipping")
            state["context"] = ""
            state["retrieved_docs"] = []
            return state
        
        logger.info("Executing RAG pipeline...")
        rag_start = time.time()
        
        try:
            # Step 1: Call RAG system
            rag_system = RAGPipeline()
            
            # Step 2: Retrieve documents
            query = state.get("query", "")
            docs = rag_system.retriever(query=query, top_k=4)
            
            # Step 3: Update state
            state["retrieved_docs"] = docs if docs else []
            state["context"] = "\n---\n".join(doc['text'] for doc in docs) if docs else ""
            state["rag_time"] = float(time.time() - rag_start)
            print((state['retrieved_docs']))
            logger.info(f'RAG retrieved {len(docs)} documents in {state["rag_time"]:.2f}s')
            return state
            
        except Exception as e:
            logger.error(f"RAG execution error: {e}")
            state["error"].append(f"RAG error: {str(e)}")
            state["context"] = ""
            state["retrieved_docs"] = []
            state["use_rag"] = False  # Fallback
            return state
    
    except Exception as e:
        logger.error(f"Error in rag_retrieval_node: {e}")
        state["error"] = [f"RAG retrieval error: {str(e)}"]
        return state


# ============================================================================
# NODE 4: Prompt Engineering
# ============================================================================

def engineer_prompt_node(state: Dict) -> Dict:
    """
    Node 4: Prompt Engineering
    
    Builds optimized prompts
    """
    logger.info("NODE 4: Prompt Engineering")
    
    try:
        query = state.get("query", "")
        context = state.get("context", "")
        # Build the prompt
        if context:
            prompt = f"""Based on the following context, answer the user's query:

Context:
{context}

User Query: {query}

Please provide a comprehensive and accurate answer."""
        else:
            prompt = f"""Answer the following query:

User Query: {query}

Please provide a comprehensive and accurate answer."""
        
        state["final_prompt"] = prompt
        state["prompt_template"] = "default"
        logger.info(f"Prompt engineered ({len(prompt)} chars)")
        return state
        
    except Exception as e:
        logger.error(f"Error in engineer_prompt_node: {e}")
        state["error"].append(f"Prompt engineering error: {str(e)}")
        state["final_prompt"] = state.get("query", "")
        return state


# ============================================================================
# NODE 5: Model Inference
# ============================================================================

def run_model_inference_node(state: Dict) -> Dict:
    """
    Node 5: Model Inference
    
    Calls LLM with optimized prompt
    """
    logger.info("NODE 5: Model Inference")
    
    try:
        from AI_Model.src.models.model_inference import Node5ModelInference
        node = Node5ModelInference()
        result = node.run_inference(state)

        logger.info("Model inference completed")
        return result
        
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        # Fallback response
        state["raw_response"] = f"I received your query: {state.get('query', 'No query')}. However, I encountered an error processing it: {str(e)}"
        state["response_tokens"] = len(state["raw_response"].split())
        state["cost"] = 0.0
        state["error"].append(f"Model inference error: {str(e)}")
        raise CustomException(e,sys)
    finally:
        return state
 

# ============================================================================
# NODE 6: Response Validation & Formatting
# ============================================================================

def validate_response_node(state: Dict) -> Dict:
    """
    Node 6: Validate Response
    
    Validates and formats the model response
    """
    logger.info("NODE 6: Response Validation")
    
    try:
        from AI_Model.src.utils.reponse_validator import Node6ResponseValidator
        validator = Node6ResponseValidator()
        result = validator.validate_response(state)
        logger.info("Response validation completed")
        return result
        
    except Exception as e:
        logger.error(f"Response validation error: {e}")
        # Fallback: use raw response as validated response
        state["validated_response"] = state.get("raw_response", "")
        state["citations"] = []
        state["confidence_score"] = 0.5
        raise CustomException(e, sys)
    finally:
        return state


# ============================================================================
# NODE 7: Logging & Feedback Collection
# ============================================================================

def log_interaction_node(state: Dict) -> Dict:
    """
    Node 7: Log Interaction
    
    Logs the interaction for monitoring
    """
    logger.info("NODE 7: Logging & Feedback")
    
    try:
        from AI_Model.src.logging.interaction_logger import Node7InteractionLogger
        logger_node = Node7InteractionLogger()
        result = logger_node.log_interaction(state)
        logger.info("Interaction logged")
        return result
        
    except Exception as e:
        logger.error(f"Logging error: {e}")
        # Fallback: just track time
        state["end_time"] = time.time()
        state["total_time"] = state["end_time"] - state.get("start_time", state["end_time"])
        logger.info(f"Total time: {state['total_time']:.2f}s")
        return state


# ============================================================================
# NODE 8: Fine-Tuning Trigger
# ============================================================================

def check_fine_tuning_trigger_node(state: Dict) -> Dict:
    """
    Node 8: Check Fine-Tuning Trigger
    
    Determines if fine-tuning should be triggered
    """
    logger.info("NODE 8: Fine-Tuning Check")
    
    try:
        from AI_Model.src.fine_tuning.fine_tuner import check_fine_tuning_trigger
        result = check_fine_tuning_trigger(state)
        logger.info("Fine-tuning check completed")
        return result
        
    except Exception as e:
        logger.error(f"Fine-tuning check error: {e}")
        # Fallback: just return state
        state["final_output"] = state.get("validated_response", state.get("raw_response", ""))
        return state
    