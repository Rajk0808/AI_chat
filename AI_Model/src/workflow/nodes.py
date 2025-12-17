import sys
from state_definition import WorkFlowState
import time
import logging
from rag.rag_pipline import RAGPipeline
from typing import Dict
from src.utils.exceptions import CustomException
logger = logging.getLogger(__name__)

# NODE 1: Input Processing Node

def input_processing_node(state : WorkFlowState) -> WorkFlowState:
    """
    Node 1: Validat and Processing user Inputs 
    
    Responsiblities:
    - Validate the user query and IDs
    - Extract MetaData
    - Intialize TimeStamp
    - set default values

    """


    # If not Valdate state
    if not state.query:
        state.error.append('Invalid Query')
        state.strategy = "error" #type: ignore
        return state
    if len(state.query) > 2000:
        state.error.append('Query too long')
        state.query = state.query[:2000] #Truncate to 2000 chars
    
    state.start_time = time.time()
    state.error = []
    state.fallback_used = False

    logger.info(f'Proccesing input query from {state.user_id} in session {state.session_id}')

    return state

# NODE 2: Decision Router Node

def decision_router_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 2 : Decides which technique to be used based on the state inputs

    Responsiblities:

    - Analyze the Query types
    - Decides on RAG usage
    - Select Appropriate Model 
    - choose Execution Strategy
    """

    #Analyze Query keywords to decide RAG usage
    rag_keywords = ['what', 'how', 'explain', 'describe', 'guide', 'help', 'list']
    state.use_rag = any(  #type: ignore
        keyword.lower() in state.query.lower()
        for keyword in rag_keywords
    )

    #Check if fine-tuned model is available for user and if perfroming well
    ft_model = False
    if ft_model and ft_model.performance_score >= 0.9:
        state.model_to_use = str(ft_model.id) #type: ignore
        logger.info(f'Using fine-tuned model {ft_model.id} for user {state.user_id}')
    else:
        state.model_to_use = 'gpt-4-turbo' #type: ignore
        logging.info('Using base model gpt-4-turbo')

    # Determine Execution Strategy
    if state.use_rag:
        state.strategy = "rag" #type: ignore
    else:
        state.strategy = "prompt_only" #type: ignore

    return state

# NODE 3: RAG Retrieval 

def rag_retrieval_node(state: WorkFlowState) -> WorkFlowState:
    """
    Node 3: Retrieve relevant documents from vector store based on user query

    Responsiblities:
    - Connect to vector store
    - Perform similarity search
    - Update state with retrieved documents
    """
   
    if not state.use_rag:
        logger.info('RAG not required, skipping document retrieval step')
        state.context = ""
        state.retrieved_docs = []
        return state
    logger.info("Executing RAG pipeline...")
    rag_start = time.time()

    try:
        # Step 1: Call RAG system
        rag_system = RAGPipeline() #---------------
        
        # Step 2: Retrieve documents
        docs = rag_system.retriever(
            query = state.query,
            top_k = 4
        )

        state.retrieved_docs = docs
        state.context = "\n---\n".join(docs)

        # Step 3: Time tracking
        state.rag_time = float(time.time() - rag_start) #type: ignore 
        logger.info(f'RAG retrived {len(docs)} documents in {state.rag_time:.2f}s')
        return state

    except Exception as e:
        state.error.append(f"RAG error: {str(e)}")
        state.context = ""
        state.use_rag = False #type: ignore
        raise CustomException(e, sys)
    
# NODE 4: PROMPT ENGINEERING

def engineer_prompt_node(state: Dict) -> Dict:
    """
    LangGraph Node 4: Prompt Engineering
    
    Builds optimized prompts for PawPilot AI
    """
    from src.prompt_engineering._init_ import Node4PromptEngineering
    node = Node4PromptEngineering()
    return node.engineer_prompt(state)


# NODE 5: MODEL INFERENCE

def run_model_inference_node(state: Dict) -> Dict:
    """
    LangGraph Node 5: Model Inference
    
    Calls LLM with optimized prompt and tracks metrics
    """
    from src.models.model_inference import Node5ModelInference
    node = Node5ModelInference()
    return node.run_inference(state)


# NODE 6: RESPONSE VALIDATION & FORMATTING

def validate_response_node(state: Dict) -> Dict:
    """LangGraph Node 6"""
    from src.utils.reponse_validator import Node6ResponseValidator
    validator = Node6ResponseValidator()
    return validator.validate_response(state)

# NODE 7: LOGGING & FEEDBACK COLLECTION

def log_interaction_node(state: Dict) -> Dict:
    """LangGraph Node 7"""
    from src.logging.interaction_logger import Node7InteractionLogger
    logger_node = Node7InteractionLogger()
    return logger_node.log_interaction(state)

# NODE 8: FINE-TUNING TRIGGER

def check_fine_tuning_trigger_node(state: Dict) -> Dict:
    """LangGraph Node 8"""
    from src.fine_tuning.fine_tuner import check_fine_tuning_trigger
    return check_fine_tuning_trigger(state)



# ============================================================================
# SUPPORTING CLASSES
# ============================================================================

#class StreamingCallback:
#    """Handle streaming responses from model"""
#    
#    def __init__(self):
#        self.full_response = ""
#    
#    def on_token(self, token: str):
#        """Called when a new token arrives"""
#        self.full_response += token
#        print(token, end="", flush=True)  # Print token in real-time
#    
#    def get_response(self) -> str:
#        """Get complete accumulated response"""
#        return self.full_response
    