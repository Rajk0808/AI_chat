from ast import Dict
from typing import List


class WorkFlowState:
    """
    This is a State object that flows through the graph nodes.
    Each Nodes recieve this State object as input, modifies it and passes it to the next node.

    Atributes:
        data (dict): A dictionary to hold arbitrary data throughout the workflow.   

    """

    #Input 
    query : str                            # User's original question
    user_id : str                          # Track who asked
    session_id : str                       # Track conversation session

    #decision Making
    use_rag = bool                         # Should we retrieve documents?
    model_to_use = str                     # "gpt-4-turbo" or "ft-xyz123"
    strategy = str                         # "rag", "prompt_only", "hybrid"

    #RAG Pipeline outputs
    retrieved_docs : List[str]             # Documents from vector DB
    rag_time = float                       # Time Taken by RAG
    context : str                          # Formatted context string

    #Prompt Engineeing Pipeline 
    prompt_template : str                  # Which template to use
    few_shots_examples : List[Dict]        # Example Q&A pairs
    final_prompt : str                     # Complete formatted prompt

    #Model Inference Outputs
    raw_response : str                     # Raw response from LLM
    response_tokens : int                  # Number of tokens in response
    cost : float                           # Cost incurred for this inference

    #Validation and Formatting
    validated_response : str               # Validated and formatted response
    citations : List[str]                  # Source citations
    confidence_score : float               # How confident are we? 0-1

    
    #Loggings and Metrics
    start_time : float                     # When did the request start
    end_time : float                       # When did the request end
    inference_time : float                 # Time taken for model inference
    total_time : float                     # Total time for the workflow


    #Error Handling
    error : List[str]                      # Any error encountered
    fallback_used : bool                   # Did we fall back to base model?


    #Final Output 
    final_output : str                     # The Final Output 