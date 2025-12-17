from langgraph.graph import START, END, StateGraph
import nodes
from state_definition import WorkFlowState
def build_complete_workflow():
    """Build complete LangGraph with all 8 nodes"""
    state = WorkFlowState()
    should_use_rag = state.use_rag
    workflow = StateGraph(WorkFlowState)
    
    # Add all nodes
    workflow.add_node("input_processing", nodes.input_processing_node(state))
    workflow.add_node("decision_router", nodes.decision_router_node)
    workflow.add_node("rag_retrieval", nodes.rag_retrieval_node)
    workflow.add_node("prompt_engineering", nodes.engineer_prompt_node)
    workflow.add_node("model_inference", nodes.run_model_inference_node)
    workflow.add_node("response_validation", nodes.validate_response_node) 
    workflow.add_node("logging", nodes.log_interaction_node)               
    workflow.add_node("fine_tuning_check", nodes.check_fine_tuning_trigger_node)  
    
    # Define edges
    workflow.add_edge(START, "input_processing")
    workflow.add_edge("input_processing", "decision_router")
    workflow.add_conditional_edges(
        "decision_router",
        should_use_rag,
        {True: "rag_retrieval", False: "prompt_engineering"}
    )
    workflow.add_edge("rag_retrieval", "prompt_engineering")
    workflow.add_edge("prompt_engineering", "model_inference")
    workflow.add_edge("model_inference", "response_validation")  
    workflow.add_edge("response_validation", "logging")           
    workflow.add_edge("logging", "fine_tuning_check")            
    workflow.add_edge("fine_tuning_check", END)
    
    return workflow.compile()