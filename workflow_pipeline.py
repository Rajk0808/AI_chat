from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import traceback
import sys
from contextlib import asynccontextmanager

from AI_Model.src.utils.exceptions import CustomException
from AI_Model.src.workflow.graph_builder import build_complete_workflow

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str
    status: str = "success"

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    status: str = "success"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
workflow = None
pipeline = None

class ChatbotPipeline:
    """Manages the chatbot workflow execution"""
    
    def __init__(self, compiled_workflow):
        self.workflow = compiled_workflow
        self.conversation_history: List[dict] = []
    
    def process_message(self, user_input: str) -> Optional[str]:
        """
        Process user message through the LangGraph workflow
        
        Args:
            user_input: User's message
            
        Returns:
            Bot response or error message
        """
        try:
            initial_state = {
                "query": user_input,
                "conversation_history": self.conversation_history,
                "use_rag": True,  
                "messages": [],
                "current_step": "input_processing"
            }
            
            logger.info(f"Processing input: {user_input[:50]}...")
            result = self.workflow.invoke(initial_state)
            
            # Extract response from result
            bot_response = self._extract_response(result)
            
            # Update conversation history
            self.conversation_history.append({
                "query": user_input,
                "bot": bot_response
            })
            
            logger.info("✓ Message processed successfully")
            return bot_response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise CustomException(e, sys)
    
    def _extract_response(self, result: dict) -> str:
        """
        Extract the final response from workflow result
        
        Args:
            result: The output from workflow.invoke()
            
        Returns:
            The bot's response string
        """
        if isinstance(result, dict):
            # Try different possible keys where response might be stored
            possible_keys = ["validated_response"]
            
            for key in possible_keys:
                if key in result and result[key]:
                    value = result[key]
                    # If it's a list of messages, get the last one
                    if isinstance(value, list) and len(value) > 0:
                        last_msg = value[-1]
                        if isinstance(last_msg, dict) and "content" in last_msg:
                            return last_msg["content"]
                        return str(last_msg)
                    return str(value)
        
        # Fallback
        return "No response generated"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
        return 
    def get_history(self) -> List[dict]:
        """Get conversation history"""
        return self.conversation_history

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events
    """
    # Startup
    global workflow, pipeline
    
    logger.info("=" * 50)
    logger.info("Starting AI Chatbot Server")
    logger.info("=" * 50)
    
    try:
        workflow = build_complete_workflow()
        pipeline = ChatbotPipeline(workflow)
        logger.info("✓ Workflow compiled successfully")
    except Exception as e:
        logger.error(f"✗ Error compiling workflow: {e}")
        logger.error(traceback.format_exc())
        workflow = None
        pipeline = None
    
    if workflow is None:
        logger.error("CRITICAL: Workflow failed to compile!")
        logger.error("Check your graph_builder.py and node definitions")
    else:
        logger.info("✓ Workflow ready")
        logger.info("Available endpoints:")
        logger.info("  POST   /api/chat      - Send message")
        logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
app = FastAPI(
    title="AI Chatbot API",
    description="FastAPI backend for AI Chatbot with LangGraph workflow",
    version="1.0.0",
    lifespan=lifespan
)

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a message to the chatbot",
    responses={
        200: {"description": "Message processed successfully"},
        400: {"description": "Invalid request (empty message)"},
        500: {"description": "Server error"}
    }
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - sends user message to chatbot
    
    Args:
        request: ChatRequest containing the user message
        
    Returns:
        ChatResponse: Bot's reply and status
        
    Raises:
        HTTPException: If message is empty or workflow not initialized
    """
    try:
        user_message = request.message.strip()
        
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please enter a message"
            )
        
        if workflow is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow not initialized. Check server logs."
            )
        
        # Process through pipeline
        response = pipeline.process_message(user_message) #type: ignore
        
        return ChatResponse(reply=response  , status="success")#type: ignore
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",  # Change "main" to your file name if different
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )