"""
workflow_pipeline.py
Integrates LangGraph with Flask backend for chatbot
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from graph_builder import build_complete_workflow
from state_definition import WorkFlowState
import logging
from typing import Optional
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compile the workflow once at startup
try:
    workflow = build_complete_workflow()
    logger.info("✓ Workflow compiled successfully")
except Exception as e:
    logger.error(f"✗ Error compiling workflow: {e}")
    workflow = None

class ChatbotPipeline:
    """Manages the chatbot workflow execution"""
    
    def __init__(self, compiled_workflow):
        self.workflow = compiled_workflow
        self.conversation_history = []
    
    def process_message(self, user_input: str) -> Optional[str]:
        """
        Process user message through the LangGraph workflow
        
        Args:
            user_input: User's message
            
        Returns:
            Bot response or error message
        """
        try:
            # Create initial state
            initial_state = {
                "user_input": user_input,
                "conversation_history": self.conversation_history,
                "use_rag": True,  # Set based on your logic
                "messages": [],
                "current_step": "input_processing"
            }
            
            # Execute workflow
            logger.info(f"Processing input: {user_input[:50]}...")
            result = self.workflow.invoke(initial_state)
            
            # Extract response from result
            bot_response = self._extract_response(result)
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_input,
                "bot": bot_response
            })
            
            logger.info("✓ Message processed successfully")
            return bot_response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg
    
    def _extract_response(self, result: dict) -> str:
        """
        Extract the final response from workflow result
        
        Args:
            result: The output from workflow.invoke()
            
        Returns:
            The bot's response string
        """
        # Adjust these keys based on your actual state structure
        if isinstance(result, dict):
            # Try different possible keys where response might be stored
            possible_keys = [
                "response",
                "final_response", 
                "output",
                "bot_response",
                "messages",
                "answer"
            ]
            
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


# Initialize pipeline
pipeline = ChatbotPipeline(workflow)


# ==================== FLASK ROUTES ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "workflow_loaded": workflow is not None
    }), 200


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    Expects: {"message": "user message"}
    Returns: {"reply": "bot response", "status": "success"}
    """
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                "reply": "Please enter a message",
                "status": "error"
            }), 400
        
        if workflow is None:
            return jsonify({
                "reply": "Workflow not initialized. Check server logs.",
                "status": "error"
            }), 500
        
        # Process through pipeline
        response = pipeline.process_message(user_message)
        
        return jsonify({
            "reply": response,
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            "reply": f"Server error: {str(e)}",
            "status": "error"
        }), 500


@app.route('/reset', methods=['POST'])
def reset_chat():
    """Reset conversation history"""
    try:
        pipeline.reset_conversation()
        return jsonify({
            "message": "Conversation reset",
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Error resetting: {str(e)}",
            "status": "error"
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        return jsonify({
            "history": pipeline.conversation_history,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Error retrieving history: {str(e)}",
            "status": "error"
        }), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    try:
        return jsonify({
            "workflow_loaded": workflow is not None,
            "conversation_length": len(pipeline.conversation_history),
            "status": "running"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting AI Chatbot Server")
    logger.info("=" * 50)
    
    if workflow is None:
        logger.error("CRITICAL: Workflow failed to compile!")
        logger.error("Check your graph_builder.py and node definitions")
    else:
        logger.info("✓ Workflow ready")
        logger.info("Available endpoints:")
        logger.info("  POST   /chat      - Send message")
        logger.info("  POST   /reset     - Reset conversation")
        logger.info("  GET    /history   - Get chat history")
        logger.info("  GET    /status    - Get server status")
        logger.info("  GET    /health    - Health check")
        logger.info("=" * 50)
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Prevent double initialization
    )