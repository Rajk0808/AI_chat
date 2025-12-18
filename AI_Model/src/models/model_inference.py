from openai import OpenAI
import time
import logging
from typing import Dict, Optional
from datetime import datetime
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from src.models.model_factory import ModelFactory
from src.utils.exceptions import CustomException
from src.utils.metrics import MetricsTracker
from datetime import datetime
from datetime import datetime
from src.utils.exceptions import CustomException
logger = logging.getLogger(__name__)

class Node5ModelInference:
    """
    NODE 5: Model Inference for PawPilot AI
    
    Responsibilities:
    - Select appropriate model (base or fine-tuned)
    - Call LLM with engineered prompt
    - Handle streaming response
    - Track tokens, cost, latency
    - Handle errors and retries
    - Monitor rate limits
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize model inference engine
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_factory = ModelFactory()
        self.metrics = MetricsTracker()
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def run_inference(self, state: Dict) -> Dict:
        """
        NODE 5: Execute model inference
        
        Args:
            state: WorkflowState with final_prompt and model selection
        
        Updates state with:
            - raw_response: Complete model output
            - response_tokens: Tokens generated
            - inference_time: How long inference took
            - cost: API cost for this request
            - model_used: Which model was used
        """
        
        logger.info("=" * 70)
        logger.info("NODE 5: MODEL INFERENCE")
        logger.info("=" * 70)
        
        try:
            # ============================================================
            # STEP 1: SELECT MODEL (Base vs Fine-tuned)
            # ============================================================
            logger.info("STEP 1: Selecting model...")
            
            model_name = self._select_model(state)
            state["model_used"] = model_name
            logger.info(f"✓ Model selected: {model_name}")
            
            
            # ============================================================
            # STEP 2: PREPARE INFERENCE PARAMETERS
            # ============================================================
            logger.info("STEP 2: Preparing inference parameters...")
            
            inference_params = self._prepare_inference_params(state, model_name)
            logger.info(f"✓ Parameters prepared: temp={inference_params['temperature']}, max_tokens={inference_params['max_tokens']}")
            
            
            # ============================================================
            # STEP 3: TRACK START TIME
            # ============================================================
            inference_start = time.time()
            logger.info("STEP 3: Starting inference timer...")
            
            
            # ============================================================
            # STEP 4: CALL MODEL WITH RETRY LOGIC
            # ============================================================
            logger.info("STEP 4: Calling LLM...")
            
            response = self._call_model_with_retry(
                prompt=state["final_prompt"],
                model=model_name,
                **inference_params
            )
            
            
            # ============================================================
            # STEP 5: EXTRACT RESPONSE
            # ============================================================
            logger.info("STEP 5: Extracting response...")
            
            raw_response = response.choices[0].message.content #type: ignore
            response_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else len(raw_response.split()) #type: ignore
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0 #type: ignore
            
            state["raw_response"] = raw_response
            state["response_tokens"] = response_tokens
            state["prompt_tokens"] = prompt_tokens
            
            logger.info(f"✓ Response extracted ({response_tokens} tokens)")
            
            
            # ============================================================
            # STEP 6: CALCULATE METRICS
            # ============================================================
            logger.info("STEP 6: Calculating metrics...")
            
            inference_time = time.time() - inference_start
            cost = self._calculate_cost(model_name, prompt_tokens, response_tokens)
            
            state["inference_time"] = inference_time
            state["cost"] = cost
            state["model_latency_ms"] = inference_time * 1000
            
            logger.info(f"✓ Inference time: {inference_time:.2f}s")
            logger.info(f"✓ Cost: ${cost:.4f}")
            
            
            # ============================================================
            # STEP 7: TRACK METRICS FOR MONITORING
            # ============================================================
            logger.info("STEP 7: Tracking metrics...")
            
            self.metrics.record_inference(
                model=model_name,
                tokens=response_tokens,
                latency=inference_time,
                cost=cost,
                module=state.get("prompt_module", "unknown")
            )
            
            logger.info("✓ Metrics recorded")
            
            
            # ============================================================
            # STEP 8: ADD INFERENCE METADATA
            # ============================================================
            logger.info("STEP 8: Adding inference metadata...")
            
            state["inference_metadata"] = {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
                "inference_time_seconds": inference_time,
                "latency_ms": inference_time * 1000,
                "cost_usd": cost,
                "timestamp": datetime.now().isoformat(),
                "temperature": inference_params["temperature"],
                "max_tokens": inference_params["max_tokens"]
            }
            
            logger.info("✓ Metadata added")
            
            
            # ============================================================
            # STEP 9: CHECK RESPONSE QUALITY
            # ============================================================
            logger.info("STEP 9: Checking response quality...")
            
            quality_check = self._check_response_quality(
                response=str(raw_response),
                module_type=state.get("prompt_module", "general_qa")
            )
            
            state["response_quality"] = quality_check
            logger.info(f"✓ Quality check: {quality_check['status']}")
            
            if quality_check["status"] == "WARNING":
                logger.warning(f"Quality warning: {quality_check['message']}")
            
            
            logger.info("=" * 70)
            logger.info("NODE 5 COMPLETE - Ready for validation")
            logger.info("=" * 70)
            
            return state
        
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            state.get("errors", []).append(f"Rate limit error - retrying later")
            raise CustomException("Rate limit exceeded - please retry in a few minutes", str(e))
        
        except APIConnectionError as e:
            logger.error(f"API connection error: {str(e)}")
            state.get("errors", []).append(f"Connection error: {str(e)}")
            raise CustomException("Failed to connect to API", str(e))
        
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            state.get("errors", []).append(f"API error: {str(e)}")
            raise CustomException(f"API error: {str(e)}", str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error in Node 5: {str(e)}", exc_info=True)
            state.get("errors", []).append(f"Inference error: {str(e)}")
            raise CustomException(f"Node 5 Model Inference failed: {str(e)}", str(e))
    
    
    # ====================================================================
    # HELPER METHODS
    # ====================================================================
    
    def _select_model(self, state: Dict) -> str:
        """
        Select which model to use: fine-tuned or base
        
        Priority:
        1. If fine-tuned model exists and performing well → use it
        2. Otherwise → use base model
        """
        
        try:
            # Check if model was already selected in Node 2
            if "model_to_use" in state:
                return state["model_to_use"]
            
            # Check if we have an active fine-tuned model
            fine_tuned_model = self.model_factory.get_active_fine_tuned_model()
            
            if fine_tuned_model and fine_tuned_model.get("performance_score", 0) >= 0.85:
                logger.info(f"Using fine-tuned model: {fine_tuned_model['id']}")
                return fine_tuned_model["id"]
            
            # Fall back to base model
            base_model = self.model_factory.get_base_model()
            logger.info(f"Using base model: {base_model}")
            return base_model
        
        except Exception as e:
            logger.warning(f"Error selecting model, using default: {str(e)}")
            return "gpt-4-turbo"
    
    
    def _prepare_inference_params(self, state: Dict, model: str) -> Dict:
        """
        Prepare parameters for model inference
        
        Adjusts temperature, max_tokens, etc based on module type
        """
        
        module_type = state.get("prompt_module", "general_qa")
        
        # Module-specific parameters
        param_config = {
            "emergency": {
                "temperature": 0.3,  # Lower for consistency in emergencies
                "max_tokens": 300,   # Keep emergency responses concise
                "top_p": 0.9
            },
            "skin_diagnosis": {
                "temperature": 0.5,  # Balanced for medical accuracy
                "max_tokens": 500,
                "top_p": 0.95
            },
            "emotion_detection": {
                "temperature": 0.6,  # Slightly higher for nuance
                "max_tokens": 400,
                "top_p": 0.95
            },
            "product_safety": {
                "temperature": 0.4,  # Lower for accuracy
                "max_tokens": 450,
                "top_p": 0.9
            },
            "behavior": {
                "temperature": 0.7,  # Higher for creativity in training plans
                "max_tokens": 600,
                "top_p": 0.95
            }
        }
        
        # Get config for this module, or use defaults
        config = param_config.get(module_type, {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.95
        })
        
        return {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "top_p": config["top_p"],
            "stream": False  # For simplicity (can be enabled for streaming)
        }
    
    
    def _call_model_with_retry(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool = False
    ):
        """
        Call OpenAI model with retry logic for resilience
        
        Retries on rate limits and connection errors
        """
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"API call attempt {attempt + 1}/{self.max_retries}...")
                base_model = self.model_factory.get_base_model()
                response = base_model.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream
                )
                
                logger.info(f"✓ API call successful on attempt {attempt + 1}")
                return response
            
            except RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
            
            except APIConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
    
    
    def _calculate_cost(self, model: str, prompt_tokens: int, response_tokens: int) -> float:
        """
        Calculate API cost for this inference
        
        Pricing as of 2024 (update as needed)
        """
        
        pricing = {
            "gpt-4-turbo": {
                "input": 0.01 / 1000,      # $0.01 per 1K tokens
                "output": 0.03 / 1000      # $0.03 per 1K tokens
            },
            "gpt-4": {
                "input": 0.03 / 1000,
                "output": 0.06 / 1000
            },
            "gpt-3.5-turbo": {
                "input": 0.0005 / 1000,
                "output": 0.0015 / 1000
            },
            "ft-gpt-3.5-turbo": {
                "input": 0.003 / 1000,     # Fine-tuned pricing
                "output": 0.004 / 1000
            }
        }
        
        # Get pricing for model, default to base pricing
        model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
        
        cost = (prompt_tokens * model_pricing["input"]) + (response_tokens * model_pricing["output"])
        
        logger.info(f"Cost calculation: {prompt_tokens} input + {response_tokens} output = ${cost:.6f}")
        
        return cost
    
    
    def _check_response_quality(self, response: str, module_type: str) -> Dict:
        """
        Check if response meets quality standards for its module
        
        Returns: {status: "OK" | "WARNING" | "FAILED", message: str}
        """
        
        checks = {
            "status": "OK",
            "message": "Response quality is acceptable",
            "warnings": []
        }
        
        # Check 1: Response not empty
        if not response or len(response.strip()) < 10:
            checks["status"] = "FAILED"
            checks["message"] = "Response too short or empty"
            return checks
        
        # Module-specific checks
        if module_type == "emergency":
            # Emergency responses MUST have numbered steps
            if not any(f"{i}." in response for i in range(1, 10)):
                checks["status"] = "WARNING"
                checks["warnings"].append("Emergency response missing numbered steps")
            
            # Must mention urgency
            if not any(word in response.lower() for word in ["urgent", "immediately", "critical", "call vet"]):
                checks["status"] = "WARNING"
                checks["warnings"].append("Emergency response lacks urgency language")
        
        elif module_type == "skin_diagnosis":
            # Diagnosis should mention severity
            if not any(word in response.lower() for word in ["severity", "urgent", "high", "medium", "low"]):
                checks["warnings"].append("Missing severity assessment")
            
            # Should mention vet
            if "vet" not in response.lower():
                checks["status"] = "WARNING"
                checks["warnings"].append("Should mention veterinarian")
        
        elif module_type == "product_safety":
            # Product safety should have clear recommendation
            if not any(word in response.lower() for word in ["safe", "toxic", "recommend", "avoid", "unsafe"]):
                checks["status"] = "WARNING"
                checks["warnings"].append("Missing safety recommendation")
        
        elif module_type == "emotion_detection":
            # Should identify specific emotion
            emotions = ["fear", "anxiety", "happiness", "aggression", "curiosity", "trust", "stress", "submission", "playfulness"]
            if not any(emotion in response.lower() for emotion in emotions):
                checks["warnings"].append("May not have identified specific emotion")
        
        # Check for toxic patterns
        if self._contains_harmful_content(response):
            checks["status"] = "FAILED"
            checks["message"] = "Response contains harmful content"
            return checks
        
        # Check for hallucinations (very long lists, repetition)
        if response.count("\n") > 50:
            checks["warnings"].append("Response very long - possible hallucination")
        
        if checks["warnings"]:
            checks["status"] = "WARNING"
            checks["message"] = "; ".join(checks["warnings"])
        
        return checks
    
    
    def _contains_harmful_content(self, response: str) -> bool:
        """Check if response contains harmful medical advice or dangerous information"""
        
        harmful_phrases = [
            "always do this",
            "guaranteed cure",
            "don't see a vet",
            "ignore the doctor"
        ]
        
        return any(phrase.lower() in response.lower() for phrase in harmful_phrases)
