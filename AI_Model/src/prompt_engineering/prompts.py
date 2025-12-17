from typing import Dict, Optional, List
import json
from pathlib import Path

class PawPilotPromptBuilder:
    """Build specialized prompts for PawPilot AI modules"""
    
    def __init__(self):
        self.templates_dir = Path("src/prompt_engineering/templates")
        self.load_all_templates()
    
    def load_all_templates(self):
        """Load all PawPilot-specific templates"""
        self.system_prompts = self._load_json("system_prompts.json")
        self.vision_prompts = self._load_json("vision_prompts.json")
        self.audio_prompts = self._load_json("audio_prompts.json")
        self.few_shot = self._load_json("few_shot_examples.json")
        self.rag_templates = self._load_json("rag_context_templates.json")
    
    def _load_json(self, filename: str) -> Dict:
        path = self.templates_dir / filename
        with open(path, 'r') as f:
            return json.load(f)
    
    # ==========================================
    # SKIN DIAGNOSIS PROMPT
    # ==========================================
    
    def build_skin_diagnosis_prompt(
        self,
        pet_profile: Dict,
        symptom_description: str,
        rag_context: str
    ) -> str:
        """
        Build prompt for analyzing pet skin conditions
        
        Args:
            pet_profile: {"name": "Max", "breed": "Golden", "age": 4, "allergies": [...]}
            symptom_description: Description of what's visible in image
            rag_context: Retrieved relevant conditions from RAG database
        """
        
        system = self.system_prompts["skin_health_diagnostic"]
        
        prompt = f"""You are a {system['role']}
        
        {system['context']}
        
        KEY PRINCIPLES:
        {chr(10).join(f"- {p}" for p in system['key_principles'])}
        
        PET INFORMATION:
        - Name: {pet_profile.get('name', 'Unknown')}
        - Breed: {pet_profile.get('breed', 'Unknown')}
        - Age: {pet_profile.get('age', 'Unknown')} years
        - Allergies: {', '.join(pet_profile.get('allergies', ['None known']))}
        - Medical History: {pet_profile.get('medical_history', 'None reported')}
        
        KNOWLEDGE BASE REFERENCE (Similar conditions from RAG):
        {rag_context}
        
        SYMPTOM ANALYSIS:
        {symptom_description}
        
        ANALYSIS REQUIRED:
        1. Examine described symptoms
        2. Cross-reference with RAG knowledge base
        3. Consider pet's health history
        4. Assess severity and urgency
        5. Provide first aid steps
        6. Recommend when to see vet
        
        OUTPUT FORMAT:
        ## Observations
        [What you see in the symptoms]
        
        ## Possible Conditions
        - [Condition 1] - Likelihood: [High/Medium/Low]
        - [Condition 2] - Likelihood: [High/Medium/Low]
        
        ## Severity Level
        [Low/Medium/High/Emergency]
        
        ## Urgency for Vet Visit
        [Within 1 week / 48-72 hours / 24 hours / IMMEDIATE]
        
        ## First Aid Steps
        1. [Action]
        2. [Action]
        3. [Action]
        
        ## Monitoring
        What to watch for and when to escalate
        
        ## Important Notes
        Always phrase as "possible conditions" not diagnosis. When uncertain, recommend vet visit."""
                
        return prompt
            
    # ==========================================
    # EMOTION DETECTION PROMPT
    # ==========================================
    
    def build_emotion_detection_prompt(
        self,
        image_features: str,
        audio_analysis: str,
        pet_profile: Dict,
        rag_emotion_data: str
    ) -> str:
        """Build prompt for EmoDetect emotion analysis"""
        
        system = self.system_prompts["voice_emotion_translator"]
        
        prompt = f"""You are an {system['role']}
        
        {system['context']}
        
        EMOTION DETECTION FRAMEWORK (From EmoDetect RAG):
        {rag_emotion_data}
        
        PET CONTEXT:
        - Name: {pet_profile.get('name')}
        - Breed: {pet_profile.get('breed')}
        - Age: {pet_profile.get('age')} years
        - Personality: {pet_profile.get('personality', 'Unknown')}
        - Recent Events: {pet_profile.get('recent_events', 'None reported')}
        
        VISUAL ANALYSIS:
        {image_features}
        
        AUDIO ANALYSIS:
        {audio_analysis}
        
        DETECTION TASK:
        1. Identify body language indicators (using EmoDetect framework)
        2. Analyze audio patterns and vocalizations
        3. Consider pet's personality and recent context
        4. Cross-reference with EmoDetect emotion taxonomy
        5. Assess confidence level of emotion detection
        6. Recommend appropriate actions
        
        OUTPUT FORMAT:
        ## Primary Emotion
        [Emotion from EmoDetect list]
        
        ## Confidence Level
        High / Medium / Low
        
        ## Key Indicators Observed
        - Body Language: [indicators]
        - Vocalizations: [audio patterns]
        - Context Clues: [situational factors]
        
        ## Root Cause Analysis
        [What triggered this emotion]
        
        ## Recommended Actions
        1. [What to do]
        2. [How to help]
        3. [When to seek help]
        
        ## Important Notes
        Be specific about which body parts indicate which emotions using EmoDetect framework."""
        
        return prompt
    
    # ==========================================
    # EMERGENCY RESPONSE PROMPT
    # ==========================================
    
    def build_emergency_prompt(
        self,
        emergency_type: str,
        symptoms: str,
        pet_profile: Dict,
        rag_emergency_protocols: str
    ) -> str:
        """Build CRITICAL CARE prompt for emergencies"""
        
        system = self.system_prompts["emergency_assistant"]
        
        prompt = f"""You are a {system['role']}

        {system['context']}
        
        KEY PRINCIPLES FOR LIFE-SAVING RESPONSES:
        {chr(10).join(f"- {p}" for p in system['key_principles'])}
        
        PET INFORMATION:
        - Age: {pet_profile.get('age')} years
        - Weight: {pet_profile.get('weight')} kg
        - Medical Conditions: {pet_profile.get('medical_conditions', 'None reported')}
        - Current Medications: {pet_profile.get('medications', 'None')}
        
        EMERGENCY TYPE: {emergency_type}
        
        SYMPTOMS REPORTED:
        {symptoms}
        
        EMERGENCY PROTOCOLS (From RAG):
        {rag_emergency_protocols}
        
        CRITICAL RESPONSE REQUIRED:
        ⚠️ PROVIDE NUMBERED STEPS ONLY - NO PARAGRAPHS
        ⚠️ START WITH SEVERITY ASSESSMENT
        ⚠️ INCLUDE WHAT NOT TO DO
        ⚠️ SPECIFY TIME WINDOWS
        
        OUTPUT FORMAT (STRICT):
        ## 🚨 SEVERITY LEVEL
        [LIFE-THREATENING / SERIOUS / URGENT]
        
        ## ⏱️ TIME CRITICAL WINDOW
        [How much time available before escalation needed]
        
        ## 🔴 IMMEDIATE ACTIONS (IN ORDER):
        1. [Action - be specific]
        2. [Action - be specific]
        3. [Action - be specific]
        4. [Action - be specific]
        
        ## ❌ WHAT NOT TO DO:
        - Do NOT [something dangerous]
        - Do NOT [something dangerous]
        - Do NOT [something dangerous]
        
        ## 📞 VET URGENCY
        [CALL IMMEDIATELY / Go to ER now / Vet within 1 hour]
        
        ## 🩺 EQUIPMENT NEEDED
        [List specific items needed]
        
        ## 👁️ WARNING SIGNS FOR ESCALATION
        [When to stop and go to vet immediately]"""
                
        return prompt
    
    # ==========================================
    # PRODUCT SAFETY PROMPT
    # ==========================================
    
    def build_product_analysis_prompt(
        self,
        product_info: Dict,
        pet_profile: Dict,
        rag_safety_database: str
    ) -> str:
        """Build prompt for product safety evaluation"""
        
        system = self.system_prompts["product_safety_evaluator"]
        
        prompt = f"""You are a {system['role']}
        
        {system['context']}
        
        SAFETY DATABASE REFERENCE:
        {rag_safety_database}
        
        PET PROFILE:
        - Species: {pet_profile.get('species')}
        - Age: {pet_profile.get('age')} years
        - Weight: {pet_profile.get('weight')} kg
        - Known Allergies: {', '.join(pet_profile.get('allergies', ['None']))}
        - Health Conditions: {pet_profile.get('health_conditions', 'None reported')}
        
        PRODUCT TO EVALUATE:
        - Name: {product_info.get('name')}
        - Type: {product_info.get('type')} (food/treat/toy/supplement)
        - Ingredients: {product_info.get('ingredients')}
        - Price: {product_info.get('price', 'Unknown')}
        
        EVALUATION STEPS:
        1. Check each ingredient against safety database
        2. Flag any toxic substances
        3. Assess portion/size appropriateness for pet
        4. Check for allergen risks specific to this pet
        5. Evaluate nutritional content
        6. Compare price vs value
        7. Suggest alternatives if needed
        
        OUTPUT FORMAT:
        ## 🏷️ Product Name & Type
        [Info]
        
        ## ✅ Safety Assessment
        Safe / Caution / Not Safe
        
        ## 🚨 Toxic Ingredients Found
        [List any toxic ingredients, or "None found"]
        
        ## ⚠️ Allergen Concerns
        [Any concerns for this specific pet]
        
        ## 📊 Safety Score
        [1-10 scale with explanation]
        
        ## 💡 Better Alternatives
        [Healthier/safer options]
        
        ## 📏 Appropriate Portion Size
        [Specific amount for this pet's weight]
        
        ## 💰 Value Assessment
        [Is it worth the price?]
        
        ## 🎯 Final Recommendation
        [Clear recommendation for this pet]"""
        
        return prompt
    
    # ==========================================
    # RAG-AWARE PROMPT BUILDER
    # ==========================================
    
    def build_rag_aware_prompt(
        self,
        module: str,
        user_query: Dict,
        pet_profile: Dict,
        rag_retrieved_data: str
    ) -> str:
        """
        Build prompt that intelligently uses RAG context
        
        The key to PawPilot: RAG data directly informs the prompt
        """
        
        if module == "skin_diagnosis":
            return self.build_skin_diagnosis_prompt(
                pet_profile, 
                user_query.get("symptom_description", ""),
                rag_retrieved_data
            )
        elif module == "emotion_detection":
            return self.build_emotion_detection_prompt(
                user_query["image_features"],
                user_query["audio_analysis"],
                pet_profile,
                rag_retrieved_data
            )
        elif module == "emergency":
            return self.build_emergency_prompt(
                user_query["emergency_type"],
                user_query["symptoms"],
                pet_profile,
                rag_retrieved_data
            )
        elif module == "product_safety":
            return self.build_product_analysis_prompt(
                user_query,
                pet_profile,
                rag_retrieved_data
            )
        else:
            raise ValueError(f"Unknown module: {module}")