from AI_Model.src.utils.exceptions import CustomException
from workflow_pipeline import ChatbotPipeline
from AI_Model.src.workflow.graph_builder import build_complete_workflow
import sys
workflow = build_complete_workflow()
try:
    bot= ChatbotPipeline(workflow)
    message = 'Do you know what is a dog ?'
    response = bot.process_message(message.strip())
 
    print(response)
except Exception as e:
    raise CustomException(e,sys)