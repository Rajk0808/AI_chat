from workflow_pipeline import ChatbotPipeline
from AI_Model.src.workflow.graph_builder import build_complete_workflow

workflow = build_complete_workflow()

bot= ChatbotPipeline(workflow)
message = 'Do you know what is a dog ?'
response = bot.process_message(message.strip())

print(response)