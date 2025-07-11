from datetime import datetime
import os
from mcp import StdioServerParameters, stdio_client
from strands import Agent, tool
from strands.tools.mcp import MCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import  load_dotenv
load_dotenv()

# GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
#
# llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         # other params...
#     )

aws_docs_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"]
        )
    )
)

AWS_DOC_INSIGHTS = """
You are a AWS service specialist with expertise in:
- Reading AWS documents
- Use the aws documentation tools to provide accurate information
"""

@tool
def aws_doc_search_specialist(query: str) -> str:
    """
    Search for AWS documentation for AWS service information and costs.
    This tool agent specializes in AWS documentation search.
    """
    with aws_docs_client:
        all_tools = (
            aws_docs_client.list_tools_sync()
        )
        doc_agent = Agent(
            system_prompt=AWS_DOC_INSIGHTS,
            tools=all_tools,
            model="gemini-2.0-flash"
        )
        return str(doc_agent(query))

AWS_DOC_INSIGHTS_PLANNER = """
You are a AWS service specialist planner agent for
for getting response from available tool agent
"""

def doc_orchestrator():
    orchestrator = Agent(
        system_prompt=AWS_DOC_INSIGHTS_PLANNER,
        tools=[
            aws_doc_search_specialist
        ],
        model="gemini-2.0-flash"
    )

    return orchestrator


def doc_search():
    orchestrator = doc_orchestrator()
    user_prompt="Get different types of EC2 instance type available in AWS"
    response=orchestrator(user_prompt)
    print(response)

doc_search()


