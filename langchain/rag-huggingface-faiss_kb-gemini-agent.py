from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import  load_dotenv
from utils.rag_utils import create_kb
from langchain.agents import Tool, initialize_agent, AgentType

load_dotenv()

llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=create_kb(),
        chain_type="stuff"
    )
qa_tool = Tool(
    name="PEP_AnnualReport_QA",
    func=rag_chain.run,
    description="Knowledgebase for Pepsico Annual Report PDF"
)

def main():
    agent = initialize_agent(
        tools=[qa_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    query = "Summarize the pepsico operating profit between 2023 and 2024"
    response=agent.invoke(query)
    print('AI response::', response)

if __name__ == "__main__":
    main()
