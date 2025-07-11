from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock
from dotenv import  load_dotenv
from utils.rag_utils import create_kb
load_dotenv()

llm = ChatBedrock(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1"
)
def main():

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=create_kb(),
        chain_type="stuff"
    )

    query = "Summarize the pepsico net revenue between 2023 and 2024"
    print('testing RAG with given prompt:' , query )
    response = rag_chain.invoke(query)
    #print(response)
    print(response['result'])

if __name__ == "__main__":
    main()
