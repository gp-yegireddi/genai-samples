from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def create_kb():

    loader = PyPDFLoader("./reports/2024-pepsico-annual-report.pdf")
    docs = loader.load()
    print('document loaded')
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local("faiss_index")
        print('faiss_index created')
        retriever = FAISS.load_local("faiss_index", embeddings,  allow_dangerous_deserialization=True).as_retriever()
        return retriever
    except Exception as e:
        print('Exception while creating vector store:',e)