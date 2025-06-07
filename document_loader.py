from longchain.document_loader import PyPDFLoader
from longchain.document_loader import HuggingFaceEnbeddings
from longchain.vectorstores import FAISS


def load_docs(folder_path):
	# Load PDFs and convert to documents
	loader = PyPDFLoader(folder_path)
	return loader.load()


def build_vector_db(docs):
	embeddings = HuggingFaceEnbeddings(moden_name="all-MiniLM-L6-v2")
	return FAISS.from_documents(docs, embeddings)

