from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    # path = 'rag\\pdfs',
    # glob = '*.pdf',
)

docs = loader.load()