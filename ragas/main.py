from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

path = "pdffiles"
loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=100, length_function=len, is_separator_regex=False)
texts = text_splitter.create_documents([pages[i].page_content for i in range(len(pages))])
print(len(texts))
print(texts[:20])
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator


generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key="")
critic_llm = ChatOpenAI(
    model="explodinggradients/Ragas-critic-llm-Qwen1.5-GPTQ",
    openai_api_key="YOA60XK036GNJ39QFIE9TETZFRK2G5VJK4S5NN3X",
    openai_api_base="https://api.runpod.ai/v2/1lh33m6cyf1cue/openai/v1",
    max_tokens=2048,
    temperature=0,
)
embeddings = OpenAIEmbeddings(openai_api_key="")

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

distributions = {simple: 0.5, multi_context: 0.25, reasoning: 0.25}

testset = generator.generate_with_langchain_docs(texts, 10, distributions)
df = testset.to_pandas()
df.to_csv("runpos.csv")
