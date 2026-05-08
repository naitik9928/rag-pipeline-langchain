from langchain_core.documents import Document

text=Document(page_content="""In modern automotive component manufacturing, assembly lines operate around the clock, with each station performing specialized tasks such as robotic welding, computer numerical control machining, or optical inspection. These stations generate massive streams of time-series data from sensors embedded in motors, conveyors, and hydraulic presses, often recording variables like vibration frequency, motor current draw, temperature gradients, and acoustic emissions during each cycle. When a critical asset like a high-speed spindle begins to degrade, subtle changes appear first in the high-frequency vibration spectrum, typically between 5 kHz and 10 kHz, which standard supervisory control and data acquisition systems may filter out as noise. Therefore, advanced predictive maintenance pipelines apply Fast Fourier Transform to raw accelerometer data, extracting features such as kurtosis and crest factor that signal early bearing wear or imbalanced rotor conditions. If these features cross a dynamic threshold calibrated on historical failure modes, the system generates a maintenance work order automatically, flagging the specific component and suggesting a replacement window before catastrophic failure occurs.

However, real-world implementations face several persistent challenges. False positives from electrical interference or tool changes can trigger unnecessary downtime, while false negatives risk unplanned stoppages that cost an average of twenty-two thousand dollars per minute in a high-volume press line. To balance these risks, many plants deploy ensemble models combining isolation forests for anomaly detection with long short-term memory networks that learn normal machine behavior across seasonal demand shifts. Additionally, quality control in the same environment depends on computer vision systems mounted directly above the conveyor belt, capturing images at one hundred twenty frames per second. These systems detect surface defects like scratches, dents, or misaligned fastener holes using convolutional neural networks pretrained on synthetic defect libraries. One major difficulty is that lighting conditions change throughout a shift as overhead fixtures warm up or as ambient sunlight enters through bay doors, causing the same defect to appear differently in morning versus afternoon production runs. To solve this, engineers apply domain randomization during training, forcing the model to learn invariant features regardless of illumination angle or intensity. Furthermore, manufacturers must address data drift over time: when a supplier changes raw material batch chemistry or a worn cutter gradually alters part geometry, the defect distribution shifts, requiring periodic retraining or online adaptation via techniques like test-time augmentation. Recursive text splitting strategies handle this content well because natural punctuation creates variable-length segments: short declarative sentences like "These stations generate massive streams of time-series data" break differently than longer, clause-heavy sentences describing Fourier transforms or ensemble models. Paragraph boundaries at challenge statements and solution descriptions also serve as clean split points, ensuring that semantic units like "false positive mitigation" or "lighting invariance" remain intact within individual chunks for vector database retrieval. Without such careful structuring, a naive splitter might cut mid-sentence inside a technical explanation, separating a method like "domain randomization" from its purpose of handling light variation, which would confuse downstream retrieval in a RAG pipeline. Therefore, this dataset deliberately alternates between operational context, technical depth, practical obstacles, and solution strategies, giving your recursive splitter clear signals to preserve meaning across chunk boundaries while still producing uniform token lengths for embedding.""")
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
load_dotenv()
embed_model=JinaEmbeddings()
split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunk=split.split_documents([text])
vector_store=Chroma(
    embedding_function=embed_model,
    collection_name="rag",
    persist_directory="vector_store"
)
vector_store.add_documents(chunk)
retriever=vector_store.as_retriever(search_kwargs={"k":4})
prompt=PromptTemplate.from_template(template="""You are a helpful AI assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:""")
model=ChatGroq(model="llama-3.1-8b-instant")
question = input("Ask a question: ")
parser=StrOutputParser()
def joining(data):
    context=" ".join(doc.page_content for doc in data)
    return context
chain_part=RunnableParallel({
    "context":retriever|RunnableLambda(joining),
    "question":RunnablePassthrough()
})
chain=chain_part|prompt|model|parser
result=chain.invoke(question)
print(result)