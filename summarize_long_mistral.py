from typing import List
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from config import config
from load_and_chunk import ProcessingPipeline
from utilities.custom_logger import CustomLogger

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Optional, List, Mapping, Any
import re

logger = CustomLogger()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 2  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm_local = LlamaCpp(
    model_path= config.MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=8192,
    max_tokens=8192,
    temperature=0.0,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    stop=["###","\n\n", "\n    |"],
    last_n_tokens_size=256,
    n_threads=7,
    n_parts=1,
    repeat_penalty=1.21,
    # rope_freq_base=10000.0,
    # rope_freq_scale=1.0,
    top_k=40,
    top_p=0.95,
    use_mlock=True,
    use_mmap=True,
    #stream=True,
)

def summarize_long_text_by_custom(_docs: List[Document]) -> str:
    def get_short_sum_chain(template: str) -> StuffDocumentsChain:
        """ prepare a summarization chain for single text
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])

        # Define LLM chain
        llm = llm_local #ChatVertexAI()
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        return StuffDocumentsChain(llm_chain=llm_chain)

    # map summarization for each chunk
    map_template = """### Instruction: 다음 문서의 간결한 요약을 작성해 주세요.
    ### Input: {text}
    ### Response: """
    # map_template = """### Instruction: Summarize the following text. Itemize key ideas of the summary in Korean. Respond in Korean only.
    # ### Input: {text}
    # ### Response: """
    #map_template = "Write 1 sentence concise summary for the following text: {text}. CONCISE SUMMARY:"
    map_chain = get_short_sum_chain(map_template)
    
    summaries = {}
    for idx, doc in enumerate(_docs):
        summ = map_chain({"input_documents": [doc]})
        summaries[idx] = summ["output_text"]
        logger.info(str(idx) + "-->" + summ["output_text"] + "\n")

    # reduce all summarizations into one single summary
    reduce_template = """### Instruction: 다음 문서의 요점을 파악하여 맥락을 알려주세요.
    ### Input: {text}
    ### Response: """
    #reduce_template = "Write a 2 sentence summary for the following text: {text}. CONCISE SUMMARY:"
    reduce_chain = get_short_sum_chain(reduce_template)
    combined_text = "\n\n".join([s.strip() for s in summaries.values()])
    context = Document(page_content="", metadata={"source": "local"})
    combined_text = re.sub(r"\s?\d\.", "-", combined_text)
    combined_text = re.sub(r"\s?\d\)", "-", combined_text)
    combined_text = re.sub(r"[①-③]", "-", combined_text)
    # combined_text = re.sub(r"[a-zA-Z]+[(\s?)(\.)(\?)(\!)]", "", combined_text)
    combined_text = re.sub(r"^\s+", "", combined_text)
    if len(combined_text.split(" ")) < config.CHUNK_SIZE :
        combined = Document(page_content=combined_text, metadata={"source": "local"})
    else:
        combined = Document(page_content=" ", metadata={"source": "local"})
    
    context = reduce_chain({"input_documents": [combined]})

    logger.info("\n\ninput text:\n")
    logger.info(f"\n{combined_text}\n")
    logger.info(f"\ncontext: \n{context['output_text'].strip()}\n")

    return combined_text, context['output_text'].strip()
