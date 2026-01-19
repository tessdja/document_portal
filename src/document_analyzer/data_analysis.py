import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from prompt.prompt_library import *

# helper function 
# trim what to send to metadata, reduces 429s dramatically
def trim_text_for_metadata(text: str, head_chars: int = 15000, tail_chars: int = 3000) -> str:
    """
    Trim document text for metadata extraction:
    - First N characters (title, abstract, intro)
    - Last M characters (references, footer info)
    """
    if len(text) <= head_chars + tail_chars:
        return text

    head = text[:head_chars]
    tail = text[-tail_chars:]

    return (
        head
        + "\n\n--- [TRUNCATED MIDDLE CONTENT] ---\n\n"
        + tail
    )

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = prompt
            
            self.log.info("DocumentAnalyzer initialized successfully")
            
            
        except Exception as e:
            # self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            # raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)
            self.log.exception("Error initializing DocumentAnalyzer")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization") from e


    def analyze_document(self, document_text: str) -> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            # 1) Trim large docs for metadata extraction
            trimmed_text = trim_text_for_metadata(document_text)

            # 2) Prompt-size logging (before calling the LLM)
            format_instructions = self.parser.get_format_instructions()

            # Rough token estimate: ~4 chars per token is a common rule of thumb for English.
            # (Not perfect, but great for spotting "way too big" prompts.)
            approx_tokens_doc = max(1, len(trimmed_text) // 4)
            approx_tokens_fmt = max(1, len(format_instructions) // 4)
            approx_tokens_total = approx_tokens_doc + approx_tokens_fmt

            self.log.info(
                "Prepared metadata prompt payload",
                original_length_chars=len(document_text),
                trimmed_length_chars=len(trimmed_text),
                format_instructions_chars=len(format_instructions),
                approx_tokens_document=approx_tokens_doc,
                approx_tokens_format_instructions=approx_tokens_fmt,
                approx_tokens_total=approx_tokens_total,
            )

            # Guardrail: ensure we are not accidentally sending the full document
            if len(trimmed_text) > 50000:
                self.log.warning(
                    "Trimmed text still large for metadata extraction",
                    trimmed_length_chars=len(trimmed_text),
                )

            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Meta-data analysis chain initialized")

            # 3) LLM call (this triggers generateContent under the hood)
            response = chain.invoke({
                "format_instructions": format_instructions,
                "document_text": trimmed_text,
            })

            self.log.info("Metadata extraction successful", keys=list(response.keys()))
            return response

        except Exception as e:
            self.log.exception("Metadata analysis failed")
            raise DocumentPortalException("Metadata extraction failed") from e
