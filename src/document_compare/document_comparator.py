import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
# Unable to import OutputFixingParser after langchain split/organization
# from langchain.output_parsers.fix import OutputFixingParser
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse,PromptType

import json
from pydantic import ValidationError

def _approx_tokens(text: str) -> int:
    # Rough heuristic: ~4 characters per token
    return max(1, len(text) // 4)

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        
        # This parser mainly provides format_instructions; chain.invoke() may return list[dict]
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        # self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        
        self.log.info("DocumentComparatorLLM initialized", model=self.llm)

    def _repair_to_valid_json(self, bad_output) -> str:
        """One-shot repair prompt to convert model output into valid JSON only."""
        repair_prompt = (
            "Fix the following output so it is VALID JSON ONLY.\n"
            "Do not include explanations, markdown, code fences, or extra text.\n"
            "The JSON must be a LIST of objects, each with keys:\n"
            "  Page (integer), Changes (string)\n\n"
            f"BAD_OUTPUT:\n{bad_output}"
        )
        return self.llm.invoke(repair_prompt).content
    
    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            # --------------------------------------------------
            # Step 1: Prepare inputs + log prompt size
            # --------------------------------------------------
            format_instruction = self.parser.get_format_instructions()

            approx_doc_tokens = _approx_tokens(combined_docs)
            approx_fmt_tokens = _approx_tokens(format_instruction)
            approx_total = approx_doc_tokens + approx_fmt_tokens

            self.log.info(
                "Prepared document comparison prompt",
                combined_docs_chars=len(combined_docs),
                format_instruction_chars=len(format_instruction),
                approx_tokens_combined_docs=approx_doc_tokens,
                approx_tokens_format_instruction=approx_fmt_tokens,
                approx_tokens_total=approx_total,
            )

            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": format_instruction
            }

            # --------------------------------------------------
            # Step 2: Invoke LLM chain
            # --------------------------------------------------
            self.log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)

            self.log.info(
                "Chain invoked successfully",
                response_type=str(type(response)),
                response_preview=str(response)[:200],
            )

            # --------------------------------------------------
            # Step 3: Normalize & validate into SummaryResponse
            # --------------------------------------------------
            try:
                # Preferred path: response is already Python (e.g., list[dict])
                parsed_models = SummaryResponse.model_validate(response).root

            except ValidationError as ve:
                # Fallback path: response may be a raw string or wrong structure.
                self.log.warning(
                    "Schema validation failed; attempting one JSON repair call",
                    error=str(ve),
                )

                # If response is not a string, convert to string before repair
                repaired_json = self._repair_to_valid_json(response)

                parsed_models = SummaryResponse.model_validate_json(repaired_json).root

            except json.JSONDecodeError as je:
                # In case we ever try to parse JSON directly and it fails
                self.log.warning(
                    "JSON decoding failed; attempting one JSON repair call",
                    error=str(je),
                )
                repaired_json = self._repair_to_valid_json(response)
                parsed_models = SummaryResponse.model_validate_json(repaired_json).root

            # --------------------------------------------------
            # Step 4: Convert to DataFrame (once)
            # --------------------------------------------------
            df = pd.DataFrame([m.model_dump() for m in parsed_models])

            if "Page" in df.columns:
                df = df.sort_values("Page").reset_index(drop=True)

            self.log.info("Comparison DataFrame created", rows=len(df))
            return df
        
        except Exception as e:
            self.log.exception("Error in compare_documents")
            raise DocumentPortalException("Error comparing documents", sys)