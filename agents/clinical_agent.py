# agents/clinical_agent.py

from models.deepseek_model import DeepSeekModel
from retrieval.medical_retriever import MedicalRetriever
from tools.tool_manager import ToolManager
import re


class ClinicalAgent:

    def __init__(self):
        self.deepseek = DeepSeekModel()
        self.tools = ToolManager()

        # ✅ Load retriever once
        try:
            self.retriever = MedicalRetriever(top_k=2)
        except Exception as e:
            print("Retriever initialization failed:", e)
            self.retriever = None

    # ---------------------------
    # ANSWER EXTRACTION
    # ---------------------------
    def extract_answer(self, text):
        match = re.search(r"\b[A-E]\b", text)
        if match:
            return match.group(0)
        return None

    # ---------------------------
    # TOOL USAGE
    # ---------------------------
    def call_tools_if_needed(self, question):
        tools_used = {}

        q = question.lower()

        if "metformin" in q:
            tools_used["drug_info"] = self.tools.run("lookup_drug", "metformin")

        if "warfarin" in q and "aspirin" in q:
            tools_used["interaction"] = self.tools.run(
                "check_interaction",
                "warfarin",
                "aspirin"
            )

        return tools_used

    # ---------------------------
    # RETRIEVAL (LESS NOISE)
    # ---------------------------
    def retrieve_context(self, question):
        try:
            if self.retriever is None:
                return ""

            docs = self.retriever.retrieve(question)

            if not docs:
                return ""

            # ✅ Use ONLY top document (important improvement)
            return docs[0]

        except Exception as e:
            print("Retrieval failed:", e)
            return ""

    # ---------------------------
    # MAIN SOLVER
    # ---------------------------
    def solve(self, question):

        # 🔹 Step 1: Context
        context = self.retrieve_context(question)

        # 🔹 Step 2: Tools
        tools_used = self.call_tools_if_needed(question)

        # 🔹 Step 3: Improved Prompt (OPTION-AWARE)
        prompt = f"""
You are a clinical pharmacology expert.

Answer the following multiple choice question.

Context:
{context}

Tool outputs:
{tools_used}

Question:
{question}

Instructions:
- Carefully evaluate ALL options (A–E)
- Eliminate incorrect options
- Select the MOST correct answer based on clinical reasoning
- Avoid guessing

IMPORTANT:
- Output ONLY one letter: A, B, C, D, or E
- Do NOT explain

Answer:
"""

        try:
            # ✅ First attempt
            response = self.deepseek.generate(prompt, max_tokens=128)
            answer = self.extract_answer(response)

            # 🔥 Retry if model fails to follow format
            if not answer:
                retry_prompt = prompt + "\nREMEMBER: Output ONLY A, B, C, D, or E."
                response = self.deepseek.generate(retry_prompt, max_tokens=64)
                answer = self.extract_answer(response)

            final_answer = answer if answer else "A"

        except Exception as e:
            print("DeepSeek failed:", e)
            final_answer = "A"

        return {
            "question": question,
            "context": context,
            "tools_used": tools_used,
            "reasoning_trace": [
                {
                    "step": "single_call",
                    "response": final_answer
                }
            ],
            "final_answer": final_answer
        }
