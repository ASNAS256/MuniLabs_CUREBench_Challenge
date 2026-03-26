# agents/clinical_agent.py

from models.deepseek_model import DeepSeekModel
from retrieval.medical_retriever import retrieve_medical_context
from tools.tool_manager import ToolManager


class ClinicalAgent:

    def __init__(self):
        self.deepseek = DeepSeekModel()
        self.tools = ToolManager()

    # ✅ single model call (fast + stable)
    def generate(self, prompt):
        try:
            response = self.deepseek.generate(prompt)
            return response, "deepseek"
        except Exception as e:
            print("DeepSeek failed:", e)
            return "Error", "failed"

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

    def solve(self, question):

        # 🔹 Step 1: Retrieve context (RAG)
        context = retrieve_medical_context(question, top_k=3)

        # 🔹 Step 2: Tool usage
        tools_used = self.call_tools_if_needed(question)

        # 🔥 SINGLE PROMPT (NO PLANNING, NO STEPS, NO REFLECTION)
        prompt = f"""
You are a clinical pharmacology expert.

Use the context and tools to answer the question.

Context:
{context}

Tool outputs:
{tools_used}

Question:
{question}

Instructions:
- Think step by step internally
- Provide ONLY final reasoning (short)
- Give final answer as one letter: A, B, C, D, or E
"""

        response, model_used = self.generate(prompt)

        return {
            "question": question,
            "context": context,
            "tools_used": tools_used,
            "reasoning_trace": [
                {
                    "step": "single_call",
                    "model_used": model_used,
                    "response": response
                }
            ],
            "final_answer": response
        }