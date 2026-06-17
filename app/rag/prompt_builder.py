from typing import List, Dict, Any


class PromptBuilder:
    """Build prompts for LLM with RAG context."""
    
    SYSTEM_PROMPT = """You are a Digimon expert assistant. Answer concisely and factually \
using only the information in the provided context. Do not add closing questions, emojis, \
or information not present in the context. If the context does not contain the answer, \
say so briefly."""
    
    @staticmethod
    def build_chat_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build a chat prompt with RAG context."""
        context_text = ""
        
        if context_chunks:
            context_text = "\n\nContext:\n"
            for i, chunk in enumerate(context_chunks, 1):
                chunk_text = chunk.get("payload", {}).get("chunk_text", "")
                digimon_name = chunk.get("payload", {}).get("name", "Unknown")
                context_text += f"\n{i}. {digimon_name}: {chunk_text}"
        
        prompt = f"""{PromptBuilder.SYSTEM_PROMPT}

{context_text}

User Question: {query}

Please provide a helpful and accurate answer based on the context above."""
        
        return prompt
