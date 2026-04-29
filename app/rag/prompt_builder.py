from typing import List, Dict, Any


class PromptBuilder:
    """Build prompts for LLM with RAG context."""
    
    SYSTEM_PROMPT = """You are a knowledgeable Digimon expert assistant. 
You help users learn about Digimons, their abilities, evolution paths, and characteristics.
Always be friendly and informative. Use the provided context to answer questions accurately.
If you don't know the answer based on the context, say so politely."""
    
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
