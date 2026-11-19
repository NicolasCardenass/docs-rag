# Retrieval-augmented generation

Retrieval-augmented generation (RAG) lets a language model answer questions using external
documents. Documents are split into chunks, and each chunk is converted into an embedding vector.

At question time, the question is embedded too, and the chunks with the most similar vectors are
retrieved from a vector database. These chunks are added to the prompt as context.

Because the answer is grounded on retrieved passages, RAG reduces hallucinations and allows the
model to cite its sources.
