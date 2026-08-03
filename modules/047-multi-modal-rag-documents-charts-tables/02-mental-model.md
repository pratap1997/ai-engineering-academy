# Mental Model: The Document Archivist

To understand Multi-Modal RAG, imagine a **Document Archivist** whose job is to index a vast library of complex manuscripts, financial reports, and blueprints.

### The Linear Reader (Traditional RAG)
A traditional RAG system acts like a blindfolded reader who traces their finger across a page, reading words sequentially. When they encounter a data table, they read across the columns blindly, creating a meaningless string of numbers and headers. They don't know they are reading a table because they cannot *see* the structure.

### The Document Archivist (Multi-Modal RAG)
The Document Archivist takes a visual step back. Before reading a single word, they analyze the geometry of the page.
1. **Zoning:** They draw boxes around distinct regions: "This is a paragraph," "This is a table," "This is a diagram."
2. **Contextualizing:** They understand that a caption belongs to the image above it, not the paragraph below it.
3. **Preserving Layout:** When indexing the table, they preserve the grid structure, perhaps transcribing it into a structured format like CSV or HTML.

By recognizing that a document is a **2D spatial layout of text blocks, tables, images, and headers**, the archivist preserves layout context during chunking. When someone asks a question, they retrieve not just raw text, but the structured block of information needed to answer the query accurately.
