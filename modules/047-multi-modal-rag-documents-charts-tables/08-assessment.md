# Assessment

1. **Why does traditional text-only chunking struggle with PDFs?**
   *Answer:* It reads text linearly (left-to-right, top-to-bottom), which destroys the structure of multi-line tables, charts, and spatial relationships.

2. **What is the purpose of a Bounding Box in layout analysis?**
   *Answer:* It defines the 2D spatial coordinates (x0, y0, x1, y1) of a specific element (like a paragraph or table) on a page.

3. **How is Intersection over Union (IoU) calculated?**
   *Answer:* By dividing the area of overlap between two bounding boxes by the total area covered by both boxes combined.

4. **What is Table Structure Recognition (TSR)?**
   *Answer:* The process of identifying the grid structure, rows, and columns of a table from a visual document to reconstruct it accurately.

5. **Why should tables be isolated during chunking?**
   *Answer:* So they remain intact as a single contextual unit for the LLM, preventing rows or columns from being split across multiple chunks.

6. **What is Layout Entropy used for?**
   *Answer:* It measures the visual complexity of a document page based on the variety and distribution of different layout classes (text, images, tables).

7. **How does IBM Docling improve RAG?**
   *Answer:* By performing layout analysis to convert complex PDFs into clean, structured formats (like Markdown) before indexing.

8. **What does the "DeepDoc" engine do in RAGFlow?**
   *Answer:* It uses vision models and OCR to intelligently segment pages and preserve document hierarchy and layout.

9. **In the Document Archivist mental model, what is the first step before reading the text?**
   *Answer:* "Zoning" or analyzing the geometry of the page to draw boxes around distinct structural regions.

10. **Why are academic papers difficult for traditional RAG?**
    *Answer:* They rely heavily on multi-column layouts, mathematical formulae, and diagrams with captions, which linear text extraction scrambles.
