# 047: Multi-Modal RAG & Complex Documents

## Overview

Traditional Retrieval-Augmented Generation (RAG) relies on text-only chunking, which often fails when processing complex documents. Documents are not merely linear streams of text; they are 2D spatial layouts containing tables, charts, diagrams, headers, and mathematical formulae. 

### The Flaws of Text-Only Chunking
When a parser naively extracts text from a PDF, it typically reads left-to-right, top-to-bottom. This process strips away the structural context:
- **Table Rows:** Multi-line cells break alignment, rendering tabular data into an incoherent word salad.
- **Chart Legends:** Spatial proximity between a legend and a chart is lost.
- **Math Equations:** Formulae often contain superscripts, subscripts, and symbols that do not linearize cleanly.

### DeepDoc & Layout Analysis
To solve this, modern systems use **DeepDoc** layout analysis and **Table Structure Recognition (TSR)**. These models process the document visually (often using architectures like LayoutLM) to identify structural blocks and their spatial relationships before converting them to text.

### Layout-Aware Chunking
Layout-aware chunking preserves the document structure. Instead of splitting text by character count, it groups elements by their spatial proximity and semantic boundaries (e.g., keeping an entire table as a single chunk or combining a chart image with its caption).
