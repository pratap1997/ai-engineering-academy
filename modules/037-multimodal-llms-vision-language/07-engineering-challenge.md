# Engineering Challenge: Multimodal Layout Parsing

## Problem Statement
Standard text extraction models strip away spatial positioning, ruining tables, sidebars, and invoices. Your challenge is to build a simulated Vision-Language Model that classifies the layout intent of document patches.

Given an image of a document (represented as a 2D grid), you must process it using the VLM pipeline and identify which patches contain "Headers", "Tables", or "Paragraphs" based on combined structural rules.

## Requirements
1. **Initialize a Document VLM**: Extend the `SimpleVLM` architecture.
2. **Patchify**: Divide a simulated $64 \times 64$ document into $16 \times 16$ patches.
3. **Cross-Modality Prompting**: Pass the prompt `Extract <IMAGE> layout`.
4. **Classification**: Read the projected feature output and classify each visual token. 

*No hints are provided.* Validate your implementation against edge cases where text spans across multiple patches.
