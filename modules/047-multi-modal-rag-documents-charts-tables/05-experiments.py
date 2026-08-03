import sys
import os

# Import the implementation module
sys.path.append(os.path.dirname(__file__))
from module_047_impl import BoundingBox, DocumentBlock, TableStructureParser, LayoutAwareChunker, MultiModalDocRetriever

def run_experiments():
    print("Running Multi-Modal RAG Experiments...")
    
    # Experiment 1: Bounding Box IoU Accuracy
    print("\nExperiment 1: Bounding Box IoU Accuracy")
    b1 = BoundingBox(0, 0, 10, 10)
    b2 = BoundingBox(5, 5, 15, 15)
    iou = b1.iou(b2)
    print(f"  Box 1: {b1.area()} sq units, Box 2: {b2.area()} sq units")
    print(f"  IoU Score: {iou:.4f} (Expected: ~0.1428)")
    
    # Experiment 2: Table-to-JSON (CSV) Conversion Fidelity
    print("\nExperiment 2: Table Conversion Fidelity")
    cells = [
        {'row': 0, 'col': 0, 'text': 'Q1'}, {'row': 0, 'col': 1, 'text': 'Q2'},
        {'row': 1, 'col': 0, 'text': '$100M'}, {'row': 1, 'col': 1, 'text': '$120M'}
    ]
    parser = TableStructureParser()
    csv_out = parser.parse_grid_to_csv(cells)
    print("  Input grid parsed to CSV format:")
    print("  " + csv_out.replace("\n", "\n  "))
    
    # Experiment 3: Text-only vs Layout-aware chunking recall on financial tables
    print("\nExperiment 3: Layout-Aware Chunking")
    blocks = [
        DocumentBlock('text', 'Financial Results', BoundingBox(0, 0, 100, 20)),
        DocumentBlock('table', csv_out, BoundingBox(0, 30, 100, 100)),
        DocumentBlock('text', 'End of report', BoundingBox(0, 110, 100, 130))
    ]
    chunker = LayoutAwareChunker()
    chunks = chunker.chunk(blocks)
    print(f"  Generated {len(chunks)} distinct semantic chunks.")
    print(f"  Table chunk preserved independently: {'[TABLE]' in chunks[1]}")
    
    # Experiment 4: Spatial chunking retrieval precision
    print("\nExperiment 4: Spatial Chunking Retrieval Precision")
    retriever = MultiModalDocRetriever(chunks)
    results = retriever.search("Q2 $120M")
    print(f"  Query: 'Q2 $120M'")
    print(f"  Top Match:")
    print("  " + results[0].replace("\n", "\n  ") if results else "  None")

if __name__ == "__main__":
    run_experiments()
