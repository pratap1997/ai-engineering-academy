import math
import json
import re
from typing import List, Dict, Any, Tuple, Optional

class BoundingBox:
    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        if x0 > x1 or y0 > y1:
            raise ValueError("Invalid coordinates")
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)
        
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        nx0 = max(self.x0, other.x0)
        ny0 = max(self.y0, other.y0)
        nx1 = min(self.x1, other.x1)
        ny1 = min(self.y1, other.y1)
        
        if nx0 < nx1 and ny0 < ny1:
            return BoundingBox(nx0, ny0, nx1, ny1)
        return None
        
    def iou(self, other: 'BoundingBox') -> float:
        inter = self.intersection(other)
        if not inter:
            return 0.0
        inter_area = inter.area()
        union_area = self.area() + other.area() - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

class DocumentBlock:
    def __init__(self, block_type: str, content: Any, bbox: BoundingBox):
        self.block_type = block_type  # text, table, image, header
        self.content = content
        self.bbox = bbox

class TableStructureParser:
    def __init__(self):
        pass
        
    def parse_grid_to_csv(self, cells: List[Dict[str, Any]]) -> str:
        if not cells:
            return ""
        max_row = max(cell['row'] for cell in cells)
        max_col = max(cell['col'] for cell in cells)
        
        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        for cell in cells:
            grid[cell['row']][cell['col']] = str(cell.get('text', '')).strip()
            
        lines = []
        for row in grid:
            escaped = []
            for c in row:
                if ',' in c or '"' in c:
                    escaped.append(f'"{c.replace("`", "")}"')
                else:
                    escaped.append(c)
            lines.append(",".join(escaped))
        return "\n".join(lines)
        
class LayoutAwareChunker:
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        
    def chunk(self, blocks: List[DocumentBlock]) -> List[str]:
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
        chunks = []
        current_chunk = []
        current_len = 0
        
        for block in sorted_blocks:
            if block.block_type == 'table':
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(f"[TABLE]\n{block.content}")
            elif block.block_type == 'image':
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(f"[IMAGE]\n{block.content}")
            elif block.block_type == 'header':
                text = str(block.content)
                if current_len + len(text) > self.max_chunk_size and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(f"# {text}")
                current_len += len(text)
            else:
                text = str(block.content)
                if current_len + len(text) > self.max_chunk_size and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(text)
                current_len += len(text)
                
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

class MultiModalDocRetriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        
    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_terms = query.lower().split()
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            words = re.findall(r'\w+', chunk_lower)
            score = sum(words.count(term) for term in query_terms)
            scored_chunks.append((score, chunk))
            
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored_chunks[:top_k] if score > 0]
