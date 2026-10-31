import io
import re
from typing import Dict, Any, List
import pypdf

class DocumentParser:
    """
    Parses resumes, technical documentation, and text files (PDF, TXT, MD).
    Extracts plain text and identifies major sections.
    """
    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        extracted_text = []
        for page_idx, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n\n".join(extracted_text)

    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Extracts typical resume/doc sections like Experience, Education, Skills, Projects.
        """
        common_headings = [
            "experience", "work history", "employment", "education", 
            "skills", "technical skills", "projects", "certifications", 
            "summary", "profile", "achievements"
        ]
        
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {"general": []}
        current_section = "general"

        for line in lines:
            stripped = line.strip().lower()
            # Check if line looks like a section header
            matched_heading = None
            for h in common_headings:
                if stripped.startswith(h) or stripped.endswith(h) or stripped == h:
                    matched_heading = h
                    break
            
            if matched_heading and len(stripped.split()) <= 4:
                current_section = matched_heading
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}
