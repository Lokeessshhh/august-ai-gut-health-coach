#!/usr/bin/env python3
"""
Q&A Dataset Generator using NVIDIA LLM API

This script reads text files from a CSV manifest and generates 150 question-answer
pairs from each file using NVIDIA's LLM API. The output is saved as a JSON dataset.

Requirements:
- sources_manifest.csv with 'raw_text_filename' column
- Text files referenced in the CSV
- NVIDIA API access
"""

import os
import json
import csv
import logging
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QADatasetGenerator:
    """Generates Q&A datasets from text files using NVIDIA LLM API."""
    
    def __init__(self, api_key: str):
        """
        Initialize the generator with NVIDIA API credentials.
        
        Args:
            api_key: NVIDIA API key for authentication
        """
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.qa_dataset = []
        
    def read_manifest(self, manifest_path: str = "sources_manifest.csv") -> List[str]:
        """
        Read the CSV manifest file and extract text filenames.
        
        Args:
            manifest_path: Path to the CSV manifest file
            
        Returns:
            List of text filenames to process
            
        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If required column is missing
        """
        try:
            filenames = []
            with open(manifest_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                if 'raw_text_filename' not in reader.fieldnames:
                    raise ValueError("CSV must contain 'raw_text_filename' column")
                
                for row in reader:
                    filename = row['raw_text_filename'].strip()
                    if filename:
                        filenames.append(filename)
                        
            logger.info(f"Found {len(filenames)} files in manifest")
            return filenames
            
        except FileNotFoundError:
            logger.error(f"Manifest file '{manifest_path}' not found")
            raise
        except Exception as e:
            logger.error(f"Error reading manifest: {e}")
            raise
    
    def read_text_file(self, filename: str) -> Optional[str]:
        """
        Read content from a text file with error handling.
        
        Args:
            filename: Path to the text file
            
        Returns:
            File content as string, or None if error occurred
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                
            if not content:
                logger.warning(f"File '{filename}' is empty")
                return None
                
            logger.info(f"Read {len(content)} characters from '{filename}'")
            return content
            
        except FileNotFoundError:
            logger.error(f"Text file '{filename}' not found")
            return None
        except UnicodeDecodeError:
            logger.error(f"Unable to decode '{filename}' as UTF-8")
            return None
        except Exception as e:
            logger.error(f"Error reading '{filename}': {e}")
            return None
    
    def generate_qa_from_llm(self, text_content: str, source_name: str, 
                           batch_size: int = 10) -> List[Dict]:
        """
        Generate Q&A pairs from text content using NVIDIA LLM API.
        
        Args:
            text_content: Source text to generate Q&A from
            source_name: Name of the source file
            batch_size: Number of Q&A pairs to generate per API call
            
        Returns:
            List of Q&A dictionaries
        """
        qa_pairs = []
        total_pairs = 150
        
        # Split text into chunks if it's too long (approximate token limit handling)
        max_chunk_size = 3000  # Conservative estimate for context window
        text_chunks = self._split_text(text_content, max_chunk_size)
        
        pairs_per_chunk = max(1, total_pairs // len(text_chunks))
        
        for i, chunk in enumerate(text_chunks):
            # Calculate how many pairs to generate from this chunk
            remaining_pairs = total_pairs - len(qa_pairs)
            pairs_to_generate = min(pairs_per_chunk, remaining_pairs)
            
            if pairs_to_generate <= 0:
                break
                
            logger.info(f"Generating {pairs_to_generate} Q&A pairs from chunk {i+1}/{len(text_chunks)} of '{source_name}'")
            
            # Generate Q&A pairs in batches
            for batch_start in range(0, pairs_to_generate, batch_size):
                batch_count = min(batch_size, pairs_to_generate - batch_start)
                
                try:
                    batch_qa = self._generate_qa_batch(chunk, source_name, batch_count)
                    qa_pairs.extend(batch_qa)
                    
                    if len(qa_pairs) >= total_pairs:
                        break
                        
                except Exception as e:
                    logger.error(f"Error generating batch for '{source_name}': {e}")
                    continue
            
            if len(qa_pairs) >= total_pairs:
                break
        
        # Ensure we have exactly 150 pairs (trim excess)
        qa_pairs = qa_pairs[:total_pairs]
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs for '{source_name}'")
        
        return qa_pairs
    
    def _split_text(self, text: str, max_size: int) -> List[str]:
        """
        Split text into manageable chunks for processing.
        
        Args:
            text: Text to split
            max_size: Maximum size per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_qa_batch(self, text_chunk: str, source_name: str, 
                          count: int) -> List[Dict]:
        """
        Generate a batch of Q&A pairs using the LLM API.
        
        Args:
            text_chunk: Text content to generate Q&A from
            source_name: Name of the source
            count: Number of Q&A pairs to generate
            
        Returns:
            List of Q&A dictionaries
        """
        prompt = f"""Based on the following text, generate exactly {count} diverse question-answer pairs. 
Each question should be clear and specific, and each answer should be comprehensive and informative.
The questions should cover different aspects of the content including main concepts, details, implications, and applications.

Text content:
{text_chunk}

Please format your response as a JSON array where each object has this structure:
{{
    "question": "Your question here",
    "answer": "Your detailed answer here"
}}

Generate exactly {count} question-answer pairs:"""

        try:
            # Call the NVIDIA LLM API
            completion = self.client.chat.completions.create(
                model="meta/llama3-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Slightly higher for more diverse questions
                top_p=1,
                max_tokens=1024,
                stream=False  # Disable streaming for easier JSON parsing
            )
            
            response_text = completion.choices[0].message.content
            
            # Parse the JSON response
            qa_pairs = self._parse_llm_response(response_text, source_name)
            return qa_pairs
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str, source_name: str) -> List[Dict]:
        """
        Parse LLM response and convert to required format.
        
        Args:
            response: Raw LLM response
            source_name: Name of the source file
            
        Returns:
            List of formatted Q&A dictionaries
        """
        qa_pairs = []
        
        try:
            # Try to extract JSON from the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON array found in response")
                return []
            
            json_str = response[json_start:json_end]
            parsed_qa = json.loads(json_str)
            
            # Convert to required format
            for item in parsed_qa:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    qa_pair = {
                        "instruction": item['question'],
                        "response": f"{item['answer']} [{source_name}]",
                        "sources": [source_name],  # Using filename as source
                        "tone": "supportive, friendly"
                    }
                    qa_pairs.append(qa_pair)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract Q&A pairs manually as fallback
            qa_pairs = self._manual_qa_extraction(response, source_name)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            
        return qa_pairs
    
    def _manual_qa_extraction(self, response: str, source_name: str) -> List[Dict]:
        """
        Fallback method to manually extract Q&A pairs from response.
        
        Args:
            response: Raw LLM response
            source_name: Name of the source file
            
        Returns:
            List of Q&A dictionaries
        """
        qa_pairs = []
        
        # Simple pattern matching for Q&A extraction
        lines = response.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            
            if line.lower().startswith(('question:', 'q:', '**question')):
                current_question = line.split(':', 1)[-1].strip()
            elif line.lower().startswith(('answer:', 'a:', '**answer')) and current_question:
                answer = line.split(':', 1)[-1].strip()
                
                qa_pair = {
                    "instruction": current_question,
                    "response": f"{answer} [{source_name}]",
                    "sources": [source_name],
                    "tone": "supportive, friendly"
                }
                qa_pairs.append(qa_pair)
                current_question = None
        
        logger.info(f"Manually extracted {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def process_files(self, filenames: List[str]) -> None:
        """
        Process all text files and generate Q&A datasets.
        
        Args:
            filenames: List of text filenames to process
        """
        total_files = len(filenames)
        processed_files = 0
        
        for i, filename in enumerate(filenames, 1):
            logger.info(f"Processing file {i}/{total_files}: {filename}")
            
            # Read text content
            text_content = self.read_text_file(filename)
            if not text_content:
                logger.warning(f"Skipping '{filename}' due to read error")
                continue
            
            # Generate Q&A pairs
            try:
                qa_pairs = self.generate_qa_from_llm(text_content, filename)
                self.qa_dataset.extend(qa_pairs)
                processed_files += 1
                
                logger.info(f"Successfully processed '{filename}' - "
                          f"Generated {len(qa_pairs)} Q&A pairs")
                
            except Exception as e:
                logger.error(f"Failed to process '{filename}': {e}")
                continue
        
        logger.info(f"Completed processing: {processed_files}/{total_files} files successful")
        logger.info(f"Total Q&A pairs generated: {len(self.qa_dataset)}")
    
    def save_dataset(self, output_file: str = "qa_dataset.json") -> None:
        """
        Save the generated Q&A dataset to a JSON file.
        
        Args:
            output_file: Output filename for the dataset
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(self.qa_dataset, file, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset saved to '{output_file}' with {len(self.qa_dataset)} Q&A pairs")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise


def main():
    """Main execution function."""
    
    # Configuration
    API_KEY = "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"
    MANIFEST_FILE = "sources_manifest.csv"
    OUTPUT_FILE = "qa_dataset.json"
    
    logger.info("Starting Q&A dataset generation")
    
    try:
        # Initialize the generator
        generator = QADatasetGenerator(API_KEY)
        
        # Read the manifest file
        logger.info(f"Reading manifest: {MANIFEST_FILE}")
        filenames = generator.read_manifest(MANIFEST_FILE)
        
        if not filenames:
            logger.warning("No files found in manifest")
            return
        
        # Process all files
        generator.process_files(filenames)
        
        # Save the dataset
        generator.save_dataset(OUTPUT_FILE)
        
        # Print summary
        print(f"\n{'='*50}")
        print("GENERATION COMPLETE")
        print(f"{'='*50}")
        print(f"Files processed: {len([f for f in file_info_list if Path(f['filename']).exists()])}")
        print(f"Total Q&A pairs: {len(generator.qa_dataset)}")
        print(f"Output saved to: {OUTPUT_FILE}")
        print(f"Log saved to: qa_generation.log")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted. Partial results may be available.")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# Example usage and testing functions
def create_sample_manifest():
    """Create a sample manifest file for testing."""
    sample_data = [
        {"raw_text_filename": "sample1.txt"},
        {"raw_text_filename": "sample2.txt"},
        {"raw_text_filename": "sample3.txt"}
    ]
    
    with open("sources_manifest.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["raw_text_filename"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    print("Sample manifest created: sources_manifest.csv")


def create_sample_text_files():
    """Create sample text files for testing."""
    sample_texts = {
        "sample1.txt": """
Artificial Intelligence (AI) is a rapidly evolving field that focuses on creating 
intelligent machines capable of performing tasks that typically require human 
intelligence. These tasks include learning, reasoning, problem-solving, perception, 
and language understanding. AI has applications across various industries including 
healthcare, finance, transportation, and entertainment.

Machine Learning is a subset of AI that enables computers to learn and improve 
from experience without being explicitly programmed. Deep Learning, a subset of 
Machine Learning, uses neural networks with multiple layers to model and understand 
complex patterns in data.
        """,
        
        "sample2.txt": """
Climate change refers to long-term shifts in global temperatures and weather patterns. 
While climate variations are natural, scientific evidence shows that human activities 
have been the dominant driver of climate change since the mid-20th century.

The primary cause is the emission of greenhouse gases, particularly carbon dioxide 
from burning fossil fuels. Effects include rising sea levels, extreme weather events, 
ecosystem disruption, and threats to food security. Mitigation strategies include 
renewable energy adoption, energy efficiency improvements, and sustainable practices.
        """,
        
        "sample3.txt": """
Python is a high-level, interpreted programming language known for its simplicity 
and readability. Created by Guido van Rossum in 1991, Python emphasizes code 
readability and allows programmers to express concepts in fewer lines of code.

Python supports multiple programming paradigms including procedural, object-oriented, 
and functional programming. It has a comprehensive standard library and a vast 
ecosystem of third-party packages. Python is widely used in web development, 
data science, artificial intelligence, automation, and scientific computing.
        """
    }
    
    for filename, content in sample_texts.items():
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content.strip())
    
    print("Sample text files created: sample1.txt, sample2.txt, sample3.txt")


def setup_test_environment():
    """Set up a complete test environment."""
    print("Setting up test environment...")
    create_sample_manifest()
    create_sample_text_files()
    print("Test environment ready!")
    print("Run the main script to generate Q&A dataset.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_test_environment()
    else:
        exit(main())