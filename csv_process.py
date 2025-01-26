import asyncio
import langchain
from typing import List, Dict, Optional
from aiohttp import ClientSession
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.cache import InMemoryCache
import pandas as pd
import logging
from tqdm import tqdm
import os
from dotenv import load_dotenv
import argparse
import json

# Load environment variables
load_dotenv()

# Configure cache
langchain.llm_cache = InMemoryCache()

class AsyncCSVProcessor:
    def __init__(self, system_prompt_path: str, batch_size: int = 5, max_concurrent: int = 3):
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
        # Initialize Azure OpenAI with caching
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0,
            cache=True
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    @staticmethod
    def parse_llm_response(response_text: str) -> Dict:
        try:
            result = {
                'validity': 'N/A',
                'category': 'N/A',
                'region': 'N/A',
                'city': 'N/A',
                'area': 'N/A',
                'brand': 'N/A',
                'series': 'N/A',
                'dealer': 'N/A',
                'tag1': '',
                'tag1_valid': 'false',
                'tag2': '',
                'tag2_valid': 'false',
                'tag3': '',
                'tag3_valid': 'false',
                'sentiment': 'Neutral',
                'business_area': 'N/A',
                'user_id': 'N/A',
                'web_link': 'N/A'
            }
            
            lines = response_text.split('\n')
            for line in lines:
                if '：' in line:  # Handle Chinese colon
                    key, value = line.split('：', 1)
                elif ':' in line:  # Handle English colon
                    key, value = line.split(':', 1)
                else:
                    continue
                    
                key = key.strip()
                value = value.strip()
                
                if '文本有效性' in key or 'Validity' in key:
                    result['validity'] = value
                elif '链路阶段' in key or 'Category' in key:
                    result['category'] = value
                elif 'Region' in key:
                    result['region'] = value
                elif 'City' in key:
                    result['city'] = value
                elif 'Area' in key:
                    result['area'] = value
                elif '品牌' in key or 'Brand' in key:
                    result['brand'] = value
                elif '车系' in key or 'Series' in key:
                    result['series'] = value
                elif '经销商' in key or 'Dealer' in key:
                    result['dealer'] = value
                elif '标签' in key or 'Tags' in key:
                    if '标签 1' in line or 'Tag 1' in line:
                        tag, valid = value.split('-')
                        result['tag1'] = tag.strip()
                        result['tag1_valid'] = valid.strip()
                    elif '标签 2' in line or 'Tag 2' in line:
                        tag, valid = value.split('-')
                        result['tag2'] = tag.strip()
                        result['tag2_valid'] = valid.strip()
                    elif '标签 3' in line or 'Tag 3' in line:
                        tag, valid = value.split('-')
                        result['tag3'] = tag.strip()
                        result['tag3_valid'] = valid.strip()
                elif '情感倾向' in key or 'Sentimental' in key:
                    result['sentiment'] = value
                elif '业务领域' in key or 'Business Area' in key:
                    result['business_area'] = value
                elif 'User ID' in key:
                    result['user_id'] = value
                elif 'Web Link' in key:
                    result['web_link'] = value
                    
            return result
        except Exception as e:
            logging.error(f"Error parsing LLM response: {str(e)}")
            return result

    async def process_row(self, row: Dict) -> Dict:
        """Process a single row with optimized prompting"""
        async with self.semaphore:
            try:
                input_text = f"ID: {row['id']}\n{row['title']}"
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=input_text)
                ]
                response = await self.llm.agenerate([messages])
                parsed_result = self.parse_llm_response(response.generations[0][0].text)
                parsed_result['id'] = row['id']
                return parsed_result
            except Exception as e:
                logging.error(f"Error processing row {row['id']}: {str(e)}")
                return {'id': row['id'], 'error': str(e)}

    async def process_batch(self, rows: List[Dict]) -> List[Dict]:
        """Process a batch of rows concurrently"""
        tasks = [self.process_row(row) for row in rows]  # Removed system_prompt argument
        try:
            return await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            return [{'id': row['id'], 'error': str(e)} for row in rows]

    async def process_csv(self, file_path: str, output_path: str) -> bool:
        """Main method to process CSV file with real-time output"""
        try:
            logging.info(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')
            required_columns = ['id', 'title']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
                
            df = df[required_columns]
            rows = df.to_dict('records')
            
            # Create output file with headers
            pd.DataFrame(columns=self.parse_llm_response("").keys()).to_csv(
                output_path, index=False, encoding='utf-8'
            )
            
            total_batches = (len(rows) + self.batch_size - 1) // self.batch_size
            
            with tqdm(total=total_batches, desc="Processing batches") as pbar:
                for i in range(0, len(rows), self.batch_size):
                    batch = rows[i:i + self.batch_size]
                    batch_results = await self.process_batch(batch)
                    
                    # Write batch results to CSV
                    pd.DataFrame(batch_results).to_csv(
                        output_path, 
                        mode='a',
                        header=False,
                        index=False,
                        encoding='utf-8'
                    )
                    pbar.update(1)
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing CSV: {str(e)}")
            raise

def process_csv_file(file_path: str, system_prompt_path: str, output_path: str) -> bool:
    """Synchronous wrapper for async processing"""
    processor = AsyncCSVProcessor(system_prompt_path)
    return asyncio.run(processor.process_csv(file_path, output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files with Azure OpenAI')
    parser.add_argument('--input', '-i', 
                      type=str,
                      default="test.csv",
                      help='Input CSV file path')
    parser.add_argument('--prompt', '-p',
                      type=str, 
                      default="system_prompt.txt",
                      help='System prompt file path')
    parser.add_argument('--output', '-o',
                      type=str,
                      default=os.path.join("outputs", "processed_results.csv"),
                      help='Output CSV file path')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    input_file = "a_facebook.csv"
    os.makedirs("outputs", exist_ok=True)
    system_prompt_path = "system_prompt.txt"
    
    try:
        os.makedirs("outputs", exist_ok=True)
        output_path = args.output
        success = process_csv_file(args.input, args.prompt, output_path)
        if success:
            logging.info(f"Processing completed successfully. Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")