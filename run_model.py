import anthropic
from dotenv import load_dotenv
import diskcache
from anthropic.types import Usage
from pypdf import PdfReader
import os
import re
import concurrent.futures
from tqdm import tqdm  # For progress tracking


load_dotenv()

PATH = "/Users/Charlie/paper-retraction-detection/misc-data/useful-retractions-ft"

PROMPT = """Paper:

{paper_text}

Do an expert-level peer review on this. Make sure to be harsh but fair. After your review give a probability that the paper will be retracted. Put the probability as a percentage in double brackets."""

client = anthropic.Anthropic()

cache = diskcache.Cache(".cache")

@cache.memoize()
def get_text_from_pdf(path: str):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

TOTAL_COST = 0

@cache.memoize()
def _run_anthropic(text: str):
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {"role": "user", "content": text},
        ],
        max_tokens=6000,
        thinking={"budget_tokens": 2000, "type": "enabled"},
    )
    return response

def get_cost(usage: Usage):
    return (usage.input_tokens * 3 + usage.output_tokens * 15) / 1000 ** 2

def extract_percentage(text: str):
    # Find the double brackets and extract the percentage
    match = re.search(r"\[\[(\d+(?:\.\d+)?)%?\]\]", text)
    if match:
        try:
            return float(match.group(1))
        except:
            return 0.0
    return 0.0

def run_pdf(path: str):
    global TOTAL_COST
    try:
        text = get_text_from_pdf(path)
        response = _run_anthropic(PROMPT.format(paper_text=text))
        TOTAL_COST += get_cost(response.usage)
        text_response = response.content[1].text
        print(text_response)
        percentage = extract_percentage(text_response)
        return (os.path.basename(path), percentage)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return (os.path.basename(path), 0.0)

# Process all PDFs concurrently
def process_all_pdfs(pdf_dir, max_workers=10):
    pdf_paths = [os.path.join(pdf_dir, pdf) for pdf in os.listdir(pdf_dir) if pdf.endswith('.pdf')]
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_pdf, path): path for path in pdf_paths}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_paths), desc="Processing PDFs"):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error with {path}: {e}")
                results.append((os.path.basename(path), 0.0))
    
    # Sort results by percentage in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results

# Run the analysis on all PDFs
results = process_all_pdfs(PATH)

# Print sorted results
print("\nResults (sorted by retraction probability):")
print("-" * 50)
for pdf_name, percentage in results:
    print(f"{percentage:.2f}% - {pdf_name}")

print(f"\nTotal API cost: ${TOTAL_COST:.4f}")