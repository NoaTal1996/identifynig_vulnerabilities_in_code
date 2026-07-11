import re
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

def extract_function_body(code_str, func_name):
    """
    Extracts the function body of `func_name` from C source code `code_str`
    using curly-brace matching.
    """
    # Pattern to match function declaration and the opening brace
    pattern = re.escape(func_name) + r'\s*\([^)]*\)\s*(?:const\s*)?\{'
    match = re.search(pattern, code_str)
    if not match:
        pattern = re.escape(func_name) + r'\s*\([^)]*\)\s*\{'
        match = re.search(pattern, code_str)
        if not match:
            return None
            
    start_idx = match.end() - 1  # Index of the open brace '{'
    
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(code_str)):
        if code_str[i] == '{':
            brace_count += 1
        elif code_str[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
                
    if end_idx != -1:
        header = code_str[match.start():start_idx]
        body = code_str[start_idx:end_idx]
        return header + body
    return None

def load_aligned_juliet(toy=False):
    """
    Loads CCompote/Juliet_LLVM and hwiwonl/nist-juliet-c, aligns them 1-to-1
    by matching filenames and function names, and preserves Juliet_LLVM's splits.
    """
    print("Loading Juliet LLVM IR dataset...")
    juliet_llvm = load_dataset("CCompote/Juliet_LLVM")
    print("Loading Juliet C source dataset...")
    juliet_c = load_dataset("hwiwonl/nist-juliet-c")
    
    # Combine all C splits to build a lookup map
    c_rows = []
    for split in juliet_c.keys():
        c_rows.extend(juliet_c[split])
        
    c_map = {}
    for row in c_rows:
        key = re.sub(r'\.c(pp)?$', '', row['instance_id'].lower())
        c_map[key] = row
        
    aligned_dataset = {}
    
    for split in ['train', 'validation', 'test']:
        print(f"Aligning Juliet {split} split...")
        llvm_split = juliet_llvm[split]
        if toy:
            llvm_split = llvm_split.select(range(min(50, len(llvm_split))))
            
        aligned_rows = []
        for row in llvm_split:
            file_base = re.sub(r'-(good|bad)$', '', row['file']).lower()
            c_row = c_map.get(file_base)
            if not c_row:
                continue
                
            label = int(row['label'])
            fun_name = row['fun_name']
            
            # Extract C source code
            c_pool = c_row['bad'] if label == 1 else c_row['good']
            c_source = extract_function_body(c_pool, fun_name)
            
            if not c_source:
                # Fallback to the whole pool if exact brace matching fails
                c_source = c_pool
                
            aligned_rows.append({
                'file': row['file'],
                'fun_name': fun_name,
                'source_code': c_source,
                'llvm_ir': row['llvm_ir_function'],
                'label': label
            })
            
        aligned_dataset[split] = Dataset.from_list(aligned_rows)
        print(f"Juliet {split} aligned successfully: {len(aligned_dataset[split])} rows.")
        
    return DatasetDict(aligned_dataset)

def load_aligned_realvul(toy=False):
    """
    Loads CCompote/CompRealVul_C (source) and CCompote/CompRealVul_LLVM (IR),
    aligns them 1-to-1 by matching function names, and preserves CompRealVul_LLVM's splits.
    """
    print("Loading CompRealVul C source dataset...")
    realvul_c = load_dataset("CCompote/CompRealVul_C", split="train")
    print("Loading CompRealVul LLVM IR dataset...")
    realvul_llvm = load_dataset("CCompote/CompRealVul_LLVM")
    
    # Build a lookup map for C source code
    c_map = {}
    for row in realvul_c:
        c_map[row['name']] = row
        
    aligned_dataset = {}
    
    for split in ['train', 'validation', 'test']:
        print(f"Aligning CompRealVul {split} split...")
        llvm_split = realvul_llvm[split]
        if toy:
            llvm_split = llvm_split.select(range(min(50, len(llvm_split))))
            
        aligned_rows = []
        for row in llvm_split:
            fun_name = row['fun_name']
            c_row = c_map.get(fun_name)
            if not c_row:
                continue
                
            label = int(row['label'])
            
            aligned_rows.append({
                'fun_name': fun_name,
                'source_code': c_row['code'],
                'llvm_ir': row['llvm_ir_function'],
                'label': label
            })
            
        aligned_dataset[split] = Dataset.from_list(aligned_rows)
        print(f"CompRealVul {split} aligned successfully: {len(aligned_dataset[split])} rows.")
        
    return DatasetDict(aligned_dataset)

def get_dataset(dataset_name, representation, toy=False):
    """
    Main entrypoint to load a split dataset.
    dataset_name: 'juliet' or 'realvul'
    representation: 'source' or 'llvm_ir'
    """
    if dataset_name.lower() == 'juliet':
        aligned = load_aligned_juliet(toy=toy)
    elif dataset_name.lower() == 'realvul':
        aligned = load_aligned_realvul(toy=toy)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
    # Map raw text features to standard Hugging Face format
    text_col = 'source_code' if representation.lower() == 'source' else 'llvm_ir'
    
    def format_row(examples):
        return {
            'text': examples[text_col],
            'label': examples['label']
        }
        
    formatted = {}
    for split in ['train', 'validation', 'test']:
        formatted[split] = aligned[split].map(
            format_row,
            remove_columns=aligned[split].column_names,
            batched=True
        )
        
    return DatasetDict(formatted)

if __name__ == "__main__":
    # Test the loader locally with the toy setting
    print("=== Testing Juliet Source ===")
    juliet_src = get_dataset('juliet', 'source', toy=True)
    print("Train sample:", juliet_src['train'][0])
    
    print("\n=== Testing CompRealVul IR ===")
    realvul_ir = get_dataset('realvul', 'llvm_ir', toy=True)
    print("Train sample:", realvul_ir['train'][0])
