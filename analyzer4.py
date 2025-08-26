import os
import sys
import shutil
import pandas as pd

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA


EXCEL_FILE = "insurance sheet.xlsx"        
CHROMA_DIR  = "./chroma_db"                
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"                   
K_RETRIEVE = 6                             


def read_all_sheets(xlsx_path: str) -> pd.DataFrame:
    
    if not os.path.isfile(xlsx_path):
        print(f"Excel file not found: {xlsx_path}")
        sys.exit(1)

    xls = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=object)
            if not df.empty:
                df["__sheet__"] = sheet
                frames.append(df)
        except Exception as e:
            print(f"Skipping sheet '{sheet}': {e}")

    if not frames:
        print("No readable sheets found.")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def rows_to_documents(df: pd.DataFrame) -> list[Document]:
   
    docs: list[Document] = []
    columns = list(df.columns)
    if "__sheet__" in columns:
        columns.remove("__sheet__")
        columns = ["__sheet__"] + columns

    for i, row in df.iterrows():
        parts = []
        for col in columns:
            val = row.get(col)
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")
        text = " | ".join(parts)
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={"row_index": int(i)}))
    return docs


def build_or_load_chroma(docs: list[Document]) -> Chroma:
    
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
            return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception:
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


def start_chat(vectordb: Chroma):
    retriever = vectordb.as_retriever(search_kwargs={"k": K_RETRIEVE})
    llm = Ollama(model=OLLAMA_MODEL)  

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",   
        return_source_documents=True,
    )

    print("\nAsk questions about your Excel data.")
    print("   (type 'exit' to quit, 'rebuild' to rebuild vectors)\n")

    while True:
        q = input("Query: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        if q.lower() == "rebuild":
            
            print("🔁 Rebuilding vectors from Excel...")
            df = read_all_sheets(EXCEL_FILE)
            docs = rows_to_documents(df)
            
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            new_db = build_or_load_chroma(docs)
            
            nonlocal_retriever = new_db.as_retriever(search_kwargs={"k": K_RETRIEVE})
            qa.retriever = nonlocal_retriever
            print("Rebuild complete.")
            continue

        try:
            resp = qa({"query": q})
            answer = resp.get("result", "").strip()
            sources = resp.get("source_documents", []) or []
            print("\nAnswer:\n", answer or "(no answer)\n")
            if sources:
                print("📎 Top matches:")
                for i, s in enumerate(sources, 1):
                    prev = s.page_content[:160].replace("\n", " ")
                    print(f"  {i}. {prev}...")
            print()
        except Exception as e:
            print("Error:", e)


def main():
    print("Reading Excel (all sheets):", EXCEL_FILE)
    df = read_all_sheets(EXCEL_FILE)

    print(f"Building / Loading Chroma at {CHROMA_DIR} ...")
    docs = rows_to_documents(df)
    vectordb = build_or_load_chroma(docs)

    print("Launching local Q&A (Ollama:", OLLAMA_MODEL, ")")
    start_chat(vectordb)

if __name__ == "__main__":
    
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
