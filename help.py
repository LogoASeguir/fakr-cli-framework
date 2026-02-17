from memory.ingest import ingest_url
doc_id = ingest_url("https://docs.python.org/3/library/tkinter.html", kind="tk_docs", title="Tkinter std docs")
print(doc_id)
