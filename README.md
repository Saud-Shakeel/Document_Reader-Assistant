# Document_Reader-Assistant

This is a `Langchain` based project using the LLM `gpt-3.5-turbo-instruct`.
The project performs the following tasks:

1. It asks the user to upload a document.
2. User can aks any question regarding the uploaded document.
3. If the Model has information about the question, it return the response to the user.

******************NOTE: Supoorted file types are [PDFS, DOCX]**************** 

BACKEND: When the file is uploaded, model converts that into chunks of texts, creates embeddings and stores them into a vectore database (I have used Qdrant). When user enters a prompt, the model performs similarity search for the query with the content and retreives the relevant information.
