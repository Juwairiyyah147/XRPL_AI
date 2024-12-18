Project Overview
The project is an advanced AI-driven XRPL (XRP Ledger) wallet and document processing application with the following capabilities:

Core Functionality:

Allows users to interact with the XRPL for wallet management and transaction processing.
Includes robust handling of XRPL-specific transactions like payments, NFT minting, and trust line setting.
Utilizes AI for intelligent document processing and chat-based assistance.

Technologies Used:

Streamlit for a user-friendly UI.
XRPL Python SDK for interaction with the XRP Ledger.
AstraDB VectorStore and embeddings for intelligent document retrieval and storage.
MongoDB for storing wallet information.
LangChain components for natural language processing and document preprocessing.
Key Features:

Wallet Management:

Wallet creation with XRPL faucet integration.
Secure wallet storage in MongoDB.
Display and retrieve wallet details.

Transaction Processing:

Handles XRPL transaction types like TicketCreate, Payment, NFTokenMint, and TrustSet.
Validates transactions before execution.
Supports user interaction for review and confirmation.

Document Processing:

Upload and preprocess XRPL-related documents (PDFs, TXT, DOCX).
Intelligent chunking and cleaning of document data.
Store processed chunks in a vector database for retrieval.

Chat Interface:

AI-driven chat for answering questions related to XRPL.
Supports contextual queries using LangChain’s retriever and vector store.
Creates transaction payloads based on user input.

Logging and Debugging:

Detailed logging with Loguru to track and diagnose issues.
Debugging-friendly setup with comprehensive error handling.

Extended Capabilities:

Provides insights into stored documents using vector database inspection.
Allows resetting and recreating the vector store collection for data management.