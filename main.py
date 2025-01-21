import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore, AstraDBChatMessageHistory
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.cache import InMemoryCache
from functools import lru_cache
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
import tempfile
from loguru import logger
from typing import List, Tuple, Any
import sys
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.wallet import Wallet
from xrpl.models import Payment, AccountInfo, NFTokenMint, TrustSet, TicketCreate
from xrpl.utils import xrp_to_drops
import json
from datetime import datetime
import re
from pymongo import MongoClient
from pathlib import Path
from xrpl.models.transactions import Transaction
from xrpl.models.transactions.transaction import TransactionType

# Enhanced logging setup
logger.remove()

# Add detailed file logging handler with rotation and retention
logger.add(
    "logs/app_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)

# Add console logging with colors and detailed formatting
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="DEBUG",  # Set to DEBUG for development
    backtrace=True,
    diagnose=True,
)

logger.info("Starting application...")

# Add before load_dotenv()
if 'ASTRA_DB_NAMESPACE' in os.environ:
    del os.environ['ASTRA_DB_NAMESPACE']
load_dotenv(override=True)
logger.debug("Environment variables loaded")

# Add these debug statements
logger.debug(f"Raw env value: {os.environ.get('ASTRA_DB_NAMESPACE')}")
logger.debug(f"Getenv value: {os.getenv('ASTRA_DB_NAMESPACE')}")

# Configuration with logging
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
COLLECTION_NAME = "xrpl_ai_coll"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-pro-002"
EMBEDDING_MODEL = "models/embedding-001"
XRPL_TESTNET_URL = "https://s.altnet.rippletest.net:51234/"
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

logger.debug(
    f"Configuration loaded: ASTRA_DB_API_ENDPOINT={ASTRA_DB_API_ENDPOINT}, COLLECTION_NAME={COLLECTION_NAME}, MODEL_NAME={MODEL_NAME}"
)

# Set Google API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
logger.debug("Google API key set")

# Initialize embeddings with logging
logger.info("Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
logger.debug(f"Embeddings initialized with model: {EMBEDDING_MODEL}")

# Initialize language model with enhanced logging
logger.info("Initializing language model...")
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.2,
    max_output_tokens=8192,
    top_p=0.95,
    top_k=40,
    cache=InMemoryCache(),
)
logger.debug(
    f"Language model initialized with parameters: temp=0.2, max_tokens=8192, top_p=0.95, top_k=40"
)

# Global variables for vector store and XRPL with logging
vector_store = None
xrpl_client = None
wallet = None
logger.debug("Global variables initialized")

# At the beginning of the file, add error handling for environment variables
def check_environment_variables():
    required_vars = [
        "GOOGLE_API_KEY",
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_NAMESPACE"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)

# Add this near the start of your code, after load_dotenv()
try:
    check_environment_variables()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

# Add this after loading environment variables
logger.debug(f"Using AstraDB namespace: {ASTRA_DB_NAMESPACE}")

@lru_cache(maxsize=1)
def initialize_vector_store():
    logger.info("Initializing vector store...")
    global vector_store
    if vector_store is None:
        try:
            logger.debug(f"Initializing vector store with namespace: {ASTRA_DB_NAMESPACE}")
            vector_store = AstraDBVectorStore(
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                token=ASTRA_DB_APPLICATION_TOKEN,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                namespace="default_keyspace",
                metric="cosine",
            )
            
            # Test with proper embedding
            logger.info("Testing AstraDB connection...")
            test_content = "test document"
            test_embedding = embeddings.embed_query(test_content)
            test_doc = Document(
                page_content=test_content,
                metadata={"source": "test"}
            )
            vector_store.add_documents([test_doc])
            logger.info("AstraDB connection test successful")
            
            logger.success("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            st.error("Failed to initialize vector store. Please check your AstraDB credentials.")
            st.stop()
    return vector_store


@lru_cache(maxsize=1)
def initialize_xrpl_client(use_testnet=True):
    """Initialize XRPL client with enhanced logging"""
    logger.info(f"Initializing XRPL client (Testnet: {use_testnet})")
    try:
        url = XRPL_TESTNET_URL
        client = JsonRpcClient(url)
        # Test connection with a simple server_info request
        server_info = client.request(xrpl.models.ServerInfo())
        logger.debug(f"Testing XRPL connection to {url}")
        logger.success(
            f"Connected to XRPL {'Testnet' if use_testnet else 'Mainnet'}: {url}"
        )
        return client
    except Exception as e:
        logger.error(f"Error connecting to XRPL: {str(e)}", exc_info=True)
        raise


def get_session_history(session_id: str):
    logger.debug(f"Getting chat history for session: {session_id}")
    try:
        history = AstraDBChatMessageHistory(
            session_id=session_id,
            collection_name="chat_history",
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            namespace=ASTRA_DB_NAMESPACE,
        )
        logger.debug(f"Chat history retrieved successfully for session: {session_id}")
        return history
    except Exception as e:
        logger.error(f"Error getting chat history: {e}", exc_info=True)
        raise


def create_wallet(client):
    """Create a new XRPL wallet with enhanced logging"""
    logger.info("Creating new XRPL wallet...")
    try:
        # Uses XRPL's faucet to generate a funded testnet wallet
        wallet = xrpl.wallet.generate_faucet_wallet(client)
        logger.debug(
            f"Wallet details: Address={wallet.classic_address}, Public Key={wallet.public_key}"
        )
        logger.success(f"Created new wallet with address: {wallet.classic_address}")
        return wallet
    except Exception as e:
        logger.error(f"Error creating wallet: {str(e)}", exc_info=True)
        raise


def get_account_info(client, wallet):
    """Get account information with detailed logging"""
    logger.info(f"Getting account info for address: {wallet.classic_address}")
    try:
        acct_info = AccountInfo(
            account=wallet.classic_address,
            ledger_index="validated",
            strict=True,
        )
        logger.debug(f"Account info request created: {acct_info.to_dict()}")
        response = client.request(acct_info)
        logger.debug(f"Raw account info response: {response.result}")
        logger.success(f"Retrieved account info for {wallet.classic_address}")
        return response.result
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}", exc_info=True)
        raise


async def prepare_transaction(tx_json, client, wallet):
    """Prepare an XRPL transaction with enhanced logging"""
    logger.info(f"Preparing {tx_json.get('transaction_type', 'UNKNOWN')} transaction...")
    logger.debug(f"Input transaction JSON: {tx_json}")
    logger.debug(f"JSON keys: {list(tx_json.keys())}")

    try:
        # Create transaction from JSON using generic Transaction model
        transaction = xrpl.models.transactions.Transaction.from_dict(tx_json)
        logger.debug(f"Created transaction object: {transaction.to_dict()}")
        logger.debug(f"Transaction object type: {type(transaction)}")

        # Run synchronous function in thread executor
        loop = asyncio.get_event_loop()
        prepared = await loop.run_in_executor(None, xrpl.transaction.autofill, transaction, client)
        logger.debug(f"Transaction prepared with fees: {prepared.to_dict()}")
        logger.success(f"{tx_json.get('transaction_type', 'UNKNOWN')} transaction prepared successfully")
        return prepared
    except Exception as e:
        logger.error(f"Error preparing transaction: {str(e)}", exc_info=True)
        logger.error(f"Transaction JSON that caused error: {tx_json}")
        raise


async def submit_transaction(prepared_tx, client, wallet):
    """Submit a signed transaction to XRPL with enhanced logging"""
    try:
        # Run synchronous function in thread executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, xrpl.transaction.submit_and_wait, prepared_tx, client, wallet)
        logger.info(f"Transaction submitted successfully: {response.result}")
        return response.result
    except Exception as e:
        logger.error(f"Error submitting transaction: {e}")
        raise


# Update system prompt to be more comprehensive
#   system_prompt = """You are an advanced AI assistant specializing in the XRP Ledger (XRPL). You can:

#1. Help users interact with XRPL through natural language
#2. Analyze and explain XRPL concepts, features and transactions
#3. Assist with wallet management and transaction preparation
#4. Guide users through complex operations step by step


#prompt = ChatPromptTemplate.from_messages(
#    [
#        ("system", system_prompt),
#        MessagesPlaceholder(variable_name="chat_history"),
#        ("human", "{input}"),
#    ]
#)

# Create chain and conversation handler
#chain = prompt | llm | StrOutputParser()

#conversation = RunnableWithMessageHistory(
#    chain,
#    get_session_history=get_session_history,
#    input_messages_key="input",
#    history_messages_key="chat_history",
#    output_messages_key="output",
#)

def clean_and_preprocess_document(content: str) -> str:
    logger.debug("Starting document cleaning and preprocessing")
    
    # Remove common noise patterns
    logger.debug("Removing noise patterns")
    patterns_to_remove = [
        r'Text from https:\/\/www\.youtube\.com\/embed\S+.*?Try watching this video on www\.youtube\.com.*?\n',
        r'Text from https:\/\/livenet\.xrpl\.org.*?JavaScript to run this app\.',
        r'\[(\d+)\]',  # Remove reference numbers
        r'^\s*Table of Contents\s*$.*?(?=\n\n)',  # Remove table of contents
        r'Click here to scroll to this section',
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Extract sections with headers
    logger.debug("Extracting sections with headers")
    sections = []
    current_section = []
    
    for line in content.split('\n'):
        # Identify headers (lines that are shorter and end with common header patterns)
        if re.match(r'^[A-Z][^.!?]{0,50}(?:[:.)]|\s*)$', line.strip()):
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(line.strip())
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Join sections with clear delimiters
    cleaned_content = '\n====================\n'.join(
        section.strip() for section in sections if section.strip()
    )
    
    logger.debug(f"Document preprocessing complete. Original length: {len(content)}, New length: {len(cleaned_content)}")
    return cleaned_content

async def process_chunks(chunks: List[Document], vector_store: AstraDBVectorStore) -> None:
    logger.info(f"Processing {len(chunks)} chunks")
    batch_size = 25
    max_retries = 3
    
    async def process_batch(batch, batch_num):
        processed_batch = []
        for i, chunk in enumerate(batch):
            try:
                # Classify the chunk
                classification = await classify_chunk_theme(chunk.page_content)
                
                # Create enhanced document with classification metadata
                enhanced_doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "theme": classification["theme"],
                        "topics": classification["topics"],
                        "complexity": classification["complexity"],
                        "summary": classification["summary"],
                        "batch_num": batch_num,
                        "chunk_num": i,
                    }
                )
                processed_batch.append(enhanced_doc)
                logger.debug(f"Processed chunk {i} in batch {batch_num}: Theme={classification['theme']}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i} in batch {batch_num}: {e}")
                # Keep original chunk if processing fails
                processed_batch.append(chunk)
        
        # Add batch to vector store with retries
        for attempt in range(max_retries):
            try:
                await asyncio.to_thread(
                    vector_store.add_documents,
                    processed_batch,
                    ids=[f"doc_{batch_num}_{i}" for i in range(len(processed_batch))]
                )
                logger.info(f"Successfully added batch {batch_num} to vector store")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to add batch {batch_num} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed for batch {batch_num}: {e}. Retrying...")
                await asyncio.sleep(1)
    
    # Process batches sequentially
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            await process_batch(batch, batch_num)
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}: {e}")
            raise

    logger.info("All chunks processed and stored successfully")


async def process_single_document(file, vector_store: AstraDBVectorStore) -> Tuple[bool, str]:
    tmp_file = None
    tmp_file_path = None
    try:
        # Create temp file with proper name
        file_extension = os.path.splitext(file.name)[1].lower()
        tmp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            delete=False,
            suffix=file_extension
        )
        tmp_file_path = tmp_file.name
        
        # Write content
        tmp_file.write(file.getvalue())
        tmp_file.flush()
        tmp_file.close()

        logger.debug(f"Processing file: {file.name} with extension: {file_extension}")

        # Load document based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            documents = await asyncio.to_thread(loader.load)
        elif file_extension == ".txt":
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = await asyncio.to_thread(loader.load)
            # Clean and preprocess the document content
            logger.debug("Cleaning and preprocessing document content")
            cleaned_content = clean_and_preprocess_document(documents[0].page_content)
            
            # Create new document with cleaned content
            documents = [Document(
                page_content=cleaned_content,
                metadata={"source": file.name}
            )]
        elif file_extension in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
            documents = await asyncio.to_thread(loader.load)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        logger.debug("Loading document content")
        
        # Initial chunking
        logger.debug("Initial document splitting")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        initial_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
        
        # Preprocess each chunk with LLM
        logger.info(f"Preprocessing {len(initial_chunks)} chunks with LLM")
        processed_chunks = []
        for chunk in initial_chunks:
            try:
                # Process with LLM
                enhanced_content = await preprocess_with_llm(chunk.page_content)
                
                # Create new document with enhanced content
                processed_chunk = Document(
                    page_content=enhanced_content,
                    metadata={
                        **chunk.metadata,
                        "preprocessed": True,
                        "original_content": chunk.page_content[:200]  # Keep preview of original
                    }
                )
                processed_chunks.append(processed_chunk)
                logger.debug(f"Successfully preprocessed chunk: {enhanced_content[:100]}...")
                
            except Exception as e:
                logger.error(f"Error preprocessing chunk: {e}")
                processed_chunks.append(chunk)  # Keep original if processing fails
        
        logger.info(f"Successfully preprocessed {len(processed_chunks)} chunks")
        await process_chunks(processed_chunks, vector_store)
        return True, file.name

    except Exception as e:
        logger.error(f"Error processing document {file.name}: {e}")
        return False, file.name
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")


async def process_multiple_documents(files) -> List[Tuple[str, bool]]:
    logger.info(f"Processing {len(files)} documents")
    vector_store = initialize_vector_store()
    logger.debug("Vector store initialized for document processing")

    logger.debug(f"Creating processing tasks for {len(files)} files")
    results = await asyncio.gather(
        *[process_single_document(file, vector_store) for file in files],
        return_exceptions=True,
    )
    logger.debug("Document processing tasks completed")

    processed_results = [
        (files[i].name, not isinstance(r, Exception) and r[0])
        for i, r in enumerate(results)
    ]
    logger.info(f"Processed {len(processed_results)} documents")
    return processed_results

async def get_transaction_parameters_info(transaction_type: TransactionType) -> list:
    """Get parameter information for a specific transaction type from AstraDB"""
    vector_store = initialize_vector_store()
    
    # Query to find relevant chunks about the transaction parameters
    query = f"What are the required and optional parameters for {transaction_type} transaction?"
    logger.debug(f"Querying vector store for {transaction_type} parameters")
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
        logger.debug(f"Found {len(relevant_docs)} relevant documents")
        return relevant_docs
    except Exception as e:
        logger.error(f"Error retrieving parameter info: {e}")
        raise

async def validate_transaction_parameters(tx_json: dict, context_docs: list) -> tuple[bool, str]:
    """Validate transaction parameters using LLM and context"""
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a transaction validator. Using the provided context about transaction parameters,
        check if the transaction JSON has all required fields with valid values.
        
        If any required parameters are missing or invalid, explain what's missing and how to provide it.
        If all required parameters are present and valid, confirm it's ready for submission.
        
        Context about parameters:
        {context}
        
        Transaction JSON:
        {transaction}
        """),
        ("human", "Is this transaction complete and valid?")
    ])
    
    # Combine context documents
    context = "\n\n".join(doc.page_content for doc in context_docs)
    
    try:
        formatted_prompt = await validation_prompt.ainvoke({
            "context": context,
            "transaction": json.dumps(tx_json, indent=2)
        })
        response = await llm.ainvoke(formatted_prompt)
        
        # Check if response indicates missing parameters
        is_valid = "ready for submission" in response.content.lower()
        return is_valid, response.content
        
    except Exception as e:
        logger.error(f"Error validating parameters: {e}")
        raise



async def transaction_request(question):
    try:
        if "wallet" not in st.session_state:
            return "Please create a wallet first before requesting transaction payloads."

        wallet = st.session_state.wallet

        # Check if we're in a follow-up conversation about parameters
        if "pending_transaction" in st.session_state:
            # Extract parameter values from the user's response
            param_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                Extract parameter values from the user's response.
                Return them as a JSON object matching the parameter names exactly.
                """),
                ("human", "{text}")
            ])
            
            formatted_prompt = await param_prompt.ainvoke({"text": question})
            response = await llm.ainvoke(formatted_prompt)
            new_params = json.loads(response.content)
            
            # Update the pending transaction with new parameters
            tx_json = {
                **st.session_state.pending_transaction,
                **new_params
            }
            
            # Validate updated transaction
            parameter_docs = await get_transaction_parameters_info(TransactionType[tx_json["TransactionType"]])
            is_valid, validation_message = await validate_transaction_parameters(tx_json, parameter_docs)
            
            if not is_valid:
                # Still missing parameters
                st.session_state.pending_transaction = tx_json
                return validation_message
            
            # All parameters are valid
            del st.session_state.pending_transaction
            return f"""Here's the complete transaction payload:
```json
{json.dumps(tx_json, indent=2)}
```
Would you like me to prepare and submit this transaction?"""

        # New transaction request
        tx_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Extract transaction details from the user request. Return a valid JSON object with:
            {{
                "transaction_type": "<type>",
                "Parameters": {{ ... transaction specific parameters ... }}
            }}
            Supported types: TicketCreate, Payment, NFTokenMint, TrustSet  # Proper case for transaction types
            Note: transaction_type should be in proper case (e.g., TicketCreate, not ticketcreate or Ticketcreate. its your job to change it to correct case)
            Note: All the parameters should be in lower case (e.g., ticket_count, not TicketCount) Its your job to change it to correct format.
            Dont add any parameters that are not mentioned in the user request.
            """),
            ("human", "{text}")
        ])
        
        formatted_prompt = await tx_prompt.ainvoke({"text": question})
        response = await llm.ainvoke(formatted_prompt)
        
        # Clean and parse the JSON
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        initial_json = json.loads(content.strip()) #converting to python dictionary

        tx_json = {
            "transaction_type": initial_json["transaction_type"],
            "account": wallet.classic_address,
            **initial_json["Parameters"]  
        }
        logger.debug(f"Initial JSON: {initial_json}")
        
      
        transaction_type = tx_json.get('transaction_type', 'UNKNOWN') 
        logger.debug(f"Identified transaction type: {transaction_type}")
        
        # Get parameter information from AstraDB
        parameter_docs = await get_transaction_parameters_info(transaction_type)
        
        # Create initial transaction JSON with any provided parameters
        #tx_json = {
        #    "TransactionType": transaction_type.name,
        #    "Account": wallet.classic_address,
        #    **{k: v for k, v in initial_json.items() 
        #       if k not in ["TransactionType", "Account"]}
        #}
        
        # Validate parameters
        is_valid, validation_message = await validate_transaction_parameters(tx_json, parameter_docs)
        
        if not is_valid:
            # Store the pending transaction
            st.session_state.pending_transaction = tx_json
            return f"""This transaction needs more information:
            
{validation_message}

Please provide the missing information and I'll help you prepare the transaction."""
        
        return f"""Here's the transaction payload:
```json
{json.dumps(tx_json, indent=2)}
```
Would you like me to prepare and submit this transaction?"""
                
    except Exception as e:
        logger.error(f"Error creating transaction payload: {e}")
        return f"Error creating transaction payload: {str(e)}"


async def chat_with_knowledge_base(question, source=None, chat_history=None):
    logger.info(f"Processing chat request - Question: {question}, Source: {source}")

    if chat_history is None:
        chat_history = []
    logger.debug(f"Chat history length: {len(chat_history)}")

    vector_store = initialize_vector_store()
    logger.debug("Vector store initialized for chat")

    search_kwargs = {"k": 10}
    if source:
        search_kwargs["filter"] = {"metadata": {"source": source}}
        logger.debug(f"Added source filter: {source}")

    logger.debug("Creating retriever with search kwargs")
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    logger.info("Retriever created")

    try:
        logger.info("Retrieving relevant documents")
        relevant_docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
        logger.debug(f"Retrieved {len(relevant_docs)} relevant documents")
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {e}")
        return "Error retrieving relevant documents. Please try again."

    # Log retrieved text content
    for i, doc in enumerate(relevant_docs):
        logger.info(f"Retrieved document {i+1}: {doc.page_content[:200]}...")

    if not relevant_docs:
        logger.warning("No relevant documents found")
        if source:
            return f"I couldn't find any relevant information in the specified source: {source}. Would you like to search all sources instead?"
        else:
            return "I couldn't find any relevant information to answer your question. Could you please rephrase or ask a different question?"

    logger.debug("Setting up contextualization prompt")
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    #! Created a separate function for transaction requests
    #if "transaction payload" in question.lower():
    #    return transaction_request(question)
    
    # For non-transaction requests, use RAG
    logger.debug("Creating history aware retriever")
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    logger.debug("Setting up QA system prompt")
    qa_system_prompt = """You are an XRPL expert assistant. Use the retrieved context to:
    1. Answer questions about XRPL concepts and features
    2. Help prepare and explain transactions
    3. Provide guidance on wallet management
    4. Explain technical details clearly
    
    If you don't know the answer, just say so. Keep answers concise.
    
    
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    logger.debug("Creating question answer chain")
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    logger.info("Invoking RAG chain")
    response = await asyncio.to_thread(
        rag_chain.invoke, {"input": question, "chat_history": chat_history}
    )

    answer = response["answer"]
    logger.info(f"Generated answer: {answer}")

    return answer



# and before the Streamlit UI section
async def inspect_vector_store():
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})  # Get more documents
    
    try:
        # Use a generic query to get as many documents as possible
        docs = await asyncio.to_thread(retriever.get_relevant_documents, "")
        logger.info(f"\nTotal documents in store: {len(docs)}")
        
        for i, doc in enumerate(docs):
            logger.info(f"\nDocument {i+1}:")
            logger.info(f"Content: {doc.page_content[:500]}...")  # First 500 chars
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("="*50)
            
    except Exception as e:
        logger.error(f"Error inspecting vector store: {e}")


async def clear_and_recreate_collection():
    try:
        # Create new vector store instance
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            namespace="default_keyspace",
            metric="cosine",
        )
        
        # Delete existing collection
        logger.info("Deleting existing collection...")
        await asyncio.to_thread(vector_store.delete_collection)
        
        # Initialize a new vector store (this will create a new collection)
        logger.info("Creating new collection...")
        new_vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            namespace="default_keyspace",
            metric="cosine",
        )
        
        # Test the new collection with a simple document
        test_doc = Document(
            page_content="Test document",
            metadata={"source": "test"}
        )
        await asyncio.to_thread(new_vector_store.add_documents, [test_doc])
        logger.info("New collection created and tested successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        return False
async def preprocess_with_llm(text: str) -> str:
    # Create a proper prompt template
    preprocess_prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze this text and ensure it's a complete, self-contained piece of information. 
        If incomplete, expand it with relevant context. Keep it focused and coherent.
        The output should be a self-contained piece of documentation."""),
        ("human", "{text}")
    ])
    
    # Format the prompt with the text
    formatted_prompt = await preprocess_prompt.ainvoke({"text": text})
    
    # Send to LLM and get response
    response = await llm.ainvoke(formatted_prompt)
    
    # Extract the content from response
    if hasattr(response, 'content'):
        return response.content
    return str(response)

async def classify_chunk_theme(text: str) -> dict:
    """Classify the theme of a text chunk using LLM."""
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document classifier for XRPL documentation.
        Analyze the text and:
        1. Identify the main theme (e.g., Tutorials, API Methods, Use Cases, Concepts, etc.)
        2. Extract key topics covered
        3. Determine the technical complexity level (Beginner, Intermediate, Advanced)
        
        Return the analysis as JSON in this format:
        {{
            "theme": "main_theme",
            "topics": ["topic1", "topic2"],
            "complexity": "level",
            "summary": "brief_summary"
        }}"""),  # Note the double curly braces for escaping
        ("human", "{text}")
    ])
    
    try:
        # Format prompt with chunk text
        formatted_prompt = await classification_prompt.ainvoke({"text": text})
        
        # Get classification from LLM
        response = await llm.ainvoke(formatted_prompt)
        
        # Parse JSON response
        if hasattr(response, 'content'):
            classification = json.loads(response.content)
        else:
            classification = json.loads(str(response))
            
        logger.debug(f"Chunk classified as: {classification['theme']}")
        return classification
        
    except Exception as e:
        logger.error(f"Error classifying chunk: {e}")
        return {
            "theme": "Unknown",
            "topics": [],
            "complexity": "Unknown",
            "summary": "Classification failed"
        }



def initialize_mongo():
    """Initialize MongoDB connection"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        logger.debug("MongoDB connection initialized")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise

def save_wallet(wallet_data):
    """Save wallet to MongoDB"""
    try:
        db = initialize_mongo()
        wallet_doc = {
            "address": wallet_data.classic_address,
            "public_key": wallet_data.public_key,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
        
        # Check if wallet already exists
        existing_wallet = db[MONGO_COLLECTION].find_one({"address": wallet_data.classic_address})
        if existing_wallet:
            logger.warning(f"Wallet {wallet_data.classic_address} already exists in database")
            return False
            
        db[MONGO_COLLECTION].insert_one(wallet_doc)
        logger.info(f"Wallet saved to database: {wallet_data.classic_address}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving wallet to MongoDB: {e}")
        raise

def get_wallet(address):
    """Retrieve wallet from MongoDB"""
    try:
        db = initialize_mongo()
        wallet = db[MONGO_COLLECTION].find_one({"address": address})
        if wallet:
            logger.debug(f"Retrieved wallet: {address}")
            return wallet
        logger.warning(f"Wallet not found: {address}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving wallet from MongoDB: {e}")
        raise


async def process_transaction():
    logger.debug("Starting process_transaction")
    logger.debug(f"Transaction data from session state: {st.session_state.review_state['transaction']}")
    
    try:
        logger.debug("Preparing transaction")
        prepared_tx = await prepare_transaction(
            st.session_state.review_state["transaction"],
            st.session_state.xrpl_client,
            st.session_state.wallet
        )
        logger.debug(f"Prepared transaction: {prepared_tx.to_dict() if prepared_tx else 'None'}")
        
        logger.debug("Submitting transaction")
        result = await submit_transaction(
            prepared_tx,
            st.session_state.xrpl_client,
            st.session_state.wallet
        )
        logger.debug(f"Submit result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in process_transaction: {str(e)}", exc_info=True)
        raise

# Chat interface section starts here
logger.debug("Initializing chat interface")
if "messages" not in st.session_state:
    logger.debug("Initializing empty message history")
    st.session_state.messages = []

# The Streamlit UI section starts
st.title("XRPL AI Wallet")
logger.info("Starting XRPL AI Wallet UI")

# Initialize XRPL client
if "xrpl_client" not in st.session_state:
    logger.info("Initializing XRPL client")
    try:
        st.session_state.xrpl_client = initialize_xrpl_client()
        logger.debug("XRPL client initialized and stored in session state")
    except Exception as e:
        logger.error(f"Failed to initialize XRPL client: {e}")
        st.error("Failed to connect to XRPL network. Please try again later.")

# Wallet management
with st.sidebar:
    st.header("Wallet Management")
    logger.debug("Rendering wallet management sidebar")

    if st.button("Create New Wallet"):
        if "wallet" in st.session_state:
            logger.info("Wallet already exists")
            st.warning("You already have a wallet")
        else:
            logger.info("Creating new wallet")
            try:
                wallet = create_wallet(st.session_state.xrpl_client)
                st.session_state.wallet = wallet
                
                # Save to MongoDB
                if save_wallet(wallet):
                    logger.success(f"New wallet created and saved: {wallet.classic_address}")
                    st.success(f"Wallet created: {wallet.classic_address}")
                else:
                    logger.warning("Wallet already exists in database")
                    st.warning("This wallet already exists in the database")
                    
            except Exception as e:
                logger.error(f"Failed to create wallet: {e}")
                st.error("Failed to create wallet. Please try again.")

# Add near the top with other session state initializations
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Add this near the top where other session state variables are initialized
if "show_review" not in st.session_state:
    st.session_state.show_review = False
    logger.debug("Initialized show_review state to False")

if "current_transaction" not in st.session_state:
    st.session_state.current_transaction = None
    logger.debug("Initialized current_transaction state to None")

# First, add this state initialization at the top with other state variables
if "review_clicked" not in st.session_state:
    st.session_state.review_clicked = False

# Move this to the top where other session states are initialized (near "messages" initialization)
if "review_state" not in st.session_state:
    st.session_state.review_state = {
        "showing": False,
        "transaction": None
    }

# Add this to the session state initialization at the top
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Document processing
uploaded_files = st.file_uploader(
    "Upload XRPL documentation", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    # Filter out already processed files
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        logger.info(f"Processing {len(new_files)} new files")
        with st.spinner(f"Processing {len(new_files)} files..."):
            results = asyncio.run(process_multiple_documents(new_files))
            for filename, success in results:
                if success:
                    logger.success(f"Successfully processed file: {filename}")
                    st.success(f"{filename} processed successfully!")
                    # Only adds filename to session state to track what's been processed
                    st.session_state.processed_files.add(filename)
                else:
                    logger.error(f"Failed to process file: {filename}")
                    st.error(f"Error processing {filename}")
    else:
        logger.info("No new files to process")


col1, col2, col3 = st.columns([1, 1, 2])  # Create three columns for better layout
with col1:
    if st.button("Inspect Vector Store"):
        with st.spinner("Inspecting vector store..."):
            asyncio.run(inspect_vector_store())
            st.success("Vector store inspection complete. Check the logs for details.")
with col2:
    if st.button("Reset Collection"):
        with st.spinner("Resetting collection..."):
            if asyncio.run(clear_and_recreate_collection()):
                st.success("Collection reset successfully!")
                st.session_state.processed_files = set()  # Clear processed files tracking
            else:
                st.error("Failed to reset collection")

# Chat interface
logger.debug("Initializing chat interface")
if "messages" not in st.session_state:
    logger.debug("Initializing empty message history")
    st.session_state.messages = []

for message in st.session_state.messages:
    logger.debug(f"Displaying message - Role: {message['role']}")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about XRPL or request a transaction:"):
    logger.info(f"New chat input received: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        logger.info("Processing chat response")
        # Create and set new event loop for chat response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Check if it's a transaction request
        if "transaction payload" in prompt.lower():
            response = loop.run_until_complete(transaction_request(prompt))
        else:
            response = loop.run_until_complete(chat_with_knowledge_base(prompt))
        
        loop.close()
        
        logger.debug(f"Chat response generated: {response}")
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
        # Clear previous transaction state when new chat response comes
        st.session_state.review_state["showing"] = False
        st.session_state.review_state["transaction"] = None
        st.session_state.last_response = response

# Modify the review section to use st.empty()
if st.session_state.last_response and "```json" in st.session_state.last_response and "Would you like me to prepare and submit this transaction?" in st.session_state.last_response:
    review_container = st.container()
    review_window = st.empty()
    
    with review_container:
        # Extract transaction data only once
        if not st.session_state.review_state["transaction"]:
            try:
                # Extract the raw JSON from the response
                json_str = st.session_state.last_response[st.session_state.last_response.find("```json") + 7 : st.session_state.last_response.rfind("```")].strip()
                logger.debug(f"Extracted JSON: {json_str}")
                
                # Just parse the JSON directly since it's already in lowercase
                tx_json = json.loads(json_str)
                
                # Store the complete transaction JSON
                st.session_state.review_state["transaction"] = tx_json
                logger.debug(f"Stored transaction JSON: {tx_json}")
            except Exception as e:
                logger.error(f"Error parsing JSON: {e}")
                st.error("Error parsing transaction data")
                st.stop()
        
        # Always show the review button
        if st.button("📝 Review Transaction", key="review_btn"):
            st.session_state.review_state["showing"] = True
        
        # Show/hide only the review window
        if st.session_state.review_state["showing"]:
            with review_window.container():
                st.markdown("### 🔍 Transaction Review")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Transaction Summary")
                    for key, value in st.session_state.review_state["transaction"].items():
                        st.markdown(f"**{key}:** {value}")
                
                with col2:
                    st.markdown("#### Raw Transaction")
                    st.code(json.dumps(st.session_state.review_state["transaction"], indent=2))
                
                st.warning("⚠️ Please review the transaction details carefully")
                
                col1, col2, _ = st.columns([1, 2, 3])
                with col1:
                    if st.button("❌ Close", key="close_btn"):
                        st.session_state.review_state["showing"] = False
        
                with col2:
                    if st.button("🚀 Submit", key="submit_btn", type="primary"):
                        with st.spinner("Processing transaction..."):
                            try:
                                # Create and set a new event loop
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                result = loop.run_until_complete(process_transaction())
                                loop.close()
                                
                                st.success("✅ Transaction submitted successfully!")
                                #st.json(result)
                                st.session_state.review_state["showing"] = False
                                st.session_state.review_state["transaction"] = None
                            except Exception as e:
                                logger.error(f"Transaction failed: {str(e)}")
                                st.error(f"❌ Transaction failed: {str(e)}")

