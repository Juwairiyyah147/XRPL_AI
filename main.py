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


async def prepare_transaction(transaction_type, params, client, wallet):
    """Prepare an XRPL transaction with enhanced logging"""
    logger.info(f"Preparing {transaction_type} transaction...")
    logger.debug(f"Transaction parameters: {params}")

    try:
        if transaction_type == "TicketCreate":
            logger.debug("Creating TicketCreate transaction")
            transaction = TicketCreate(
                account=wallet.classic_address,  # Note: lowercase 'account'
                ticket_count=int(params.get("TicketCount", 1))  # Convert to int
            )
            logger.debug(f"Created TicketCreate transaction: {transaction.to_dict()}")

        elif transaction_type == "Payment":
            transaction = Payment(
                account=wallet.classic_address,
                destination=params["destination"],
                amount=xrp_to_drops(params["amount"]),
            )
        elif transaction_type == "NFTokenMint":
            transaction = NFTokenMint(
                account=wallet.classic_address,
                uri=params["uri"],
                flags=params.get("flags", 0),
                transfer_fee=params.get("transfer_fee", 0),
                nftoken_taxon=params.get("nftoken_taxon", 0),
            )
        elif transaction_type == "TrustSet":
            transaction = TrustSet(
                account=wallet.classic_address, 
                limit_amount=params["limit_amount"]
            )
        else:
            logger.error(f"Unsupported transaction type: {transaction_type}")
            raise ValueError(f"Unsupported transaction type: {transaction_type}")

        logger.debug(f"Transaction object created: {transaction.to_dict()}")
        prepared = xrpl.transaction.autofill(transaction, client)
        logger.debug(f"Transaction prepared with fees: {prepared.to_dict()}")
        logger.success(f"{transaction_type} transaction prepared successfully")
        return prepared
    except Exception as e:
        logger.error(
            f"Error preparing {transaction_type} transaction: {str(e)}", exc_info=True
        )
        raise


async def submit_transaction(prepared_tx, client, wallet):
    """Submit a signed transaction to XRPL with enhanced logging"""
    try:
        response = xrpl.transaction.submit_and_wait(prepared_tx, client, wallet)
        logger.info(f"Transaction submitted successfully: {response.result}")
        return response.result
    except Exception as e:
        logger.error(f"Error submitting transaction: {e}")
        raise


# Update system prompt to be more comprehensive
system_prompt = """You are an advanced AI assistant specializing in the XRP Ledger (XRPL). You can:

1. Help users interact with XRPL through natural language
2. Analyze and explain XRPL concepts, features and transactions
3. Assist with wallet management and transaction preparation
4. Guide users through complex operations step by step

For transactions, you will:
1. Extract relevant parameters from user requests
2. Format them into proper transaction objects
3. Explain the implications and costs
4. Ask for explicit confirmation before executing
5. Provide clear feedback on results

When preparing transactions, format them as JSON like:

{
    "TransactionType": "<type>",
    "Account": "<source>",
    // ... other fields specific to transaction type
}

Always verify critical details like:
- Transaction type is valid
- Amounts are properly formatted
- Addresses are valid
- Required fields are present
- Fee is appropriate

Ask for clarification if any required information is missing."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create chain and conversation handler
chain = prompt | llm | StrOutputParser()

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
)

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

    # Check if this is a transaction creation request
    if "transaction payload" in question.lower():
        try:
            if "wallet" not in st.session_state:
                return "Please create a wallet first before requesting transaction payloads."

            wallet = st.session_state.wallet

            # Extract transaction details from question using LLM
            tx_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract transaction details from the user request. Your response must be a valid JSON object, with no additional text, comments, or formatting.
    Ensure the response does not contain markdown or code block markers like ```json.
                Return a JSON object with:
                {{
                    "TransactionType": "<type>",
                    "Parameters": {{ "key": "value", ... }}
                }}
                Ensure the JSON is valid and complete.

                Supported types: Payment, NFTokenMint, TrustSet, TicketCreate"""),
                ("human", "{text}")
            ])

            # Get transaction details from LLM
            formatted_prompt = await tx_prompt.ainvoke({"text": question})
            response = await llm.ainvoke(formatted_prompt)

            # Log raw response for debugging
            logger.debug(f"Raw AI response: {response}")

            if not response or not hasattr(response, 'content'):
                logger.error("Empty or invalid response from AI model")
                return "Error: AI model did not return a valid response. Please refine your request."
            
            raw_content = response.content.strip()
            # Remove code block markers if present
            if raw_content.startswith("```json") and raw_content.endswith("```"):
                raw_content = raw_content[7:-3].strip()
            
            # Attempt to parse the response as JSON
            try:
                tx_details = json.loads(raw_content)
                logger.debug(f"Parsed transaction details: {tx_details}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing transaction JSON: {raw_content} - {e}")
                return "Error: Failed to parse transaction payload. Please try again with more specific details."

            # Create transaction parameters
            params = {
                "TransactionType": tx_details["TransactionType"],
                "Account": wallet.classic_address,
                **tx_details["Parameters"]
            }

            # Validate parameters using prepare_transaction
            try:
                await asyncio.to_thread(prepare_transaction,
                    params["TransactionType"],
                    params,
                    st.session_state.xrpl_client,
                    wallet
                )
                logger.debug(f"Transaction parameters validated: {params}")
            except Exception as e:
                logger.error(f"Invalid transaction parameters: {e}")
                return f"Error in transaction parameters: {str(e)}"

            return f"""Here's the transaction payload for {params['TransactionType']}:
```json
{json.dumps(params, indent=2)}

```
Would you like me to prepare and submit this transaction?"""
                
        except Exception as e:
            logger.error(f"Error creating transaction payload: {e}")
            return f"Error creating transaction payload: {str(e)}"

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


# Add helper function to prepare and validate transactions
async def prepare_transaction_from_json(
    tx_json: dict, client, wallet
) -> Tuple[bool, str, Any]:
    """
    Prepare and validate a transaction from JSON format.
    Returns (success, message, prepared_tx)
    """
    logger.info("Preparing transaction from JSON")
    logger.debug(f"Transaction JSON: {tx_json}")

    try:
        # Validate transaction type
        if "TransactionType" not in tx_json:
            logger.error("Transaction type missing from JSON")
            return False, "Transaction type is required", None

        # Convert to proper transaction model
        logger.debug("Converting to transaction model")
        tx = xrpl.models.Transaction.from_xrpl(tx_json)
        logger.debug(f"Created transaction model: {tx}")

        # Autofill fields like fee, sequence, etc
        logger.info("Autofilling transaction fields")
        prepared_tx = await asyncio.to_thread(xrpl.transaction.autofill, tx, client)
        logger.debug(f"Prepared transaction: {prepared_tx}")

        # Basic validation
        if not prepared_tx.is_valid():
            logger.error("Transaction validation failed")
            return False, "Invalid transaction after preparation", None

        logger.success("Transaction prepared successfully")
        return True, "Transaction prepared successfully", prepared_tx

    except Exception as e:
        logger.error(f"Error preparing transaction: {e}", exc_info=True)
        return False, f"Error: {str(e)}", None


# Update conversation handler to use enhanced transaction processing
async def handle_conversation(question: str, session_id: str, client=None, wallet=None):
    """Enhanced conversation handler with transaction support"""
    logger.info(f"Handling conversation for session {session_id}")
    logger.debug(f"Question: {question}")

    try:
        # Get relevant context from vector store
        logger.info("Retrieving context from knowledge base")
        context = await chat_with_knowledge_base(question)
        logger.debug(f"Retrieved context: {context}")

        # Add transaction context if available
        if client and wallet:
            logger.debug(f"Adding wallet context for address: {wallet.classic_address}")
            context += f"\nActive wallet: {wallet.classic_address}"

        # Process response through RAG chain
        logger.info("Invoking conversation chain")
        response = await conversation.ainvoke(
            {"input": question, "session_id": session_id, "context": context}
        )
        logger.debug(f"Raw response: {response}")

        # Check for transaction intent
        if "TransactionType" in response:
            logger.info("Transaction intent detected")
            # Extract transaction JSON
            try:
                logger.debug("Parsing transaction JSON")
                tx_json = json.loads(response)
                logger.debug(f"Parsed transaction JSON: {tx_json}")

                success, msg, prepared_tx = await prepare_transaction_from_json(
                    tx_json, client, wallet
                )

                if not success:
                    logger.error(f"Transaction preparation failed: {msg}")
                    return f"Transaction preparation failed: {msg}"

                # Ask for confirmation
                logger.info("Preparing transaction confirmation message")
                return {
                    "type": "transaction_confirmation",
                    "message": f"Ready to submit {tx_json['TransactionType']} transaction. Details:\n{json.dumps(prepared_tx.to_dict(), indent=2)}\n\nDo you want to proceed?",
                    "prepared_tx": prepared_tx,
                }

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}", exc_info=True)
                return "Could not parse transaction JSON from response"

        logger.info("Returning conversation response")
        return response

    except Exception as e:
        logger.error(f"Error in conversation handler: {e}", exc_info=True)
        return f"Error processing request: {str(e)}"


# Add transaction execution handler
async def execute_transaction(prepared_tx, client, wallet):
    """Execute a prepared transaction after confirmation"""
    logger.info("Executing prepared transaction")
    logger.debug(f"Transaction details: {prepared_tx}")

    try:
        logger.info("Submitting transaction")
        result = await submit_transaction(prepared_tx, client, wallet)
        logger.debug(f"Transaction result: {result}")

        if result.is_successful():
            logger.success("Transaction executed successfully")
            return {
                "success": True,
                "message": "Transaction submitted successfully",
                "details": result.result,
            }
        else:
            logger.error(f"Transaction failed: {result.result}")
            return {
                "success": False,
                "message": "Transaction failed",
                "details": result.result,
            }

    except Exception as e:
        logger.error(f"Transaction execution error: {e}", exc_info=True)
        return {"success": False, "message": f"Error executing transaction: {str(e)}"}


# Move this function up, after other async function definitions 
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

# Add after the document upload section and before the chat interface
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
        response = asyncio.run(chat_with_knowledge_base(prompt))
        logger.debug(f"Chat response generated: {response}")
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
        
        # Single transaction handling flow
        if "```json" in response and "Would you like me to prepare and submit this transaction?" in response:
            review_container = st.container()
            
            with review_container:
                if st.button("📝 Review Transaction", type="secondary"):
                    # Transaction details section
                    st.markdown("### 🔍 Transaction Review")
                    
                    # Extract and parse JSON
                    json_str = response[response.find("```json") + 7 : response.rfind("```")].strip()
                    tx_data = json.loads(json_str)
                    
                    # Display details in two columns
                    left_col, right_col = st.columns(2)
                    
                    with left_col:
                        st.markdown("#### Transaction Summary")
                        st.markdown(f"**Type:** {tx_data['TransactionType']}")
                        st.markdown(f"**Account:** {tx_data['Account']}")
                        for key, value in tx_data.items():
                            if key not in ['TransactionType', 'Account']:
                                st.markdown(f"**{key}:** {value}")
                    
                    with right_col:
                        st.markdown("#### Raw Transaction")
                        st.code(json.dumps(tx_data, indent=2), language="json")
                    
                    # Warning and submit button
                    st.warning("⚠️ Please review the transaction details carefully")
                    
                    if st.button("🚀 Submit Transaction", type="primary"):
                        with st.spinner("Processing transaction..."):
                            try:
                                async def process_transaction():
                                    prepared_tx = await prepare_transaction(
                                        tx_data["TransactionType"],
                                        tx_data,
                                        st.session_state.xrpl_client,
                                        st.session_state.wallet
                                    )
                                    result = await submit_transaction(
                                        prepared_tx,
                                        st.session_state.xrpl_client,
                                        st.session_state.wallet
                                    )
                                    return result
                                
                                result = asyncio.run(process_transaction())
                                st.success("✅ Transaction submitted successfully!")
                                st.json(result)
                                
                            except Exception as e:
                                st.error(f"❌ Transaction failed: {str(e)}")


