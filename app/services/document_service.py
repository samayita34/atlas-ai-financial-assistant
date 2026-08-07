"""Document ingestion and retrieval service for Atlas AI Financial Assistant.

This module implements :class:`DocumentService`, responsible for the full
lifecycle of user-uploaded financial documents (PDF only, at present):

    1. Accepting raw uploaded PDF bytes.
    2. Extracting plain text from the PDF.
    3. Splitting the extracted text into overlapping semantic chunks.
    4. Generating vector embeddings for each chunk via the Gemini embedding
       model.
    5. Persisting document metadata and chunk embeddings via SQLAlchemy 2.0
       async ORM models.
    6. Retrieving the most relevant chunks for a given user query using
       cosine-distance similarity search (pgvector).
    7. Assembling clean, LLM-ready context strings from retrieved chunks.

Assumptions about collaborating modules (see accompanying message for full
detail, since these files were not available at generation time):

- ``app.database.models`` exposes ``Document`` and ``DocumentChunk`` ORM
  models, with ``DocumentChunk.embedding`` typed as
  ``pgvector.sqlalchemy.Vector`` and supporting the ``.cosine_distance()``
  comparator for similarity search.
- ``app.config.get_settings()`` returns a ``Settings`` object exposing
  ``GEMINI_API_KEY``, ``GEMINI_EMBEDDING_MODEL``, ``EMBEDDING_DIMENSION``,
  ``DOCUMENT_CHUNK_SIZE``, and ``DOCUMENT_CHUNK_OVERLAP``.

All database access is performed through an injected ``AsyncSession``,
keeping this service free of any knowledge of session/engine lifecycle
(Single Responsibility / Dependency Inversion).
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Final

from google import genai
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import Document, DocumentChunk

logger: Final[logging.Logger] = logging.getLogger(__name__)

# Sentence/paragraph boundary separators used by the recursive splitter,
# ordered from "most preferred split point" to "least preferred".
_SPLIT_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", " ", "")


class DocumentIngestionError(Exception):
    """Raised when a document cannot be parsed, chunked, or embedded."""


class DocumentService:
    """Handles ingestion and retrieval of user financial documents.

    The service accepts uploaded PDF documents, converts them into
    searchable, embedded text chunks, and later retrieves the most relevant
    chunks for a given natural-language query so an LLM can ground its
    answers in the user's own financial documents (statements, reports,
    contracts, etc.).

    Instances are cheap to construct and hold no long-lived state beyond
    the injected database session and a lazily-created Gemini client, so a
    new instance should be created per request/unit-of-work, matching the
    lifecycle of the injected :class:`AsyncSession`.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the document service.

        Args:
            session: An active SQLAlchemy 2.0 async session, injected by the
                caller (e.g. a FastAPI dependency). This service never
                creates, commits ownership of, or closes the session itself
                beyond issuing ``flush``/``commit`` calls for its own unit
                of work.
        """
        self._session = session
        self._settings = get_settings()
        self._genai_client = genai.Client(
            api_key=self._settings.gemini_api_key.get_secret_value()
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_document(
        self,
        *,
        user_id: int,
        filename: str,
        file_content: bytes,
        content_type: str = "application/pdf",
    ) -> Document:
        """Ingest an uploaded PDF document end-to-end.

        Extracts text, splits it into chunks, generates embeddings for each
        chunk, and persists the document plus its chunks in a single unit
        of work.

        Args:
            user_id: Identifier of the user who owns this document.
            filename: Original filename as uploaded by the user.
            file_content: Raw bytes of the uploaded PDF file.
            content_type: MIME type of the uploaded file. Defaults to
                ``"application/pdf"``, the only type currently supported.

        Returns:
            The persisted :class:`Document` instance, with its ``chunks``
            relationship populated.

        Raises:
            DocumentIngestionError: If text extraction, chunking, or
                embedding generation fails, or if no extractable text is
                found in the document.
        """
        if content_type != "application/pdf":
            raise DocumentIngestionError(
                f"Unsupported content type '{content_type}'. Only PDF documents "
                "are currently supported."
            )

        document = Document(
            user_id=user_id,
            title=filename,
            source=filename,
            is_processed=False,
        )
        self._session.add(document)
        await self._session.flush()  # Assigns document.id without committing.

        try:
            raw_text = self._extract_text_from_pdf(file_content)
            if not raw_text.strip():
                raise DocumentIngestionError(
                    f"No extractable text found in document '{filename}'. "
                    "The PDF may be scanned/image-only and require OCR."
                )

            chunks = self._split_into_chunks(
                raw_text,
                chunk_size=self._settings.rag_chunk_size,
                chunk_overlap=self._settings.rag_chunk_overlap,
            )
            if not chunks:
                raise DocumentIngestionError(
                    f"Document '{filename}' produced no chunks after splitting."
                )

            embeddings = await self._generate_embeddings_batch(chunks)

            document_chunks = [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk_text,
                    embedding=embedding,
                    created_at=datetime.now(timezone.utc),
                )
                for index, (chunk_text, embedding) in enumerate(
                    zip(chunks, embeddings, strict=True)
                )
            ]
            self._session.add_all(document_chunks)

            document.is_processed = True
            await self._session.commit()
            await self._session.refresh(document)

            logger.info(
                "Ingested document id=%s filename=%s chunks=%d",
                document.id,
                filename,
                len(document_chunks),
            )
            return document

        except DocumentIngestionError:
            document.is_processed = False
            await self._session.commit()
            raise
        except Exception as exc:  # noqa: BLE001 - convert to domain error
            document.is_processed = False
            await self._session.commit()
            logger.exception("Failed to ingest document '%s'", filename)
            raise DocumentIngestionError(
                f"Failed to ingest document '{filename}': {exc}"
            ) from exc

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract plain text from raw PDF bytes.

        Args:
            file_content: Raw bytes of the PDF file.

        Returns:
            The concatenated text of all pages, separated by double
            newlines to preserve page boundaries as paragraph breaks.

        Raises:
            DocumentIngestionError: If the PDF cannot be parsed at all
                (e.g. corrupted file or invalid format).
        """
        try:
            reader = PdfReader(io.BytesIO(file_content))
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            raise DocumentIngestionError(f"Unable to parse PDF file: {exc}") from exc

        pages_text: list[str] = []
        for page_number, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - skip unreadable pages
                logger.warning(
                    "Failed to extract text from page %d; skipping.", page_number
                )
                page_text = ""
            if page_text.strip():
                pages_text.append(page_text.strip())

        return "\n\n".join(pages_text)

    def _split_into_chunks(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks along natural boundaries.

        Implements a recursive character-based splitter: it attempts to
        split on the most "semantic" separator first (paragraph breaks),
        falling back to progressively finer-grained separators (line
        breaks, sentence breaks, spaces, and finally raw characters) only
        for pieces that still exceed ``chunk_size``. Adjacent chunks share
        ``chunk_overlap`` characters of context to reduce information loss
        at chunk boundaries.

        Args:
            text: The full document text to split.
            chunk_size: Target maximum number of characters per chunk.
            chunk_overlap: Number of trailing characters from one chunk to
                repeat at the start of the next chunk.

        Returns:
            A list of non-empty text chunks, each at most approximately
            ``chunk_size`` characters (individual unsplittable tokens may
            exceed this slightly).

        Raises:
            DocumentIngestionError: If ``chunk_overlap`` is not smaller
                than ``chunk_size``.
        """
        if chunk_overlap >= chunk_size:
            raise DocumentIngestionError(
                "chunk_overlap must be smaller than chunk_size "
                f"(got chunk_size={chunk_size}, chunk_overlap={chunk_overlap})."
            )

        raw_pieces = self._recursive_split(text, chunk_size, list(_SPLIT_SEPARATORS))
        merged = self._merge_with_overlap(raw_pieces, chunk_size, chunk_overlap)
        return [piece.strip() for piece in merged if piece.strip()]

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        separators: list[str],
    ) -> list[str]:
        """Recursively split ``text`` using the first workable separator.

        Args:
            text: Text segment to split.
            chunk_size: Target maximum chunk size in characters.
            separators: Ordered candidate separators, most preferred first.
                The empty string as a final entry forces a hard character
                split as a last resort.

        Returns:
            A list of text segments, each ideally no larger than
            ``chunk_size`` characters.
        """
        if len(text) <= chunk_size:
            return [text] if text else []

        if not separators:
            return [text]

        separator, *remaining_separators = separators

        if separator == "":
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        parts = text.split(separator) if separator else [text]
        if len(parts) == 1:
            # Separator not present in this segment; try the next one.
            return self._recursive_split(text, chunk_size, remaining_separators)

        results: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) <= chunk_size:
                results.append(part)
            else:
                results.extend(
                    self._recursive_split(part, chunk_size, remaining_separators)
                )
        return results

    def _merge_with_overlap(
        self,
        pieces: list[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Greedily merge small pieces into chunks near ``chunk_size``.

        Consecutive pieces are concatenated (joined by a single space)
        until adding the next piece would exceed ``chunk_size``. Each new
        chunk is seeded with the trailing ``chunk_overlap`` characters of
        the previous chunk to preserve local context across boundaries.

        Args:
            pieces: Ordered text segments produced by
                :meth:`_recursive_split`.
            chunk_size: Target maximum chunk size in characters.
            chunk_overlap: Number of trailing characters to carry over
                into the next chunk.

        Returns:
            A list of merged chunks.
        """
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                overlap_seed = current[-chunk_overlap:] if chunk_overlap else ""
                current = f"{overlap_seed} {piece}".strip() if overlap_seed else piece
            else:
                # Single piece already exceeds chunk_size; keep as-is.
                chunks.append(piece)
                current = ""

        if current:
            chunks.append(current)

        return chunks

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding vector for the given text.

        Args:
            text: Text content to embed.

        Returns:
            The embedding as a list of floats, with dimensionality equal
            to ``settings.EMBEDDING_DIMENSION``.

        Raises:
            DocumentIngestionError: If the Gemini embedding API call fails.
        """
        embeddings = await self._generate_embeddings_batch([text])
        return embeddings[0]

    async def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts via the Gemini API.

        The underlying ``google-genai`` client is synchronous, so the call
        is offloaded to a worker thread via :func:`asyncio.to_thread` to
        avoid blocking the event loop.

        Args:
            texts: List of text chunks to embed, in order.

        Returns:
            A list of embedding vectors, one per input text, in the same
            order as ``texts``.

        Raises:
            DocumentIngestionError: If the embedding API call fails or
                returns a mismatched number of embeddings.
        """
        try:
            response = await asyncio.to_thread(
                self._genai_client.models.embed_content,
                model=self._settings.embedding_model,
                contents=texts,
            )
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            raise DocumentIngestionError(
                f"Gemini embedding request failed: {exc}"
            ) from exc

        raw_embeddings = response.embeddings or []
        embeddings = [
            list(embedding.values) if embedding.values is not None else []
            for embedding in raw_embeddings
        ]

        if len(embeddings) != len(texts):
            raise DocumentIngestionError(
                f"Expected {len(texts)} embeddings from Gemini, got "
                f"{len(embeddings)}."
            )

        return embeddings

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve_relevant_chunks(
        self,
        *,
        user_id: int,
        query: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """Retrieve the most relevant document chunks for a user query.

        Embeds the query text and performs a cosine-similarity nearest
        neighbor search (via pgvector's ``cosine_distance`` comparator)
        restricted to documents owned by ``user_id``.

        Args:
            user_id: Identifier of the user whose documents should be
                searched. Ensures users can never retrieve another user's
                document chunks.
            query: Natural-language query to search for.
            top_k: Maximum number of chunks to return, ordered from most
                to least relevant.

        Returns:
            A list of the most relevant :class:`DocumentChunk` instances,
            ordered by ascending cosine distance (i.e. descending
            relevance). May be shorter than ``top_k`` if fewer chunks
            exist.

        Raises:
            DocumentIngestionError: If query embedding generation fails.
        """
        if not query.strip():
            return []

        query_embedding = await self._generate_embedding(query)

        statement = (
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.user_id == user_id)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def get_context_for_query(
        self,
        *,
        user_id: int,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Build a clean, LLM-ready context string for a user query.

        Retrieves the most relevant chunks for the query and concatenates
        their content into a single string suitable for direct inclusion
        in an LLM prompt, with lightweight source annotations so the model
        can (optionally) reference which document a fact came from.

        Args:
            user_id: Identifier of the user whose documents should be
                searched.
            query: Natural-language query to build context for.
            top_k: Maximum number of chunks to include in the context.

        Returns:
            A newline-separated context string. Returns an empty string if
            no relevant chunks are found (callers should treat this as
            "no document context available", not as an error).
        """
        chunks = await self.retrieve_relevant_chunks(
            user_id=user_id, query=query, top_k=top_k
        )
        if not chunks:
            return ""

        document_ids = {chunk.document_id for chunk in chunks}
        documents_by_id = await self._get_documents_by_ids(document_ids)

        context_sections: list[str] = []
        for chunk in chunks:
            document = documents_by_id.get(chunk.document_id)
            source_label = document.title if document else "unknown document"
            context_sections.append(f"[Source: {source_label}]\n{chunk.content}")

        return "\n\n".join(context_sections)

    async def _get_documents_by_ids(
        self, document_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, Document]:
        """Fetch documents by primary key, keyed by their ``id``.

        Args:
            document_ids: Set of document primary keys to fetch.

        Returns:
            A mapping of document id to :class:`Document` instance, for
            every id in ``document_ids`` that exists.
        """
        if not document_ids:
            return {}

        statement = select(Document).where(Document.id.in_(document_ids))
        result = await self._session.execute(statement)
        return {document.id: document for document in result.scalars().all()}