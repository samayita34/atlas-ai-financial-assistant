"""
app/bot/telegram_bot.py

Telegram bot interface for Atlas AI Financial Assistant.

Handles incoming messages, commands, voice notes, and document uploads
from Telegram users, delegating to the LangGraph agent and backing
services (MemoryService, DocumentService, SpeechService).
"""

from __future__ import annotations

import io
import logging
import traceback
from typing import BinaryIO, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from app.config import Environment, get_settings
from app.database.database import AsyncSessionLocal
from app.database.models import MessageRole, User
from app.langgraph.graph import _DEV_ERROR_PREFIX, run_agent
from app.services.document_service import DocumentIngestionError, DocumentService
from app.services.financial_data_service import FinancialDataService
from app.services.memory_service import MemoryService
from app.services.speech_service import SpeechService
from app.services.watchlist_service import WatchlistService

logger = logging.getLogger(__name__)

router = Router(name="telegram_bot_router")

# Shared global service references set during setup_bot
_memory_service: Optional[MemoryService] = None
_financial_data_service: Optional[FinancialDataService] = None
_document_service: Optional[DocumentService] = None
_speech_service: Optional[SpeechService] = None
_watchlist_service: Optional[WatchlistService] = None


def create_bot() -> Bot:
    """
    Construct an aiogram Bot instance using the configured Telegram token.
    """
    settings = get_settings()
    token = settings.telegram_bot_token.get_secret_value()
    return Bot(
        token=token,
        default=DefaultBotProperties(
            parse_mode=ParseMode.MARKDOWN,
        ),
    )


# ---------------------------------------------------------------------------
# Helper: User Sync
# ---------------------------------------------------------------------------

async def _get_or_create_telegram_user(msg: Message) -> User:
    """
    Ensure the user corresponding to the Telegram message sender exists in the DB.
    """
    telegram_id = msg.from_user.id if msg.from_user else 0
    username = msg.from_user.username if msg.from_user else None
    full_name = msg.from_user.full_name if msg.from_user else None

    async with AsyncSessionLocal() as session:
        memory = MemoryService(session)
        user = await memory.get_or_create_user(
            telegram_id=telegram_id,
            username=username,
            full_name=full_name,
        )
        return user


def _extract_bytes(file_obj: BinaryIO | io.BytesIO | bytes | None) -> bytes:
    """
    Safely extract raw bytes from a downloaded Telegram file object,
    handling bytes, BytesIO, or standard BinaryIO streams.
    """
    if file_obj is None:
        return b""
    if isinstance(file_obj, bytes):
        return file_obj
    if isinstance(file_obj, io.BytesIO):
        return file_obj.getvalue()
    if hasattr(file_obj, "read"):
        return file_obj.read()
    return b""



# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """
    Handle the /start command. Registers the user and sends a welcome message.
    """
    if not message.from_user:
        return

    user = await _get_or_create_telegram_user(message)

    welcome_text = (
        f"👋 Welcome to *Atlas AI Financial Assistant*, {user.full_name or 'there'}!\n\n"
        "I am your personal AI financial analyst. Here is what I can do for you:\n"
        "• 📈 *Market Research*: Ask about stocks, financial metrics, and company news.\n"
        "• 📄 *Document Analysis*: Upload financial PDFs (earnings reports, 10-Ks) and ask questions.\n"
        "• 🎙️ *Voice Notes*: Send me voice messages with your questions.\n"
        "• 📋 *Watchlist*: Track your favorite companies.\n\n"
        "Type your question or upload a document to get started!"
    )
    await message.answer(welcome_text)


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    """
    Handle the /help command.
    """
    help_text = (
        "💡 *Atlas Help & Capabilities*\n\n"
        "1. *Company Research*: 'What is Apple's current stock price and recent news?'\n"
        "2. *Document Q&A*: Send a PDF document and ask 'Summarize the revenue breakdown.'\n"
        "3. *Watchlist*: 'Add AAPL and MSFT to my watchlist' or 'Show my watchlist'.\n"
        "4. *Voice Notes*: Tap and hold the mic button to speak your query.\n"
    )
    await message.answer(help_text)


@router.message(F.document)
async def handle_document_upload(message: Message) -> None:
    """
    Handle user document (PDF) uploads.
    """
    if not message.document or not message.from_user:
        return

    doc = message.document
    filename = doc.file_name or "uploaded_document.pdf"
    mime_type = doc.mime_type or ""

    if not (filename.lower().endswith(".pdf") or "pdf" in mime_type.lower()):
        await message.answer("⚠️ Only PDF documents are currently supported. Please upload a PDF file.")
        return

    status_msg = await message.answer("📥 Downloading and ingesting your document...")

    try:
        user = await _get_or_create_telegram_user(message)

        # Download document file bytes from Telegram
        bot = message.bot
        if bot is None:
            await status_msg.edit_text("❌ Telegram bot session unavailable.")
            return

        file_info = await bot.get_file(doc.file_id)
        if not file_info.file_path:
            await status_msg.edit_text("❌ Failed to retrieve file path from Telegram.")
            return

        file_bytes_io = await bot.download_file(file_info.file_path)
        file_bytes = _extract_bytes(file_bytes_io)

        async with AsyncSessionLocal() as session:
            doc_service = DocumentService(session)
            ingested_doc = await doc_service.ingest_document(
                user_id=user.id,
                filename=filename,
                file_content=file_bytes,
                content_type="application/pdf",
            )

        await status_msg.edit_text(
            f"✅ *Document Ingested Successfully!*\n\n"
            f"📄 *Title*: {filename}\n"
            f"🆔 *ID*: `{ingested_doc.id}`\n\n"
            "You can now ask questions about this document!"
        )

    except DocumentIngestionError as exc:
        logger.error("Document ingestion error: %s", exc)
        await status_msg.edit_text(f"❌ Document ingestion failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during document upload")
        await status_msg.edit_text(f"❌ An error occurred while processing your document: {exc}")


@router.message(F.voice)
async def handle_voice_message(message: Message) -> None:
    """
    Handle voice messages by transcribing audio and running the agent.
    """
    if not message.voice or not message.from_user:
        return

    status_msg = await message.answer("🎙️ Transcribing voice message...")

    try:
        user = await _get_or_create_telegram_user(message)

        bot = message.bot
        if bot is None:
            await status_msg.edit_text("❌ Telegram bot session unavailable.")
            return

        file_info = await bot.get_file(message.voice.file_id)
        if not file_info.file_path:
            await status_msg.edit_text("❌ Failed to retrieve voice file from Telegram.")
            return

        file_bytes_io = await bot.download_file(file_info.file_path)
        audio_bytes = _extract_bytes(file_bytes_io)

        speech_svc = _speech_service or SpeechService()
        transcribed_text = await speech_svc.transcribe_audio(audio_bytes)

        if not transcribed_text.strip():
            await status_msg.edit_text("❓ Could not transcribe any clear speech from your voice message.")
            return

        await status_msg.edit_text(f"🗣️ *Transcribed*: \"{transcribed_text}\"\n\n🤔 Processing your query...")

        await _process_user_query(
            message=message,
            user=user,
            text=transcribed_text,
            status_msg=status_msg,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling voice message")
        await status_msg.edit_text(f"❌ Failed to process voice message: {exc}")


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text_message(message: Message) -> None:
    """
    Handle standard text questions from users.
    """
    if not message.text or not message.from_user:
        return

    user = await _get_or_create_telegram_user(message)
    await _process_user_query(message=message, user=user, text=message.text)


async def _process_user_query(
    message: Message,
    user: User,
    text: str,
    status_msg: Optional[Message] = None,
) -> None:
    """
    Save user message, run LangGraph agent, save assistant message, and reply.
    """
    async with AsyncSessionLocal() as session:
        memory = MemoryService(session)
        await memory.add_message(
            user_id=user.id,
            role=MessageRole.USER,
            content=text,
        )

    # Instantiate default services if global references were not supplied via setup_bot
    try:
        async with AsyncSessionLocal() as session:
            mem_svc = _memory_service or MemoryService(session)
            doc_svc = _document_service or DocumentService(session)
            fin_svc = _financial_data_service or FinancialDataService()
            wl_svc = _watchlist_service or WatchlistService(financial_data_service=fin_svc)

            response_text = await run_agent(
                user_id=user.telegram_id,
                text=text,
                memory_service=mem_svc,
                financial_data_service=fin_svc,
                document_service=doc_svc,
                watchlist_service=wl_svc,
            )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception(
            "run_agent raised an unexpected exception for user_id=%s", user.telegram_id
        )
        settings = get_settings()
        if settings.environment == Environment.LOCAL:
            # Plain text with sentinel — telegram_bot sends this without ParseMode
            # so Telegram never rejects it due to Markdown-special chars in tracebacks.
            response_text = (
                f"{_DEV_ERROR_PREFIX}\n"
                f"[DEV] run_agent raised {type(exc).__name__}\n"
                f"{exc}\n\n"
                f"--- traceback (last 1500 chars) ---\n{tb[-1500:]}"
            )
        else:
            response_text = (
                "I ran into an issue putting that together. "
                "Could you try rephrasing or asking again in a moment?"
            )

    async with AsyncSessionLocal() as session:
        memory = MemoryService(session)
        await memory.add_message(
            user_id=user.id,
            role=MessageRole.ASSISTANT,
            content=response_text,
        )

    # Dev error messages are plain text (no Markdown) so Telegram doesn't
    # reject them when the traceback contains Markdown-special characters.
    # Use an explicit parse_mode= argument (not **kwargs) so the type checker
    # can resolve the exact type instead of spreading str|None over every param.
    effective_parse_mode: Optional[str] = (
        None if response_text.startswith(_DEV_ERROR_PREFIX) else ParseMode.MARKDOWN
    )

    if status_msg:
        try:
            await status_msg.edit_text(response_text, parse_mode=effective_parse_mode)
        except TelegramBadRequest as exc:
            if "can't parse entities" in str(exc).lower():
                logger.warning(
                    "Telegram rejected Markdown entities in response for user_id=%s; resending as plain text. error=%s",
                    user.telegram_id,
                    exc,
                )
                await status_msg.edit_text(response_text, parse_mode=None)
            else:
                raise
    else:
        try:
            await message.answer(response_text, parse_mode=effective_parse_mode)
        except TelegramBadRequest as exc:
            if "can't parse entities" in str(exc).lower():
                logger.warning(
                    "Telegram rejected Markdown entities in response for user_id=%s; resending as plain text. error=%s",
                    user.telegram_id,
                    exc,
                )
                await message.answer(response_text, parse_mode=None)
            else:
                raise


# ---------------------------------------------------------------------------
# Setup Bot & Dispatcher
# ---------------------------------------------------------------------------

def setup_bot(
    memory_service: Optional[MemoryService] = None,
    financial_data_service: Optional[FinancialDataService] = None,
    document_service: Optional[DocumentService] = None,
    speech_service: Optional[SpeechService] = None,
    watchlist_service: Optional[WatchlistService] = None,
    bot: Optional[Bot] = None,
) -> tuple[Bot, Dispatcher]:
    """
    Initialize and wire up the Bot and Dispatcher with router handlers.
    """
    global _memory_service, _financial_data_service, _document_service, _speech_service, _watchlist_service

    if memory_service is not None:
        _memory_service = memory_service
    if financial_data_service is not None:
        _financial_data_service = financial_data_service
    if document_service is not None:
        _document_service = document_service
    if speech_service is not None:
        _speech_service = speech_service
    if watchlist_service is not None:
        _watchlist_service = watchlist_service

    active_bot = bot or create_bot()
    dp = Dispatcher()
    dp.include_router(router)

    return active_bot, dp
