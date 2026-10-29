"""Decode OpenAI Chat and Responses media content parts into canonical parts.

Text, image, video, audio, and file parts are flattened in the caller's order. Each
media part carries exactly one of inline data, an http(s) URL, or a provider
handle: an OpenAI Files ``file_id`` becomes an OpenAI-scoped handle, and an
``s3://``, ``gs://``, or Gemini Files URI becomes a Bedrock, Vertex, or Gemini
handle. The gateway never fetches or uploads media on the caller's behalf.
"""

from __future__ import annotations

from exp.common.models.content import (
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    MessageContentPart,
    TextContentPart,
    audio_part_from_input_audio,
    document_part_from_file_data,
    image_part_from_url,
    media_handle_from_uri,
    video_part_from_url,
)
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.wire_models import (
    _ChatAudioPart,
    _ChatFilePart,
    _ChatImagePart,
    _ChatVideoPart,
    _ContentPart,
    _ResponsesFilePart,
    _ResponsesImagePart,
    _TextPart,
)


def message_content(
    content: str | tuple[_ContentPart, ...] | None,
    param: str,
) -> tuple[str | None, tuple[MessageContentPart, ...]]:
    """Flatten wire content parts, retaining attachments in the caller's order.

    Args:
        content: Wire content: plain text, ordered parts, or absent.
        param: Public parameter path used to report an invalid attachment.

    Returns:
        The flattened text and, only for a message that carries an image,
        a video, audio, or a document, the ordered canonical parts. A text-only
        message keeps its previous representation exactly, so nothing
        downstream changes for it.

    Raises:
        OpenAIProtocolError: An image or video reference is not a supported
            URL or base64 data URL, an audio part is not base64 WAV or MP3,
            or a file is not an inline PDF.
    """
    if content is None or isinstance(content, str):
        return content, ()
    parts: list[MessageContentPart] = []
    for index, part in enumerate(content):
        if isinstance(part, _TextPart):
            # An empty text part carries no content and contributes nothing to
            # the flattened text, while Anthropic and Gemini reject an empty
            # block outright. Real clients emit one beside an attachment
            # (OpenCode 1.18.26, captured live 2026-09-02), so it is dropped
            # here rather than failing a turn that does carry an image.
            if part.text:
                parts.append(TextContentPart(text=part.text))
            continue
        if isinstance(part, _ChatVideoPart):
            try:
                parts.append(video_part_from_url(part.video_url.url))
            except ValueError as exc:
                location = f"{param}.{index}.video_url"
                raise invalid_field(
                    location,
                    f"'{location}' must be an http(s) URL, a base64 data URL of an MP4, "
                    "MPEG, QuickTime, WebM, FLV, 3GPP, or WMV video, or an s3://, gs://, "
                    "or Gemini Files URI of an uploaded video.",
                ) from exc
            continue
        if isinstance(part, _ChatAudioPart):
            audio = part.input_audio
            try:
                parts.append(audio_part_from_input_audio(audio.data, audio.format))
            except ValueError as exc:
                location = f"{param}.{index}.input_audio"
                hint = f"'{location}' must carry base64 audio data with format 'wav' or 'mp3'."
                raise invalid_field(location, hint) from exc
            continue
        if isinstance(part, (_ChatFilePart, _ResponsesFilePart)):
            parts.append(_document_part(part, f"{param}.{index}"))
            continue
        if isinstance(part, _ResponsesImagePart) and part.file_id is not None:
            parts.append(
                ImageContentPart(
                    handle=_openai_handle(part.file_id, f"{param}.{index}.file_id"),
                    detail=part.detail,
                )
            )
            continue
        url, detail = (
            (part.image_url.url, part.image_url.detail)
            if isinstance(part, _ChatImagePart)
            else (part.image_url or "", part.detail)
        )
        try:
            parts.append(image_part_from_url(url, detail=detail))
        except ValueError as exc:
            location = f"{param}.{index}.image_url"
            raise invalid_field(
                location,
                f"'{location}' must be an http(s) URL, a base64 data URL of a PNG, "
                "JPEG, GIF, or WebP image, or an s3://, gs://, or Gemini Files URI "
                "of an uploaded image whose suffix states its format.",
            ) from exc
    text = "".join(part.text for part in parts if part.kind == "text")
    if all(part.kind == "text" for part in parts):
        return text, ()
    return text, tuple(parts)


def _document_part(part: _ChatFilePart | _ResponsesFilePart, param: str) -> DocumentContentPart:
    """Convert one ``file`` or ``input_file`` part into the canonical document.

    Args:
        part: Validated caller file part.
        param: Public parameter path of the part, used to report an invalid file.

    Returns:
        The canonical document part carrying the caller's bytes or URL.

    Raises:
        OpenAIProtocolError: The file data is not an inline PDF, the file URL
            is neither http(s) nor a provider handle URI, or the file id is
            not an OpenAI Files handle.
    """
    if isinstance(part, _ChatFilePart):
        file_data, filename, location = part.file.file_data, part.file.filename, f"{param}.file"
        file_id = part.file.file_id
    else:
        file_data, filename, location = part.file_data, part.filename, param
        file_id = part.file_id
    if file_id is not None:
        return DocumentContentPart(
            handle=_openai_handle(file_id, f"{location}.file_id"), name=filename or None
        )
    if file_data is None:
        file_url = part.file_url if isinstance(part, _ResponsesFilePart) else None
        try:
            handle = media_handle_from_uri(file_url or "")
            if handle is not None:
                return DocumentContentPart(handle=handle, name=filename or None)
            return DocumentContentPart(url=file_url, name=filename or None)
        except ValueError as exc:
            raise invalid_field(
                f"{param}.file_url",
                f"'{param}.file_url' must be an http(s) URL or an s3://, gs://, or "
                "Gemini Files URI of an uploaded PDF.",
            ) from exc
    try:
        return document_part_from_file_data(file_data, name=filename)
    except ValueError as exc:
        raise invalid_field(
            f"{location}.file_data",
            f"'{location}.file_data' must be the base64 bytes of a PDF, bare or as a "
            "data:application/pdf;base64 URL, within the size limit.",
        ) from exc


def _openai_handle(file_id: str, location: str) -> MediaHandle:
    """Wrap one OpenAI Files id as a provider-scoped handle.

    Args:
        file_id: Caller ``file_id`` value.
        location: Public parameter path of the field, for the error.

    Returns:
        The OpenAI-scoped handle.

    Raises:
        OpenAIProtocolError: The id does not have the ``file-...`` shape.
    """
    try:
        return MediaHandle(provider="openai", reference=file_id)
    except ValueError as exc:
        raise invalid_field(
            location, f"'{location}' must be an OpenAI Files id of the form file-..."
        ) from exc
