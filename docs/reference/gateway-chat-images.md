# Generated images in chat

Image-emitting Chat models on Gemini and OpenAI-compatible wires return inline raster images
in `choices[].delta.images[].image_url.url` while streaming and
`choices[].message.images[].image_url.url` when buffered. Each URL is a complete base64 PNG,
JPEG, WebP, or GIF. Text and token usage remain in their ordinary fields. Replay generated
images as assistant `content` parts with `type: "image_url"`; Gemini reconstructs caller-owned
history with its documented signature-validation bypass, as it does for replayed function calls.
This does not preserve provider-private reasoning signatures. Assistant image history is refused
on fallback wires that cannot preserve it.
Image lanes request text and images explicitly on Gemini and OpenRouter, allow bounded SSE
frames up to 64 MiB, and share the 64 MiB aggregate output bound. Output guardrails that inspect
text cannot inspect image pixels, so guarded image lanes buffer and fail closed before delivery.
Each image is limited to 16 MiB of base64 text and 12 MiB of compressed image bytes before
base64 decoding. Parsed dimensions must stay within 16,384 pixels per dimension, and the
decoded pixel buffer must fit within 64 MiB before raster allocation (including a 4,096-square
RGBA image). Decoder working memory also receives a 64 MiB best-effort limit; this is not a
total process-memory ceiling. Header parsing, compressed-data copies, and provider frame/JSON
buffers add overhead. Raster and compressed buffers are released before constructing the output URL.
WebP canvas and embedded frame headers, including animation frames, are checked before codec
decoding; their dimensions must agree, and their RGBA buffers must also fit within 64 MiB.
Known image-emitting models on unsupported completion surfaces are refused before dispatch; use
Chat Completions or a separately supported Images API route. The Images API capability remains
independent.
