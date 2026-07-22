"""Single resolver for every vision_analyze image source -> bytes + mime.

All source handling (data:/http(s)/file/local/container) funnels through
:func:`resolve_image_source` so size and magic-byte checks are enforced exactly
once.  Returns raw bytes (not a path): the downstream step is base64 -> data URL
(RFC 2397) and provider base64 content blocks.

Security (terminal-backend confinement, GHSA-gpxw-6wxv-w3qq): under a non-local
terminal backend the file tools are confined to the sandbox (SECURITY.md 2.2),
but vision read images host-side. This resolver enforces the same boundary:

  * local backend            -> read any host path (chosen posture, unchanged)
  * non-local backend:
      path in a media cache   -> host-read (the gateway/download caches live on
                                 the host and are bind-mounted into the sandbox)
      path anywhere else      -> read the bytes *inside the sandbox* via exec-read
                                 (the agent can already ``cat`` any container file;
                                 this stays within the sandbox boundary and never
                                 reaches the host's ``/etc/passwd`` / ``~/.ssh``).

So a prompt-injected ``vision_analyze('/etc/passwd')`` under Docker reads the
*container's* file (what every other tool sees), not the host's — no escape —
while container-only images (tmpfs ``/workspace``, root-owned) are still
deliverable. This is the unified delivery + confinement model: the same
mechanism that fixes "vision can't see container files" also closes the escape.
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Raw-bytes INGEST budget — what the resolver will load before handing off.
# This is deliberately the 50MB download cap (tools/vision_tools._VISION_MAX_DOWNLOAD_BYTES),
# NOT the 20MB provider payload cap. The 20MB cap (_MAX_BASE64_BYTES) is a
# *post-resize* limit enforced at the call sites: an oversized raw image must
# still reach the resizer so it can be downscaled under the payload cap. Capping
# raw bytes at 20MB here would reject every 20-50MB photo before resize can run.
_MAX_INGEST_BYTES = 50 * 1024 * 1024


class ImageResolutionError(Exception):
    def __init__(self, message: str, *, src: str = "", origin: str = ""):
        super().__init__(message)
        self.src, self.origin = src, origin


class UnsupportedScheme(ImageResolutionError):
    pass


class SourceUnsafe(ImageResolutionError):  # SSRF / path-allowlist
    pass


class SourceTooLarge(ImageResolutionError):
    pass


class SourceNotFound(ImageResolutionError):
    pass


class NotAnImage(ImageResolutionError):
    pass


class CorruptImage(NotAnImage):
    """Magic bytes and header look right, but the image is unsafe to embed.

    Two distinct failure modes both raise this:

    1. Truncated / corrupt pixel data — a truncated download (or a screenshot
       tool that wrote a partial file, see issue #69078) can keep a fully-
       formed PNG/JPEG signature and header while the compressed data trails
       off mid-stream. The magic-byte sniff in ``_finalize`` can't catch
       that — it only looks at the first few bytes — so a corrupt file
       passes the sniff, gets base64-embedded into conversation history, and
       permanently poisons the session on replay (the bad bytes are
       immutable history).
    2. Decompression bombs — a small, well-formed file can still decode to
       an enormous pixel buffer. See ``verify_decodable_image`` for the
       pixel-count ceiling and PIL bomb-warning promotion that catch this
       before ``.load()`` allocates the buffer.

    This subclasses ``NotAnImage`` so existing callers that only catch the
    base class keep working, while callers that care can catch this
    specifically to distinguish "not an image at all" from "unsafe image".
    """
    pass


@dataclass
class ResolveContext:
    task_id: Optional[str] = None


@dataclass
class ResolvedImage:
    data: bytes
    mime: str
    origin: str  # one of: data | http | file | local | container


# Explicit URL scheme, e.g. "ftp://", "s3://". Bare Windows drive paths
# ("C:\x.png") don't match because they lack the "//".
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


async def resolve_image_source(src: str, ctx: ResolveContext) -> ResolvedImage:
    if not isinstance(src, str) or not src.strip():
        raise SourceNotFound("image_url is required", src=str(src))
    s = src.strip()
    if s.startswith("data:"):
        data, mime = _resolve_data_url(s)
        return _finalize(data, mime, "data", s)
    if s.startswith(("http://", "https://")):
        reason = _http_block_reason(s)
        if reason:
            raise SourceUnsafe(reason, src=s)
        return _finalize(await _download_to_bytes(s), "", "http", s)

    if _SCHEME_RE.match(s) and not s.lower().startswith("file://"):
        raise UnsupportedScheme(
            "Unrecognized image source scheme. Use an http(s) URL, a local "
            "file path, a file:// URI, or a data: URL.",
            src=s,
        )

    # Everything else is a filesystem path — including bare relative names
    # like "pic.png" (accepted on main; a path-shape gate here regressed them).
    candidate = s[len("file://"):] if s.lower().startswith("file://") else s
    p = Path(os.path.expanduser(candidate))
    # Confinement decision (see module docstring). Under a non-local backend
    # a path is host-readable ONLY if it lands in a media cache (after
    # translating a container-visible cache path back to its host mount);
    # every other path is read inside the sandbox via exec-read, so a host
    # path outside the caches never yields the host's bytes.
    host_target = _permitted_host_read_target(p, ctx)
    if host_target is not None and host_target.is_file():
        # Shared credential-read guard (agent.file_safety, #57698): refuse
        # secret-bearing files (.env, auth.json, ...) with an intentional,
        # specific error instead of relying on the magic-byte sniff to
        # reject them incidentally. Same chokepoint the image-gen/video-gen
        # provider plugins enforce on model-supplied local paths. Import is
        # best-effort (guard unavailability must not break image loading);
        # a real block always propagates.
        try:
            from agent.file_safety import raise_if_read_blocked
        except Exception:  # noqa: BLE001 — guard unavailable: proceed
            raise_if_read_blocked = None
        if raise_if_read_blocked is not None:
            try:
                raise_if_read_blocked(str(host_target))
            except ValueError as exc:
                raise SourceUnsafe(str(exc), src=s, origin="file")
        data = await asyncio.to_thread(host_target.read_bytes)
        return _finalize(data, "", "file", s)
    if _is_local_terminal_backend():
        # Local backend: any path was host-readable, so a miss simply means
        # the file doesn't exist — no sandbox to fall back to.
        raise SourceNotFound(f"image file not found: '{p}'", src=s, origin="file")
    # Not a permitted host read (or the host file is absent) -> read the
    # bytes inside the sandbox. Under a sandbox this reads the container's
    # filesystem, never the host's.
    return await _resolve_container_fallback(p, ctx, s)


def _resolve_data_url(s: str) -> tuple[bytes, str]:
    header, _, payload = s.partition(",")
    if ";base64" not in header:
        raise NotAnImage("data: URL must be base64-encoded", src=s[:64])
    declared = header[len("data:"):].split(";", 1)[0].strip() or "application/octet-stream"
    # Cheap pre-decode size gate on the encoded length (~4/3 expansion).
    if (len(payload) * 3) // 4 > _MAX_INGEST_BYTES:
        raise SourceTooLarge("data: URL exceeds size limit", src=s[:64])
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise NotAnImage(f"invalid base64 in data: URL: {exc}", src=s[:64])
    return data, declared  # real mime verified in _finalize via magic bytes


def _http_block_reason(url: str) -> Optional[str]:
    """Return a human-readable block reason, or None when the URL is allowed.

    Pre-flight short-circuit: policy-blocked URLs are refused BEFORE any
    network I/O. ``_download_image`` re-checks policy internally (per attempt
    and against the final redirect target) — that second evaluation is
    intentional, not redundant: this one guarantees no bytes move for a
    blocked URL; the inner one covers redirects and non-resolver callers.
    Preserves the specific website-policy message so the agent sees *why*.
    """
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access

    if not is_safe_url(url):
        return "blocked: unsafe or private URL"
    blocked = check_website_access(url)
    if blocked:
        return blocked.get("message") or "blocked by website policy"
    return None


async def _download_to_bytes(url: str) -> bytes:
    import tempfile

    from tools.vision_tools import _download_image

    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        # Enforces the 50MB stream cap, redirect SSRF guard, and website policy.
        await _download_image(url, tmp)
        return await asyncio.to_thread(tmp.read_bytes)
    except PermissionError as exc:  # website policy block
        raise SourceUnsafe(str(exc), src=url, origin="http")
    finally:
        tmp.unlink(missing_ok=True)


def _is_local_terminal_backend() -> bool:
    """True when the terminal backend runs directly on the host.

    Mirrors ``tools.browser_tool._is_local_backend`` and terminal_tool's own
    dispatch, which key off ``TERMINAL_ENV``.
    """
    return os.getenv("TERMINAL_ENV", "local").strip().lower() in ("local", "")


def _media_cache_roots() -> list:
    """Agent-managed media cache directories under HERMES_HOME (host side).

    The only host paths vision may read under a non-local backend: gateway-
    downloaded inbound media and the tools' own URL-download temp dirs. Covers
    the consolidated ``cache/`` layout and the legacy flat directories.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return [
        home / "cache",  # cache/images, cache/vision, cache/video(s), cache/audio
        home / "image_cache",
        home / "audio_cache",
        home / "video_cache",
        home / "temp_vision_images",
        home / "temp_video_files",
    ]


def _permitted_host_read_target(p: Path, ctx: ResolveContext) -> Optional[Path]:
    """Return the host path to read, or ``None`` if a host read is not permitted.

    - Local backend: any path is permitted (chosen posture). Returns ``p``.
    - Non-local backend: permitted only if the path resolves inside a media
      cache root. A container-visible cache path (e.g. ``/root/.hermes/cache/
      images/x.png``) is first translated back to its host mount; anything that
      is not under a cache returns ``None`` so the caller routes it to the
      in-sandbox exec-read instead of reading the host filesystem.
    """
    if _is_local_terminal_backend():
        try:
            return p.resolve()
        except Exception:  # noqa: BLE001 — unresolved path: let is_file() fail downstream
            return p

    from tools.credential_files import from_agent_visible_cache_path

    host_candidate = Path(from_agent_visible_cache_path(str(p)))
    try:
        real = host_candidate.resolve()
    except Exception:  # noqa: BLE001 — cannot resolve -> not a safe host read
        return None
    for root in _media_cache_roots():
        try:
            real.relative_to(root.resolve())
            return real
        except ValueError:
            continue
    return None


def _get_active_env(task_id: Optional[str]):
    if not task_id:
        return None
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id)
    except Exception:
        return None


async def _resolve_container_fallback(p: Path, ctx: ResolveContext, src: str) -> ResolvedImage:
    """Read the image bytes inside the sandbox (fail-closed when none exists).

    Reached when a host read is not permitted or the host file is absent. The
    agent can already ``cat`` any container file (file_operations.py reads
    root-owned mode-600 files this way), so this stays within the same sandbox
    boundary and never touches the host filesystem. ``--`` stops a leading-dash
    path from being parsed as a ``base64`` option; ``base64 -w0`` is GNU-only,
    so pipe through ``tr -d`` for BusyBox.

    Fail-closed: if there is no active sandbox env we refuse rather than falling
    back to a host read, so a non-cache host path under a sandbox never leaks.
    """
    import asyncio
    import shlex

    env = _get_active_env(ctx.task_id)
    if env is None:
        raise SourceNotFound(
            f"'{p}' is not reachable inside the sandbox and no active sandbox "
            f"session is available to read it",
            src=src, origin="container")

    # Bound the read INSIDE the sandbox: head -c caps at ingest-limit+1 bytes
    # so a huge file (or /dev/zero) can't stream unbounded base64 into host
    # memory — the +1 byte lets us distinguish "exactly at the cap" from
    # "over the cap" after decode. The input redirect (< path) avoids argv
    # entirely, so leading-dash paths can't be parsed as options; base64
    # -w0 is GNU-only, so pipe through tr -d for BusyBox.
    # env.execute is a blocking backend exec; keep it off the event loop so a
    # multi-MB base64 read doesn't stall every other coroutine.
    qp = shlex.quote(str(p))
    res = await asyncio.to_thread(
        env.execute,
        f"head -c {_MAX_INGEST_BYTES + 1} < {qp} | base64 | tr -d '\\n'")
    if res.get("returncode", 1) != 0:
        raise SourceNotFound(f"could not read '{p}' inside the sandbox", src=src, origin="container")
    try:
        data = base64.b64decode(res.get("output", ""), validate=True)
    except Exception as exc:
        raise NotAnImage(f"sandbox returned non-image data for '{p}': {exc}", src=src)
    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("image exceeds size limit", src=src, origin="container")
    return _finalize(data, "", "container", src)


_PIL_IMPORT_WARNED = False
_PIL_CODEC_WARNED: set = set()

# PIL's own default decompression-bomb ceiling (Image.MAX_IMAGE_PIXELS). We
# pin an explicit constant rather than reading the PIL attribute at call time
# so behavior doesn't silently change if something in the process mutates
# that global (some libraries do, to allow legitimately huge images).
_MAX_DECODE_PIXELS = 89_478_485

# A minimal, statically-embedded, known-good 1x1 image per format — used
# ONLY to probe whether this Pillow build can actually decode the format
# (as opposed to merely having the extension registered). These are NOT
# generated via PIL.Image(...).save() at call time: if the *encoder* were
# also missing/broken, generating the probe on the fly would confuse "can't
# create the probe" with "can't decode the format", and would burn a
# round-trip through PIL on every check. Regenerate with:
#   Image.new("RGB", (1, 1), (10, 20, 30)).save(buf, format=<FMT>)
_MIME_TO_PROBE_BYTES = {
    "image/png": bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c49444154789c63e0129103000068003d5408a3f70000000049454e44ae"
        "426082"
    ),
    "image/jpeg": None,  # populated lazily below (JPEG probes are large)
    "image/gif": bytes.fromhex(
        "474946383761010001008100000a141e0000000000000000002c000000000100"
        "010000080400010404003b"
    ),
    "image/bmp": bytes.fromhex(
        "424d3a0000000000000036000000280000000100000001000000010018000000"
        "000004000000c40e0000c40e000000000000000000001e140a00"
    ),
    "image/webp": bytes.fromhex(
        "524946462e0000005745425056503820220000007001009d012a01000100014"
        "0262594027401400000fefc378157f7d4e83e2be00000"
    ),
}


def _decoder_probe_bytes(mime: str) -> Optional[bytes]:
    probe = _MIME_TO_PROBE_BYTES.get(mime)
    if probe is not None:
        return probe
    if mime == "image/jpeg":
        # JPEG's minimal valid file is a few hundred bytes (Huffman tables
        # etc.) — not worth hand-transcribing as a hex literal, so this one
        # IS generated once and cached. A JPEG encoder missing but decoder
        # present (or vice versa) is not a realistic partial-build shape
        # (they ship as one libjpeg binding), so the "probe generation
        # itself fails" edge case this file's other entries avoid doesn't
        # meaningfully apply here.
        try:
            from PIL import Image as _PILImage
            import io as _io
            buf = _io.BytesIO()
            _PILImage.new("RGB", (1, 1), (10, 20, 30)).save(buf, format="JPEG")
            probe = buf.getvalue()
            _MIME_TO_PROBE_BYTES["image/jpeg"] = probe
        except Exception:
            return None
    return probe


def _decoder_available_for_mime(mime: str) -> bool:
    """True if this Pillow build can actually DECODE ``mime``, not just that
    it has the extension/plugin registered.

    Partial Pillow builds (e.g. no libwebp at compile time) can still import
    cleanly and register the ``.webp`` extension, yet fail to open a *valid*
    WEBP file — that's a missing/broken codec, not a corrupt image, and must
    not be conflated with :class:`CorruptImage`. We check this the same way
    the real validation does: open + load a small, known-good file of the
    format and see whether Pillow actually decodes it.
    """
    probe = _decoder_probe_bytes(mime)
    if probe is None:
        return True  # unknown mime, or probe unavailable — let the normal
        # open/verify path decide rather than silently skipping validation.
    try:
        from PIL import Image as _PILImage
        import io as _io
        with _PILImage.open(_io.BytesIO(probe)) as _img:
            _img.load()
        return True
    except Exception:
        return False


def verify_decodable_image(data: bytes, mime: str) -> Optional[str]:
    """Decode-validate raster bytes; return an error string, or None if OK.

    This is the single, reusable embed-time gate: ANY call site about to
    base64-embed image bytes into an outgoing message (conversation history,
    a native multimodal tool result, an ACP resource block, ...) should route
    through this before doing so. The magic-byte sniff used elsewhere only
    inspects the first few bytes, so a truncated file (cut mid-IDAT, mid-scan-
    line, etc.) that still has a valid signature + header sails through it
    untouched — the file *looks* like a PNG at a glance but the pixel stream
    never actually decodes. That's the byte-exact root cause paultaki traced
    in issue #69078: a browser_cdp ``Page.captureScreenshot`` screenshot got
    truncated on disk, and every existing gate (extension, magic bytes,
    declared media_type) happily passed it through to get base64-embedded
    into immutable conversation history, permanently poisoning the session.

    PIL gotcha: ``Image.verify()`` alone is NOT enough — it only checks
    structural integrity (headers/chunks) and can pass on a file that still
    fails to decode its actual pixel data. We must also ``.load()`` it, which
    forces the pixel stream to be read. ``verify()`` invalidates the Image
    object for further use, so we re-open the same bytes before ``.load()``.

    Also guards against decompression bombs: ``_MAX_INGEST_BYTES`` bounds the
    *compressed* size, but a small, well-formed PNG can still decode to
    hundreds of megabytes of pixel data (e.g. a highly-compressible solid-
    color image at absurd dimensions). We check the pixel count from the
    header (available without a full decode) against the same ceiling PIL
    itself defaults to (``Image.MAX_IMAGE_PIXELS``), and additionally promote
    PIL's own ``DecompressionBombWarning`` to an error scoped to this call so
    a bomb that slips past the explicit check is still caught before ``load()``
    can allocate the full pixel buffer.

    SVG has no pixel stream to decode here (it's rasterized to PNG later by
    the vision call sites), so it's skipped.

    Fail-open on a missing Pillow or a missing codec for this mime: Pillow is
    a soft dependency everywhere else in the vision path (see
    ``_image_exceeds_dimension``), and we never want a missing optional
    install — or a Pillow build without a particular codec — to break
    embedding of an otherwise-valid image.
    """
    if mime == "image/svg+xml":
        return None
    try:
        from PIL import Image as _PILImage, UnidentifiedImageError as _PILUnidentifiedImageError
    except ImportError:
        global _PIL_IMPORT_WARNED
        if not _PIL_IMPORT_WARNED:
            _PIL_IMPORT_WARNED = True
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "Pillow not installed — skipping decode validation of "
                "embedded images (magic-byte sniff still applies)")
        return None

    if not _decoder_available_for_mime(mime):
        if mime not in _PIL_CODEC_WARNED:
            _PIL_CODEC_WARNED.add(mime)
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "No Pillow codec registered for %s — skipping decode "
                "validation for this format (magic-byte sniff still "
                "applies)", mime)
        return None

    import io as _io

    # NOTE on thread safety: an earlier version of this function wrapped the
    # open/verify/load calls in `warnings.catch_warnings()` +
    # `simplefilter("error", ...)` to promote PIL's DecompressionBombWarning
    # to an exception. `warnings.catch_warnings()` mutates the PROCESS-GLOBAL
    # warning filter for the duration of the `with` block — under the
    # threaded gateway, a concurrent request's unrelated
    # DecompressionBombWarning could get promoted to an exception too (or
    # vice versa), a real cross-request bug caught in review. The explicit
    # pixel-count check below runs BEFORE `.load()` and is sufficient on its
    # own to reject an oversized image without ever calling `.load()` on it,
    # so the warning-promotion was redundant defense — removed rather than
    # made thread-safe via a module-level `Image.MAX_IMAGE_PIXELS` set,
    # since the explicit pre-check alone already closes the gap.
    try:
        with _PILImage.open(_io.BytesIO(data)) as _probe:
            _probe.verify()
        # verify() invalidates the Image object — re-open the same bytes and
        # force a full pixel decode, which is what actually catches a
        # truncated compressed stream that a structural verify() misses.
        with _PILImage.open(_io.BytesIO(data)) as _decoded:
            width, height = _decoded.size
            pixels = width * height
            if pixels > _MAX_DECODE_PIXELS:
                return (
                    f"{mime} image is {width}x{height} ({pixels:,} px), "
                    f"which exceeds the {_MAX_DECODE_PIXELS:,} px "
                    f"decode-safety ceiling (decompression-bomb guard)"
                )
            _decoded.load()
    except _PILImage.DecompressionBombError as exc:
        return f"{mime} image rejected as a decompression bomb: {exc}"
    except (_PILUnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return f"{mime} bytes failed to decode: {exc}"
    return None


# Internal alias kept for readability at call sites within this module.
_verify_decodable_image = verify_decodable_image


def _finalize(data: bytes, declared_mime: str, origin: str, src: str) -> ResolvedImage:
    """Intrinsic-correctness chokepoint: ingest byte cap + magic-byte sniff +
    decode validation.

    The cap here is the generous 50MB *ingest* budget, not the 20MB provider
    payload cap — a 20-50MB image must survive this step so the call site can
    resize it under the payload cap. See ``_MAX_INGEST_BYTES``.
    """
    from tools.vision_tools import _detect_image_mime_type_from_bytes

    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("image exceeds size limit", src=src, origin=origin)
    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is None:
        if b"<svg" in data[:4096].lower():
            # Pass SVG through — the vision call sites rasterize it to PNG
            # via _normalize_to_supported_image before embedding (providers
            # only ingest raster images).
            return ResolvedImage(data=data, mime="image/svg+xml", origin=origin)
        raise NotAnImage("source is not a recognized image", src=src, origin=origin)
    decode_error = verify_decodable_image(data, sniffed)
    if decode_error is not None:
        raise CorruptImage(
            f"image has a valid {sniffed} header but is unsafe to embed: "
            f"{decode_error}",
            src=src, origin=origin)
    return ResolvedImage(data=data, mime=sniffed, origin=origin)
