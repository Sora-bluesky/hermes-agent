"""Tests for tools/image_source.py — the unified vision image-source resolver.

Covers the delivery contract (data:/http/file/local/container source handling,
size cap, magic-byte sniff) AND the terminal-backend confinement security model
(GHSA-gpxw-6wxv-w3qq): under a non-local backend, host reads are confined to the
media caches and every other path is read inside the sandbox via exec-read.
"""

import base64
import importlib
import io
import os
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image


def _make_png(size=(4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(size=(4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(0, 255, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_with_corrupted_idat(size=(64, 64), keep_frac=0.6) -> bytes:
    """A PNG that survives PIL's ``verify()`` but fails ``load()``.

    This is the actual PIL gotcha the fix targets: ``verify()`` only checks
    PNG *container* structure (chunk lengths/CRCs, presence of IEND) — it
    does not run the zlib pixel decompression that ``load()`` does. Simply
    slicing bytes off the end of a PNG (as a naive "truncated file" fixture
    would) breaks the *container* itself (missing IEND / bad chunk length),
    which fails at ``verify()`` already and never exercises the ``load()``
    path this fix specifically added.

    To reproduce the real failure mode, we cut the compressed data *inside*
    the IDAT chunk short, then re-frame it as a well-formed chunk (declared
    length shrunk to match, CRC-32 recomputed over the now-shorter type+data)
    and leave every other chunk — crucially IEND — untouched. The container
    is entirely self-consistent (verify() passes), but the zlib stream ends
    mid-way through the pixel data, so decompression during load() raises.
    This mirrors a screenshot tool (e.g. browser_cdp's
    ``Page.captureScreenshot``) that got cut off mid-write: the file *looks*
    structurally fine but the pixels never fully arrived.
    """
    # Use random pixel data — a solid color compresses to a tiny IDAT that's
    # too short to meaningfully truncate.
    data = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    buf = io.BytesIO()
    data.save(buf, format="PNG")
    png = buf.getvalue()

    pos = 8  # past the 8-byte PNG signature
    out = [png[:8]]
    while pos < len(png):
        length = struct.unpack(">I", png[pos:pos + 4])[0]
        ctype = png[pos + 4:pos + 8]
        cdata = png[pos + 8:pos + 8 + length]
        crc = png[pos + 8 + length:pos + 12 + length]
        if ctype == b"IDAT":
            cdata = cdata[: max(4, int(len(cdata) * keep_frac))]
            crc = struct.pack(">I", zlib.crc32(ctype + cdata) & 0xFFFFFFFF)
        out.append(struct.pack(">I", len(cdata)) + ctype + cdata + crc)
        pos += 12 + length
    return b"".join(out)


# Real, fully-decodable images — _finalize now decode-validates (not just
# magic-byte sniffs), so fixtures must survive an actual PIL .verify()+.load().
PNG = _make_png()
JPEG = _make_jpeg()


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    importlib.reload(isrc)
    return isrc


class TestDataUrl:
    @pytest.mark.asyncio
    async def test_valid_data_url_resolves_to_bytes(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(PNG).decode()
        res = await isrc.resolve_image_source(
            f"data:image/png;base64,{b64}", isrc.ResolveContext())
        assert res.data == PNG
        assert res.mime == "image/png"
        assert res.origin == "data"

    @pytest.mark.asyncio
    async def test_non_image_data_url_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(isrc.NotAnImage):
            await isrc.resolve_image_source(
                f"data:text/plain;base64,{b64}", isrc.ResolveContext())


class TestLocalBackend:
    @pytest.mark.asyncio
    async def test_local_backend_reads_any_host_path(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "outside" / "pic.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_file_uri_scheme_stripped(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.jpg"
        img.write_bytes(JPEG)
        res = await isrc.resolve_image_source(f"file://{img}", isrc.ResolveContext())
        assert res.mime == "image/jpeg"

    @pytest.mark.asyncio
    async def test_bare_relative_path_resolves(self, tmp_path, monkeypatch):
        """A cwd-relative bare filename ('pic.png') is a valid local source —
        main accepted it; the resolver must not regress it (PR review)."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        monkeypatch.chdir(tmp_path)
        res = await isrc.resolve_image_source("pic.png", isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_unknown_url_scheme_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        with pytest.raises(isrc.UnsupportedScheme):
            await isrc.resolve_image_source(
                "ftp://example.com/pic.png", isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_svg_passes_through_for_rasterization(self, tmp_path, monkeypatch):
        """SVG has no raster magic bytes but is passed through with mime
        image/svg+xml so the vision call sites can rasterize it to PNG."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        svg = tmp_path / "art.svg"
        svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        svg.write_bytes(svg_bytes)
        res = await isrc.resolve_image_source(str(svg), isrc.ResolveContext())
        assert res.mime == "image/svg+xml"
        assert res.data == svg_bytes


class TestNonLocalBackendConfinement:
    """The security model: under a sandbox backend, host reads are confined to
    the media caches; every other path is read inside the sandbox."""

    @pytest.mark.asyncio
    async def test_media_cache_path_host_read(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        cached = home / "cache" / "images" / "inbound.png"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(PNG)
        # No sandbox env needed — a cache path is host-read directly.
        res = await isrc.resolve_image_source(str(cached), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_host_secret_outside_cache_routes_to_sandbox_not_host(self, tmp_path, monkeypatch):
        """A non-cache host path (e.g. /etc/passwd) must NOT be host-read — it
        routes to the in-sandbox exec-read, which reads the CONTAINER's file."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        # A real host file outside the caches, holding a "secret".
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY-DO-NOT-LEAK")

        # Fake sandbox env: its exec-read returns a *different* (container) image,
        # proving we read the container filesystem, not the host secret.
        container_png_b64 = base64.b64encode(PNG).decode()
        calls = {}

        def fake_execute(cmd, **kw):
            calls["cmd"] = cmd
            return {"returncode": 0, "output": container_png_b64}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            res = await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))

        # Read came from the sandbox exec-read, returning the container image —
        # the host secret bytes never appear.
        assert res.origin == "container"
        assert res.data == PNG
        assert b"HOST-PRIVATE-KEY" not in res.data
        assert "head -c" in calls["cmd"] and "< " in calls["cmd"]  # bounded, redirect-safe form

    @pytest.mark.asyncio
    async def test_non_cache_path_fails_closed_without_sandbox(self, tmp_path, monkeypatch):
        """No active sandbox env -> refuse rather than fall back to a host read."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY")

        with patch("tools.image_source._get_active_env", return_value=None):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))

    @pytest.mark.asyncio
    async def test_symlink_in_cache_pointing_outside_is_not_host_read(self, tmp_path, monkeypatch):
        """A symlink planted inside a cache dir that points at a host secret must
        not be host-read (resolve() escapes the cache) — it routes to sandbox."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        secret = tmp_path / "outside" / "id_rsa"
        secret.parent.mkdir(parents=True)
        secret.write_bytes(b"HOST-PRIVATE-KEY")
        cache_dir = home / "cache" / "images"
        cache_dir.mkdir(parents=True)
        link = cache_dir / "sneaky.png"
        try:
            link.symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported")

        # Fails closed (no sandbox) rather than host-reading the symlink target.
        with patch("tools.image_source._get_active_env", return_value=None):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(str(link), isrc.ResolveContext(task_id="t1"))


class TestExecReadSafety:
    @pytest.mark.asyncio
    async def test_exec_read_is_bounded_and_redirect_safe(self, tmp_path, monkeypatch):
        """Leading-dash paths go through an input redirect (no argv exposure)
        and the read is size-bounded via head -c."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        captured = {}

        def fake_execute(cmd, **kw):
            captured["cmd"] = cmd
            return {"returncode": 0, "output": base64.b64encode(PNG).decode()}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            await isrc.resolve_image_source(
                "/workspace/-i-etc-shadow.png", isrc.ResolveContext(task_id="t1"))
        assert f"head -c {isrc._MAX_INGEST_BYTES + 1} < " in captured["cmd"]
        assert "'-i-etc-shadow.png'" in captured["cmd"] or "-i-etc-shadow.png" in captured["cmd"]

    @pytest.mark.asyncio
    async def test_exec_read_over_cap_rejected(self, tmp_path, monkeypatch):
        """A sandbox file larger than the ingest cap is rejected, not embedded."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        # head -c returns cap+1 bytes for an oversized file.
        over = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * (isrc._MAX_INGEST_BYTES - 7)).decode()

        def fake_execute(cmd, **kw):
            return {"returncode": 0, "output": over}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            with pytest.raises(isrc.SourceTooLarge):
                await isrc.resolve_image_source(
                    "/workspace/huge.png", isrc.ResolveContext(task_id="t1"))

    @pytest.mark.asyncio
    async def test_exec_read_nonzero_returncode_raises(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        def fake_execute(cmd, **kw):
            return {"returncode": 1, "output": ""}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(
                    "/workspace/nope.png", isrc.ResolveContext(task_id="t1"))


class TestSvgNormalization:
    """SVG resolves end-to-end: the resolver passes it through as
    image/svg+xml and the vision call sites rasterize it to PNG via
    _normalize_to_supported_image (PR #52688, folded in)."""

    @pytest.mark.asyncio
    async def test_svg_rasterized_when_converter_available(self, tmp_path, monkeypatch):
        from tools import vision_tools as vt
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        svg = tmp_path / "art.svg"
        svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>')

        def fake_rasterize(svg_path, out_path):
            out_path.write_bytes(PNG)
            return True

        with patch.object(vt, "_rasterize_svg_to_png", side_effect=fake_rasterize):
            res = await isrc.resolve_image_source(str(svg), isrc.ResolveContext())
            assert res.mime == "image/svg+xml"
            path, mime, err = vt._normalize_to_supported_image(svg, "image/svg+xml")
        assert err is None
        assert mime == "image/png"
        assert path.read_bytes() == PNG
        path.unlink()

    def test_svg_actionable_error_when_no_converter(self, tmp_path, monkeypatch):
        from tools import vision_tools as vt
        _reload(monkeypatch, tmp_path / "hermes")
        svg = tmp_path / "art.svg"
        svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
        with patch.object(vt, "_rasterize_svg_to_png", return_value=False):
            path, mime, err = vt._normalize_to_supported_image(svg, "image/svg+xml")
        assert path is None
        assert "rasterizer" in err


class TestDecodeValidation:
    """Issue #69078 root cause: a truncated download can keep a valid magic
    byte signature + header while the pixel stream never actually decodes
    (e.g. a browser_cdp Page.captureScreenshot cut off mid-IDAT). The
    magic-byte sniff alone lets it through; _finalize must also decode-verify."""

    @pytest.mark.asyncio
    async def test_truncated_png_rejected(self, tmp_path, monkeypatch):
        """Reporter's exact scenario: PNG container is structurally valid
        (verify() passes) but the compressed pixel stream inside IDAT was
        cut short (load() fails) — the actual PIL gotcha, not just a
        naively sliced-off file (which would fail at verify() already and
        never exercise the load()-only code path this fix added)."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        truncated = _make_png_with_corrupted_idat()
        img = tmp_path / "truncated.png"
        img.write_bytes(truncated)
        with pytest.raises(isrc.CorruptImage):
            await isrc.resolve_image_source(str(img), isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_truncated_jpeg_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        full = _make_jpeg(size=(64, 64))
        truncated = full[: len(full) // 2]
        img = tmp_path / "truncated.jpg"
        img.write_bytes(truncated)
        with pytest.raises(isrc.CorruptImage):
            await isrc.resolve_image_source(str(img), isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_valid_png_survives_finalize(self, tmp_path, monkeypatch):
        """Guard against rejecting legitimate content: a normal PIL-generated
        PNG must pass through _finalize unmodified."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "ok.png"
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == PNG
        assert res.mime == "image/png"

    @pytest.mark.asyncio
    async def test_valid_jpeg_survives_finalize(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "ok.jpg"
        img.write_bytes(JPEG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == JPEG
        assert res.mime == "image/jpeg"

    @pytest.mark.asyncio
    async def test_pil_unavailable_fails_open(self, tmp_path, monkeypatch):
        """Missing Pillow must never break embedding of otherwise-valid,
        magic-sniffed bytes — decode validation is a soft-dependency add-on,
        mirroring the fail-open pattern in _image_exceeds_dimension."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "ok.png"
        img.write_bytes(PNG)

        real_import = __import__

        def _blocked_import(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("simulated missing Pillow")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == PNG
        assert res.mime == "image/png"

    @pytest.mark.asyncio
    async def test_truncated_bytes_rejected_via_data_url(self, tmp_path, monkeypatch):
        """Same guarantee on the data: URL route, not just local files."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        truncated = _make_png_with_corrupted_idat()
        b64 = base64.b64encode(truncated).decode()
        with pytest.raises(isrc.CorruptImage):
            await isrc.resolve_image_source(
                f"data:image/png;base64,{b64}", isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_decompression_bomb_rejected(self, tmp_path, monkeypatch):
        """A small, well-formed file that decodes to an enormous pixel count
        must be rejected before .load() allocates the full buffer — a
        highly-compressible image can be tiny on disk yet decode to
        hundreds of megapixels.

        Uses mode "1" (1 bit/pixel, packed) rather than "RGB" for the
        100-megapixel fixture: RGB would allocate ~286 MiB of raw pixel
        buffer just to *build* the test fixture (before the guard is even
        exercised), where "1" needs ~12 MB. The guard itself is dtype-
        agnostic — it rejects on width*height alone, before any pixel
        buffer is allocated — so the bit depth used to construct the
        fixture doesn't change what's being verified.
        """
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        w = h = 10_000  # 100,000,000 px > _MAX_DECODE_PIXELS (89,478,485)
        buf = io.BytesIO()
        Image.new("1", (w, h)).save(buf, format="PNG")
        bomb = buf.getvalue()
        img = tmp_path / "bomb.png"
        img.write_bytes(bomb)
        with pytest.raises(isrc.CorruptImage, match="decode-safety|decompression bomb"):
            await isrc.resolve_image_source(str(img), isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_missing_codec_fails_open(self, tmp_path, monkeypatch):
        """A Pillow build without a decoder for a given format (e.g. no
        libwebp) must not be treated as a corrupt file — the magic-byte
        sniff already established it, decode validation just isn't
        possible with this Pillow build, so it fails open like a missing
        Pillow install does.

        Exercises the REAL ``_decoder_available_for_mime`` code path (not a
        patched return value): its probe bytes for image/webp are swapped
        for garbage, so the function's own open+load attempt genuinely
        fails and it computes False on its own — indistinguishable, from
        the function's perspective, from a Pillow build that lacks the
        libwebp codec. The webp file bytes themselves are the exact literal
        ``tools.image_source`` uses as its own probe (a real, valid WEBP)
        rather than a PIL-encoded one, so this doesn't need a working WEBP
        encoder in this environment either.
        """
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        webp_bytes = isrc._MIME_TO_PROBE_BYTES["image/webp"]
        img = tmp_path / "ok.webp"
        img.write_bytes(webp_bytes)

        with patch.dict(isrc._MIME_TO_PROBE_BYTES, {"image/webp": b"not actually a webp file"}):
            res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == webp_bytes
        assert res.mime == "image/webp"

    def test_does_not_mutate_process_global_warning_filters(self, tmp_path, monkeypatch):
        """Regression for a thread-safety bug caught in review: an earlier
        version wrapped the decode in ``warnings.catch_warnings()`` +
        ``simplefilter("error", ...)`` to promote PIL's
        DecompressionBombWarning to an exception. ``catch_warnings()``
        mutates the PROCESS-GLOBAL warning filter list for the duration of
        the ``with`` block — under the threaded gateway, a concurrent
        request's unrelated warning could get promoted to an exception too.
        The explicit pixel-count pre-check (already run before ``.load()``)
        is sufficient on its own, so the fix removed the filter mutation
        entirely rather than trying to make it thread-safe.

        Verified structurally: ``warnings.simplefilter`` must not be called
        at all during validation of either a normal image OR a bomb-sized
        one (the path most likely to reach for a warning filter).
        """
        isrc = _reload(monkeypatch, tmp_path / "hermes")

        with patch("warnings.simplefilter", side_effect=AssertionError(
                "verify_decodable_image must not touch the process-global "
                "warning filter — see thread-safety note in the function")):
            # Normal image: must not touch warnings.simplefilter.
            assert isrc.verify_decodable_image(PNG, "image/png") is None

            # Bomb-sized image: must be rejected via the explicit
            # pixel-count pre-check alone, still without touching
            # warnings.simplefilter.
            bomb_buf = io.BytesIO()
            Image.new("1", (10_000, 10_000)).save(bomb_buf, format="PNG")
            err = isrc.verify_decodable_image(bomb_buf.getvalue(), "image/png")
            assert err is not None and "decode-safety" in err
