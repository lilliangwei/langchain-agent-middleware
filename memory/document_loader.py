"""
文档加载器 + 文本切分器 + 扫描版 PDF OCR（三合一本文件）。

DocumentLoader：按文件扩展名分发到对应解析器，统一返回纯文本。
    .txt / .md   ：直接读取
    .pdf         ：pypdf（延迟导入）；扫描版/图片型 PDF 自动 fallback 多模态 OCR
    .docx        ：python-docx（import docx，延迟导入）
    .xlsx / .xls ：openpyxl（延迟导入，data_only + read_only 加速）
    .csv         ：标准库 csv
旧版 .doc（二进制 Word）python-docx 不支持，会明确提示转 .docx。

TextChunker：递归字符切分，适配中英文混合场景（原 text_chunker.py 并入）。

OCR（原 ocr.py 并入）：基于 PyMuPDF 渲染 + Ollama 多模态视觉模型识别。
    PDF 每页 → fitz 渲染成 JPEG → base64 data url
           → LangChain ChatOpenAI 多模态 invoke → 视觉模型返回文字
OCR 降级链（三层，均不阻断）：
    1. PyMuPDF 未安装 → ImportError → 返回 pypdf 结果
    2. 视觉模型未配置/调用异常 → 返回 pypdf 结果
    3. 单页 OCR 失败 → 空字符串占位，不影响其他页
"""
import base64
import csv
import os
from typing import List, Optional

from utils.logger import logger

OCR_PROMPT = (
    "你是一个精准的 OCR 助手。请识别图片中的全部文字，要求：\n"
    "1. 保留原文的段落、标题、列表结构，输出为纯文本 Markdown。\n"
    "2. 中文、英文、数字、符号均如实识别，不要翻译、不要总结、不要补充。\n"
    "3. 表格用 Markdown 表格语法还原。\n"
    "4. 仅输出识别到的文字内容，不要任何解释或前后缀。"
)


# ==================== 扫描版 PDF OCR（原 ocr.py 并入） ====================
def pdf_pages_to_base64(
    path: str, dpi: Optional[int] = None, fmt: Optional[str] = None
) -> List[str]:
    """
    用 PyMuPDF 渲染每页为图片，返回 base64 data url 列表。

    fmt: jpeg（默认，小）或 png（无损，大）。
    """
    import fitz  # PyMuPDF，延迟导入

    dpi = dpi or int(os.getenv("OCR_DPI", "150"))
    fmt = fmt or os.getenv("OCR_IMAGE_FORMAT", "jpeg")
    if fmt not in ("jpeg", "png"):
        fmt = "jpeg"

    doc = fitz.open(path)
    results: List[str] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            if fmt == "jpeg":
                img_bytes = pix.tobytes("jpeg", quality=80)
            else:
                img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("ascii")
            mime = "image/jpeg" if fmt == "jpeg" else "image/png"
            results.append(f"data:{mime};base64,{b64}")
    finally:
        doc.close()
    return results


def _extract_one_page(vision_client, image_data_url: str, timeout: int) -> str:
    """对单页图片调多模态视觉模型识别文字（LangChain ChatOpenAI 多模态消息）"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=OCR_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "请识别这张图片中的全部文字。"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]),
    ]
    resp = vision_client.invoke(messages, timeout=timeout)
    return resp.content or ""


def ocr_pdf(
    path: str,
    vision_client,
    dpi: Optional[int] = None,
    timeout: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> List[str]:
    """
    对整本 PDF 做 OCR，返回每页文本的 List[str]（按下标对应页码，0-based）。

    单页失败对应位置为空字符串，不阻断整本。
    max_pages 安全阀，防止超大 PDF 拖垮 OCR。
    """
    dpi = dpi or int(os.getenv("OCR_DPI", "150"))
    timeout = timeout or int(os.getenv("OCR_TIMEOUT", "120"))
    max_pages = max_pages or int(os.getenv("OCR_MAX_PAGES", "20"))

    images = pdf_pages_to_base64(path, dpi=dpi)
    total = len(images)
    if max_pages and total > max_pages:
        logger.warning(
            f"[OCR] PDF 页数 {total} 超过上限 {max_pages}，仅 OCR 前 {max_pages} 页"
        )
        images = images[:max_pages]

    pages_text: List[str] = []
    for i, img in enumerate(images, 1):
        try:
            txt = _extract_one_page(vision_client, img, timeout)
            pages_text.append(txt)
            logger.info(f"[OCR] 第 {i}/{len(images)} 页识别完成，{len(txt)} 字符")
        except Exception as e:
            logger.error(f"[OCR] 第 {i} 页识别失败：{e}")
            pages_text.append("")  # 该页空，不阻断后续
    return pages_text


# ==================== 文本切分器（基于 LangChain RecursiveCharacterTextSplitter） ====================
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    对 LangChain RecursiveCharacterTextSplitter 的薄封装，
    保持原 split(text, chunk_size, overlap) 接口，下游（rag_agent / main.py）零改动。

    相比原自研实现的改进：
    - overlap 由 splitter 原生实现（相邻块共享尾部片段），不再硬拼前块尾巴破坏块开头语义
    - keep_separator=True 保留分隔符，块边界更完整
    - 超长无分隔段等边缘情况交给 LangChain 成熟逻辑兜底
    """
    # 中文 + 英文标点都覆盖；末尾 "" 表示按字符兜底切分
    SEPARATORS = [
        "\n\n", "\n", "。", "！", "？", "!", "?", "，", ",", "；", ";", " ", "",
    ]

    def split(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """递归切分文本，返回块列表。空文本返回空列表。"""
        if not text or not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            separators=self.SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            keep_separator=True,
            strip_whitespace=True,
        )
        return [c for c in splitter.split_text(text) if c.strip()]


# ==================== 文档加载器 ====================
class DocumentLoader:
    SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls", ".csv"}

    def __init__(
        self,
        pool=None,
        enable_ocr: Optional[bool] = None,
    ):
        """
        pool：注入 LLMPool 后 _load_pdf 才能经 pool.get_vision_client() 调多模态 OCR。
        enable_ocr：默认读 .env 的 ENABLE_OCR；未开启或视觉客户端不可用时退化为纯 pypdf。
        """
        self.vision_client = pool.get_vision_client() if pool else None
        self.enable_ocr = (
            enable_ocr
            if enable_ocr is not None
            else os.getenv("ENABLE_OCR", "false").lower() == "true"
        )

    def load(self, file_path: str) -> str:
        """加载文件并返回纯文本。不支持时抛 ValueError。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED:
            raise ValueError(
                f"不支持的文件类型: {ext}，支持: {sorted(self.SUPPORTED)}"
            )

        handlers = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_doc_legacy,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".csv": self._load_csv,
        }
        try:
            text = handlers[ext](file_path)
        except ImportError as e:
            raise ValueError(
                f"解析 {ext} 缺少依赖库，请 pip install 对应库: {e}"
            ) from e

        if not text or not text.strip():
            logger.warning(f"文件 {file_path} 解析后内容为空")
        return text

    # ---------- 文本类 ----------
    def _load_text(self, path: str) -> str:
        # 多编码兜底：utf-8 → gbk → utf-8-ignore
        for enc in ("utf-8", "gbk", "utf-8"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # 最终兜底：忽略非法字符
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---------- PDF ----------
    def _load_pdf(self, path: str) -> str:
        from pypdf import PdfReader  # 延迟导入，未安装时不影响其他格式

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        raw_text = "\n\n".join(pages)

        # 判定是否需要 OCR：整篇为空 或 任一页为空（扫描版/部分扫描）
        need_ocr = self.enable_ocr and self.vision_client and (
            not raw_text.strip() or any(not p.strip() for p in pages)
        )
        if not need_ocr:
            return raw_text

        # OCR 逻辑在本文件内，fitz 延迟导入，未安装时抛 ImportError 降级为 pypdf
        try:
            ocr_pages = ocr_pdf(path=path, vision_client=self.vision_client)
        except ImportError as e:
            logger.warning(
                f"[PDF] OCR 依赖缺失（PyMuPDF 未安装），降级使用 pypdf 结果：{e}"
            )
            return raw_text
        except Exception as e:
            logger.warning(f"[PDF] OCR 失败，降级使用 pypdf 结果：{e}")
            return raw_text

        # 合并策略：全扫描直接用 OCR；部分扫描逐页取 pypdf 文字或 OCR 文字
        if not raw_text.strip():
            return "\n\n".join(ocr_pages)

        merged = []
        for i, p in enumerate(pages):
            if p.strip():
                merged.append(p)
            elif i < len(ocr_pages) and ocr_pages[i].strip():
                merged.append(ocr_pages[i])
        return "\n\n".join(merged)

    # ---------- Word ----------
    def _load_docx(self, path: str) -> str:
        import docx  # python-docx，延迟导入

        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _load_doc_legacy(self, path: str) -> str:
        # 旧版 .doc 二进制 Word，python-docx 不支持
        raise ValueError(
            "旧版 .doc 格式不支持，请用 Word 另存为 .docx 后再上传"
        )

    # ---------- Excel ----------
    def _load_excel(self, path: str) -> str:
        import openpyxl  # 延迟导入

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        chunks: List[str] = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    chunks.append(" | ".join(cells))
        return "\n".join(chunks)

    # ---------- CSV ----------
    def _load_csv(self, path: str) -> str:
        # 多编码兜底
        for enc in ("utf-8", "gbk", "utf-8"):
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    return "\n".join(" | ".join(r) for r in reader)
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            return "\n".join(" | ".join(r) for r in reader)
