import os
import sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

from marker.settings import settings
from marker.config.printer import CustomClickPrinter
from streamlit.runtime.uploaded_file_manager import ( UploadedFile, UploadedFileRec )

import base64
import io
import subprocess
import json
import re
import string
import tempfile
from typing import Any, Dict
import click

import pypdfium2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.schema import BlockTypes

COLORS = [
    "#4e79a7",
    "#f28e2c",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab"
]

with open(
    os.path.join(os.path.dirname(__file__), "streamlit_app_blocks_viz.html"), encoding="utf-8"
) as f:
    BLOCKS_VIZ_TMPL = string.Template(f.read())


@st.cache_data()
def parse_args():
    # Use to grab common cli options
    @ConfigParser.common_options
    def options_func():
        pass

    def extract_click_params(decorated_function):
        if hasattr(decorated_function, '__click_params__'):
            return decorated_function.__click_params__
        return []

    cmd = CustomClickPrinter("Marker app.")
    extracted_params = extract_click_params(options_func)
    cmd.params.extend(extracted_params)
    ctx = click.Context(cmd)
    try:
        cmd_args = sys.argv[1:]
        cmd.parse_args(ctx, cmd_args)
        return ctx.params
    except click.exceptions.ClickException as e:
        return {"error": str(e)}

@st.cache_resource()
def load_models():
    return create_model_dict()


def convert_pdf(fname: str, config_parser: ConfigParser) -> (str, Dict[str, Any], dict):
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    return converter(fname)


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=settings.OUTPUT_IMAGE_FORMAT)
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f'<img src="data:image/{settings.OUTPUT_IMAGE_FORMAT.lower()};base64,{encoded}" alt="{img_alt}" style="max-width: 100%;">'
    return img_html


def markdown_insert_images(markdown, images):
    image_tags = re.findall(r'(!\[(?P<image_title>[^\]]*)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)

    for image in image_tags:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if image_path in images:
            markdown = markdown.replace(image_markdown, img_to_html(images[image_path], image_alt))
    return markdown


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        page = doc[page_num]
        png_image = page.render(
            scale=dpi / 72,
        ).to_pil().convert("RGB")
    else:
        png_image = Image.open(pdf_file).convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file: UploadedFile):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        return len(doc) - 1
    else:
        return 1


def pillow_image_to_base64_string(img: Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def block_display(image: Image, blocks: dict | None = None, dpi=96):
    if blocks is None:
        blocks = {}

    image_data_url = (
        'data:image/jpeg;base64,' + pillow_image_to_base64_string(image)
    )

    template_values = {
        "image_data_url": image_data_url,
        "image_width": image.width, "image_height": image.height,
        "blocks_json": blocks, "colors_json": json.dumps(COLORS),
        "block_types_json": json.dumps({
            bt.name: i for i, bt in enumerate(BlockTypes)
        })
    }
    return components.html(
        BLOCKS_VIZ_TMPL.substitute(**template_values),
        height=image.height
    )


st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

model_dict = load_models()
cli_options = parse_args()


st.markdown("""
# Marker Demo

This app will let you try marker, a PDF or image -> Markdown, HTML, JSON converter. It works with any language, and extracts images, tables, equations, etc.

Find the project [here](https://github.com/VikParuchuri/marker).
""")

in_file: UploadedFile = st.sidebar.file_uploader("PDF, document, or image file:", type=["pdf", "png", "jpg", "jpeg", "gif", "pptx", "docx", "xlsx", "html", "epub"])

if in_file is None:
    st.stop()

# by szczynk
# Session state initialization for processed PDF
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = None


# Track file changes and reset processing
current_file_id = in_file.file_id
if st.session_state.processed_pdf is not None and st.session_state.processed_pdf.file_id != f"processed_{current_file_id}":
    st.session_state.processed_pdf = None

filetype = in_file.type

# by szczynk
# PDF processing controls
with st.sidebar.expander("PDF Processing Options") as processing_opt:
    trim_size = st.text_input("Trim size (e.g., '0mm 36mm 0mm 33mm')", value="0mm 36mm 0mm 33mm")
    page_sel = st.text_input("Page selection (e.g., '1,3-')", value="1,3-")
    process_pdf = st.button("Apply PDF Processing")

# Process PDF with pdfjam
if process_pdf:
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_out:
        
        tmp_in.write(in_file.getvalue())
        tmp_in.seek(0)
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name

    # Build command dynamically based on user input
    cmd = ["pdfjam", "--fitpaper", "true"]
    
    # Add trim if specified
    if trim_size.strip():
        cmd.extend(['--trim', trim_size.strip()])
    
    cmd.append(tmp_in_path)
    
    # Add page selection if specified
    if page_sel.strip():
        cmd.extend([page_sel.strip()])
    
    cmd.extend(['--outfile', tmp_out_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        with open(tmp_out_path, 'rb') as f:
            processed_data = io.BytesIO(f.read())
            record = UploadedFileRec(
                file_id=f"processed_{in_file.file_id}",
                name=f"processed_{in_file.name}",
                type="application/pdf",
                data=processed_data.getvalue(),
            )

            processed_file = UploadedFile(record, None)
            st.session_state.processed_pdf = processed_file

        st.success("PDF processed successfully!")
    else:
        st.error(f"PDF processing failed: {result.stderr}")

    os.remove(tmp_in_path)
    os.remove(tmp_out_path)
    st.rerun()


# Use processed PDF if available
if st.session_state.processed_pdf is not None:
    in_file = st.session_state.processed_pdf

with col1:
    page_count = page_count(in_file)
    page_number = st.number_input(f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count)
    pil_image = get_page_image(in_file, page_number)
    image_placeholder = st.empty()

    with image_placeholder:
        block_display(pil_image)


page_range = st.sidebar.text_input("Page range to parse, comma separated like 0,5-10,20", value=f"{page_number}-{page_count}")
output_format = st.sidebar.selectbox("Output format", ["markdown", "json", "html"], index=0)
run_marker = st.sidebar.button("Run Marker")

with st.sidebar.expander("Advanced Options") as advanced_opt:
    use_llm = st.checkbox("Use LLM", help="Use LLM for higher quality processing", value=False)
    
    # by szczynk
    if use_llm:
        llm_service = st.sidebar.selectbox(
            "LLM Service",
            ["marker.services.ollama.OllamaService", "marker.services.gemini.GoogleGeminiService"],
            index=0
        )

        # If OllamaService is selected, show additional inputs
        if llm_service == "marker.services.ollama.OllamaService":
            ollama_base_url = st.sidebar.text_input("Ollama Base URL", value="http://172.17.0.1:11434")
            ollama_model = st.sidebar.text_input("Ollama Model", value="llama3.2-vision")

    show_blocks = st.checkbox("Show Blocks", help="Display detected blocks, only when output is JSON", value=False, disabled=output_format != "json")
    force_ocr = st.checkbox("Force OCR", help="Force OCR on all pages", value=False)
    strip_existing_ocr = st.checkbox("Strip existing OCR", help="Strip existing OCR text from the PDF and re-OCR.", value=False)
    debug = st.checkbox("Debug", help="Show debug information", value=False)

if not run_marker:
    st.stop()

# Run Marker
with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb+") as temp_pdf:
    temp_pdf.write(in_file.getvalue())
    temp_pdf.seek(0)
    filename = temp_pdf.name
    cli_options = {
        "output_format": output_format,
        "page_range": page_range,
        "force_ocr": force_ocr,
        "debug": debug,
        "output_dir": settings.DEBUG_DATA_FOLDER if debug else None,
        "use_llm": use_llm,
        "strip_existing_ocr": strip_existing_ocr
    }
    
    # by szczynk
    if use_llm:
        cli_options["llm_service"] = llm_service

        if llm_service == "marker.services.ollama.OllamaService":
            cli_options["ollama_base_url"] = ollama_base_url
            cli_options["ollama_model"] = ollama_model

    config_parser = ConfigParser(cli_options)
    rendered = convert_pdf(
        filename,
        config_parser
    )
    page_range = config_parser.generate_config_dict()["page_range"]
    first_page = page_range[0] if page_range else 0

text, ext, images = text_from_rendered(rendered)
with col2:
    # by szczynk
    # Streamlit's download button to download the file
    file_name = f"{in_file.name}.{ext}"
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col3:
        pass
    with col2 :
        center_button = st.download_button(
            label="Download File",
            data=text,
            file_name=file_name,
            mime="text/plain" if output_format != "json" else "application/json",
        )

    # Render the result
    if output_format == "markdown":
        text = markdown_insert_images(text, images)
        st.markdown(text, unsafe_allow_html=True)
    elif output_format == "json":
        st.json(text)
    elif output_format == "html":
        st.html(text)

if output_format == "json" and show_blocks:
    with image_placeholder:
        block_display(pil_image, text)

if debug:
    with col1:
        debug_data_path = rendered.metadata.get("debug_data_path")
        if debug_data_path:
            pdf_image_path = os.path.join(debug_data_path, f"pdf_page_{first_page}.png")
            img = Image.open(pdf_image_path)
            st.image(img, caption="PDF debug image", use_container_width=True)
            layout_image_path = os.path.join(debug_data_path, f"layout_page_{first_page}.png")
            img = Image.open(layout_image_path)
            st.image(img, caption="Layout debug image", use_container_width=True)
        st.write("Raw output:")
        st.code(text, language=output_format)
