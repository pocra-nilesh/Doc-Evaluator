import os
import re
import json
import tempfile

import cv2
import numpy as np
import streamlit as st

from pdf2image import convert_from_path
import c2pa


# ================================================================
# DOCUMENT VALIDATOR
# ================================================================

class DocumentValidator:

    def __init__(
        self,
        ela_quality=95,
        ela_scale=25,
        block_size=32,
        variance_threshold=10.0,
        min_content_ratio=0.05,
        min_edge_ratio=0.005,
        ela_mean_threshold=10.0,
        ela_std_threshold=16.5
    ):

        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.block_size = block_size
        self.variance_threshold = variance_threshold
        self.min_content_ratio = min_content_ratio
        self.min_edge_ratio = min_edge_ratio

        # ELA thresholds
        self.ela_mean_threshold = ela_mean_threshold
        self.ela_std_threshold = ela_std_threshold


    # ============================================================
    # C2PA / CONTENT CREDENTIALS
    # ============================================================

    def _check_content_credentials(self, file_path):

        """
        Checks the ORIGINAL uploaded file for C2PA Content Credentials.

        This does NOT attempt to guess AI generation from pixels.
        """

        result = {
            "status": "NO_CREDENTIALS",
            "has_credentials": False,
            "valid": False,
            "ai_generated": False,
            "ai_edited": False,
            "issuer": None,
            "actions": [],
            "digital_source_types": [],
            "error": None
        }

        try:

            reader = c2pa.Reader(file_path)

            manifest_json = reader.json()

            if not manifest_json:
                return result

            if isinstance(manifest_json, str):
                manifest_store = json.loads(manifest_json)
            else:
                manifest_store = manifest_json

            result["has_credentials"] = True

            active_manifest_id = manifest_store.get(
                "active_manifest"
            )

            manifests = manifest_store.get(
                "manifests",
                {}
            )

            if not active_manifest_id:

                result["status"] = "INVALID_CREDENTIAL"

                result["error"] = (
                    "C2PA manifest exists but no active "
                    "manifest was found."
                )

                return result

            active_manifest = manifests.get(
                active_manifest_id
            )

            if not active_manifest:

                result["status"] = "INVALID_CREDENTIAL"

                result["error"] = (
                    "Active C2PA manifest could not be located."
                )

                return result

            # ----------------------------------------------------
            # Issuer
            # ----------------------------------------------------

            result["issuer"] = active_manifest.get(
                "claim_generator"
            )

            if not result["issuer"]:

                result["issuer"] = active_manifest.get(
                    "claim_generator_info"
                )

            # ----------------------------------------------------
            # Assertions
            # ----------------------------------------------------

            assertions = active_manifest.get(
                "assertions",
                []
            )

            for assertion in assertions:

                label = assertion.get(
                    "label",
                    ""
                )

                data = assertion.get(
                    "data",
                    {}
                )

                if label in (
                    "c2pa.actions",
                    "c2pa.actions.v2"
                ):

                    actions = data.get(
                        "actions",
                        []
                    )

                    for action in actions:

                        action_name = action.get(
                            "action"
                        )

                        action_record = {
                            "action": action_name
                        }

                        if "description" in action:

                            action_record[
                                "description"
                            ] = action["description"]

                        if "softwareAgent" in action:

                            action_record[
                                "software_agent"
                            ] = action["softwareAgent"]

                        digital_source_type = action.get(
                            "digitalSourceType"
                        )

                        if digital_source_type:

                            action_record[
                                "digital_source_type"
                            ] = digital_source_type

                            result[
                                "digital_source_types"
                            ].append(
                                digital_source_type
                            )

                        result["actions"].append(
                            action_record
                        )

                        # ----------------------------------------
                        # AI-generated media
                        # ----------------------------------------

                        if digital_source_type:

                            normalized = (
                                str(digital_source_type)
                                .lower()
                                .replace("_", "")
                                .replace("-", "")
                            )

                            if (
                                "trainedalgorithmicmedia"
                                in normalized
                            ):

                                result[
                                    "ai_generated"
                                ] = True

                        # ----------------------------------------
                        # AI-generated material in composite
                        # ----------------------------------------

                        if digital_source_type:

                            normalized = (
                                str(digital_source_type)
                                .lower()
                                .replace("_", "")
                                .replace("-", "")
                            )

                            if (
                                "compositewithtrainedalgorithmicmedia"
                                in normalized
                            ):

                                result[
                                    "ai_edited"
                                ] = True

            # ----------------------------------------------------
            # Final status
            # ----------------------------------------------------

            result["valid"] = True

            if result["ai_generated"]:

                result["status"] = (
                    "AI_GENERATED_CREDENTIAL"
                )

            elif result["ai_edited"]:

                result["status"] = (
                    "AI_EDITED_CREDENTIAL"
                )

            else:

                result["status"] = (
                    "CREDENTIAL_VERIFIED"
                )

            return result

        except Exception as e:

            result["error"] = str(e)

            result["status"] = "NO_CREDENTIALS"

            return result


    # ============================================================
    # PDF STRUCTURE / MULTIPLE %%EOF CHECK
    # ============================================================

    def _check_pdf_eof_markers(self, file_path):

        """
        Checks the ORIGINAL uploaded file (raw bytes, not the rendered
        image) for multiple '%%EOF' markers.

        A PDF's structure normally contains exactly one '%%EOF'. Every
        additional one means the file was saved again after its first
        save (a PDF "incremental update"). This is a strong structural
        signal that the document is not in its original, single-save
        state -- content, form fields, or signatures may have been
        added or altered after creation.

        This is a byte-level check, independent of the ELA/pixel
        analysis, so it only applies to PDFs (not JPG/PNG uploads).
        """

        result = {
            "applicable": False,
            "eof_count": 0,
            "eof_offsets": [],
            "flagged": False,
            "error": None
        }

        if not file_path.lower().endswith(".pdf"):
            return result

        result["applicable"] = True

        try:

            with open(file_path, "rb") as f:
                data = f.read()

            offsets = [
                match.start()
                for match in
                re.finditer(rb"%%EOF", data)
            ]

            result["eof_count"] = len(offsets)
            result["eof_offsets"] = offsets
            result["flagged"] = len(offsets) > 1

            if len(offsets) == 0:
                result["error"] = (
                    "No '%%EOF' marker found -- file may not be a "
                    "valid or complete PDF."
                )

        except Exception as e:

            result["error"] = str(e)

        return result


    # ============================================================
    # ELA
    # ============================================================

    def _run_ela(self, original):

        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            self.ela_quality
        ]

        success, encoded_img = cv2.imencode(
            ".jpg",
            original,
            encode_param
        )

        if not success:

            raise ValueError(
                "Failed to encode image for ELA."
            )

        resaved = cv2.imdecode(
            encoded_img,
            1
        )

        ela_img = cv2.absdiff(
            original,
            resaved
        )

        ela_img = cv2.multiply(
            ela_img,
            np.array([
                self.ela_scale,
                self.ela_scale,
                self.ela_scale
            ], dtype=np.float32)
        )

        return ela_img


    # ============================================================
    # TEXTURE / CONTENT CHECK
    # ============================================================

    def _check_grid_variance(self, gray):

        h, w = gray.shape

        active_blocks = 0
        total_blocks = 0

        for y in range(
            0,
            h,
            self.block_size
        ):

            for x in range(
                0,
                w,
                self.block_size
            ):

                block = gray[
                    y:y + self.block_size,
                    x:x + self.block_size
                ]

                if (
                    block.shape[0] != self.block_size
                    or
                    block.shape[1] != self.block_size
                ):

                    continue

                total_blocks += 1

                _, std_dev = cv2.meanStdDev(
                    block
                )

                if (
                    std_dev[0][0]
                    >
                    self.variance_threshold
                ):

                    active_blocks += 1

        ratio = (
            active_blocks / total_blocks
            if total_blocks > 0
            else 0
        )

        passed = (
            ratio >= self.min_content_ratio
        )

        return passed, ratio


    # ============================================================
    # EDGE DENSITY
    # ============================================================

    def _check_edge_density(self, gray):

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_pixels = cv2.countNonZero(
            edges
        )

        ratio = (
            edge_pixels / edges.size
        )

        passed = (
            ratio >= self.min_edge_ratio
        )

        return passed, ratio


    # ============================================================
    # LOAD ALL DOCUMENT PAGES
    # ============================================================

    def _load_document(self, file_path):

        """
        Returns a list of pages.

        PDF:
            Every page is rendered.

        JPG/PNG:
            One page is returned.
        """

        pages = []

        # --------------------------------------------------------
        # PDF
        # --------------------------------------------------------

        if file_path.lower().endswith(".pdf"):

            poppler_sys_path = (
                r"C:\Program Files\poppler"
                r"\poppler-26.02.0"
                r"\Library\bin"
            )

            rendered_pages = convert_from_path(
                file_path,
                dpi=200,
                poppler_path=poppler_sys_path
            )

            if not rendered_pages:

                raise ValueError(
                    "PDF appears to be empty or corrupted."
                )

            for page_number, page in enumerate(
                rendered_pages,
                start=1
            ):

                color_img = cv2.cvtColor(
                    np.array(page),
                    cv2.COLOR_RGB2BGR
                )

                gray_img = cv2.cvtColor(
                    color_img,
                    cv2.COLOR_BGR2GRAY
                )

                pages.append({
                    "page_number": page_number,
                    "color": color_img,
                    "gray": gray_img
                })

        # --------------------------------------------------------
        # Image
        # --------------------------------------------------------

        else:

            color_img = cv2.imread(
                file_path
            )

            if color_img is None:

                raise ValueError(
                    "Unable to read uploaded image."
                )

            gray_img = cv2.cvtColor(
                color_img,
                cv2.COLOR_BGR2GRAY
            )

            pages.append({
                "page_number": 1,
                "color": color_img,
                "gray": gray_img
            })

        return pages


    # ============================================================
    # ANALYZE SINGLE PAGE
    # ============================================================

    def _analyze_page(
        self,
        page_number,
        color_img,
        gray_img
    ):

        # --------------------------------------------------------
        # ELA
        # --------------------------------------------------------

        ela_matrix = self._run_ela(
            color_img
        )

        # --------------------------------------------------------
        # Texture
        # --------------------------------------------------------

        variance_passed, variance_ratio = (
            self._check_grid_variance(
                gray_img
            )
        )

        # --------------------------------------------------------
        # Edge density
        # --------------------------------------------------------

        edges_passed, edge_ratio = (
            self._check_edge_density(
                gray_img
            )
        )

        # --------------------------------------------------------
        # Blank detection
        # --------------------------------------------------------

        is_blank = not (
            variance_passed
            or
            edges_passed
        )

        # --------------------------------------------------------
        # ELA statistics
        # --------------------------------------------------------

        ela_gray = cv2.cvtColor(
            ela_matrix,
            cv2.COLOR_BGR2GRAY
        )

        ela_mean = float(
            np.mean(ela_gray)
        )

        ela_std = float(
            np.std(ela_gray)
        )

        # --------------------------------------------------------
        # ELA anomaly
        # --------------------------------------------------------

        possible_manipulation = (
            ela_mean > self.ela_mean_threshold
            or
            ela_std > self.ela_std_threshold
        )

        # --------------------------------------------------------
        # Page status
        # --------------------------------------------------------

        if is_blank:

            status = (
                "REJECTED_BLANK_OR_CORRUPT"
            )

        elif possible_manipulation:

            status = (
                "POSSIBLE_MANIPULATION_DETECTED"
            )

        else:

            status = (
                "STRUCTURE_VERIFIED"
            )

        return {

            "page_number": page_number,

            "status": status,

            "is_blank": is_blank,

            "possible_forgery": (
                possible_manipulation
            ),

            "metrics": {

                "texture_content_density":
                    round(
                        variance_ratio,
                        4
                    ),

                "text_edge_density":
                    round(
                        edge_ratio,
                        4
                    ),

                "ela_mean":
                    round(
                        ela_mean,
                        4
                    ),

                "ela_standard_deviation":
                    round(
                        ela_std,
                        4
                    )
            },

            "visuals": {

                "ela_processed_array":
                    ela_matrix,

                "original_array":
                    color_img
            }
        }


    # ============================================================
    # MAIN DOCUMENT ANALYSIS
    # ============================================================

    def analyze_document(
        self,
        file_path
    ):

        # --------------------------------------------------------
        # Check original file C2PA
        # --------------------------------------------------------

        credentials = (
            self._check_content_credentials(
                file_path
            )
        )

        # --------------------------------------------------------
        # Check original file PDF structure (multiple %%EOF)
        # --------------------------------------------------------

        pdf_structure = (
            self._check_pdf_eof_markers(
                file_path
            )
        )

        # --------------------------------------------------------
        # Load ALL pages
        # --------------------------------------------------------

        pages = self._load_document(
            file_path
        )

        page_results = []

        # --------------------------------------------------------
        # Analyze every page
        # --------------------------------------------------------

        for page in pages:

            page_result = self._analyze_page(

                page["page_number"],

                page["color"],

                page["gray"]

            )

            page_results.append(
                page_result
            )

        # ========================================================
        # DOCUMENT LEVEL AGGREGATION
        # ========================================================

        total_pages = len(
            page_results
        )

        blank_pages = [
            p for p in page_results
            if p["is_blank"]
        ]

        forged_pages = [
            p for p in page_results
            if p["possible_forgery"]
        ]

        # --------------------------------------------------------
        # Document status
        # --------------------------------------------------------

        if len(blank_pages) == total_pages:

            status = (
                "REJECTED_BLANK_OR_CORRUPT"
            )

        elif credentials["ai_generated"]:

            status = (
                "AI_GENERATED_CREDENTIAL_DETECTED"
            )

        elif credentials["ai_edited"]:

            status = (
                "AI_EDITED_CREDENTIAL_DETECTED"
            )

        elif pdf_structure["flagged"]:

            status = (
                "POSSIBLE_TAMPERING_MULTIPLE_EOF"
            )

        elif len(forged_pages) > 0:

            status = (
                "POSSIBLE_MANIPULATION_DETECTED"
            )

        elif credentials["status"] == "CREDENTIAL_VERIFIED":

            status = (
                "STRUCTURE_VERIFIED_WITH_CREDENTIALS"
            )

        else:

            status = (
                "STRUCTURE_VERIFIED"
            )

        # --------------------------------------------------------
        # Aggregate ELA statistics
        # --------------------------------------------------------

        ela_means = [
            p["metrics"]["ela_mean"]
            for p in page_results
        ]

        ela_stds = [
            p["metrics"]["ela_standard_deviation"]
            for p in page_results
        ]

        # --------------------------------------------------------
        # Return document report
        # --------------------------------------------------------

        return {

            "status": status,

            "total_pages": total_pages,

            "blank_pages": [
                p["page_number"]
                for p in blank_pages
            ],

            "forged_pages": [
                p["page_number"]
                for p in forged_pages
            ],

            "content_credentials":
                credentials,

            "pdf_structure":
                pdf_structure,

            "document_metrics": {

                "average_ela_mean":
                    round(
                        float(
                            np.mean(ela_means)
                        ),
                        4
                    ),

                "maximum_ela_mean":
                    round(
                        float(
                            np.max(ela_means)
                        ),
                        4
                    ),

                "average_ela_std":
                    round(
                        float(
                            np.mean(ela_stds)
                        ),
                        4
                    ),

                "maximum_ela_std":
                    round(
                        float(
                            np.max(ela_stds)
                        ),
                        4
                    )
            },

            "pages":
                page_results
        }


# ================================================================
# STREAMLIT APPLICATION
# ================================================================

st.set_page_config(
    page_title="Document Validator",
    page_icon="📄",
    layout="wide"
)

st.title(
    "📄 Multi-Page Document Validator"
)

st.write(
    "Upload a PDF, JPG, or PNG document to check "
    "every page for blank/corrupt content, "
    "C2PA AI provenance, PDF structural tampering, "
    "and possible image manipulation."
)


# ================================================================
# FILE UPLOAD
# ================================================================

uploaded_file = st.file_uploader(
    "Upload document",
    type=[
        "pdf",
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # ------------------------------------------------------------
    # Save uploaded file temporarily
    # ------------------------------------------------------------

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name

    try:

        # --------------------------------------------------------
        # Run validation
        # --------------------------------------------------------

        validator = DocumentValidator()

        with st.spinner(
            "Analyzing all pages..."
        ):

            result = validator.analyze_document(
                temp_path
            )

        # ========================================================
        # OVERALL STATUS
        # ========================================================

        st.subheader(
            "Validation Result"
        )

        status = result["status"]

        if status == (
            "REJECTED_BLANK_OR_CORRUPT"
        ):

            st.error(
                "❌ BLANK / CORRUPT DOCUMENT"
            )

        elif status == (
            "AI_GENERATED_CREDENTIAL_DETECTED"
        ):

            st.error(
                "🤖 AI-GENERATED CONTENT CREDENTIAL DETECTED"
            )

        elif status == (
            "AI_EDITED_CREDENTIAL_DETECTED"
        ):

            st.warning(
                "🤖 AI-EDITED CONTENT CREDENTIAL DETECTED"
            )

        elif status == (
            "POSSIBLE_TAMPERING_MULTIPLE_EOF"
        ):

            st.warning(
                "⚠️ MULTIPLE %%EOF MARKERS DETECTED "
                "-- PDF WAS SAVED/MODIFIED MORE THAN ONCE"
            )

        elif status == (
            "POSSIBLE_MANIPULATION_DETECTED"
        ):

            st.warning(
                "⚠️ POSSIBLE IMAGE MANIPULATION DETECTED"
            )

        elif status == (
            "STRUCTURE_VERIFIED_WITH_CREDENTIALS"
        ):

            st.success(
                "✅ STRUCTURE VERIFIED + CONTENT CREDENTIALS FOUND"
            )

        else:

            st.success(
                "✅ DOCUMENT STRUCTURE VERIFIED"
            )


        # ========================================================
        # DOCUMENT SUMMARY
        # ========================================================

        st.subheader(
            "Document Summary"
        )

        pdf_structure = result["pdf_structure"]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Total Pages",
                result["total_pages"]
            )

        with col2:

            st.metric(
                "Pages with ELA Anomaly",
                len(
                    result["forged_pages"]
                )
            )

        with col3:

            st.metric(
                "Blank / Corrupt Pages",
                len(
                    result["blank_pages"]
                )
            )

        with col4:

            st.metric(
                "Pages Analyzed",
                result["total_pages"]
            )

        with col5:

            if pdf_structure["applicable"]:

                st.metric(
                    "PDF %%EOF Markers",
                    pdf_structure["eof_count"],
                    delta=(
                        "Tampered"
                        if pdf_structure["flagged"]
                        else "Clean"
                    ),
                    delta_color=(
                        "inverse"
                        if pdf_structure["flagged"]
                        else "normal"
                    )
                )

            else:

                st.metric(
                    "PDF %%EOF Markers",
                    "N/A"
                )


        # ========================================================
        # PAGE-BY-PAGE RESULTS
        # ========================================================

        st.subheader(
            "📊 Page-by-Page Analysis"
        )

        table_data = []

        for page in result["pages"]:

            if page["is_blank"]:

                page_status = (
                    "❌ Blank / Corrupt"
                )

            elif page["possible_forgery"]:

                page_status = (
                    "⚠️ Possible Manipulation"
                )

            else:

                page_status = "✅ No ELA Anomaly"

            table_data.append({

                "Page":
                    page["page_number"],

                "Status":
                    page_status,

                "ELA Mean":
                    page["metrics"]["ela_mean"],

                "ELA Std Dev":
                    page["metrics"][
                        "ela_standard_deviation"
                    ],

                "Texture Density":
                    page["metrics"][
                        "texture_content_density"
                    ],

                "Edge Density":
                    page["metrics"][
                        "text_edge_density"
                    ]
            })

        st.dataframe(
            table_data,
            width="stretch"
        )


        # ========================================================
        # ANOMALOUS PAGES
        # ========================================================

        if len(
            result["forged_pages"]
        ) > 0:

            st.subheader(
                "⚠️ Pages Requiring Review"
            )

            st.write(
                "The following pages exceeded the "
                "configured ELA thresholds:"
            )

            st.write(
                ", ".join(
                    f"Page {p}"
                    for p in result[
                        "forged_pages"
                    ]
                )
            )


        # ========================================================
        # DOCUMENT METRICS
        # ========================================================

        st.subheader(
            "📈 Document ELA Metrics"
        )

        metrics = result[
            "document_metrics"
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Average ELA Mean",
                f"{metrics['average_ela_mean']:.2f}"
            )

        with col2:

            st.metric(
                "Maximum ELA Mean",
                f"{metrics['maximum_ela_mean']:.2f}"
            )

        with col3:

            st.metric(
                "Average ELA Std Dev",
                f"{metrics['average_ela_std']:.2f}"
            )

        with col4:

            st.metric(
                "Maximum ELA Std Dev",
                f"{metrics['maximum_ela_std']:.2f}"
            )


        # ========================================================
        # PDF STRUCTURE / %%EOF DETAILS
        # ========================================================

        if pdf_structure["applicable"]:

            st.subheader(
                "🧾 PDF Structure Check"
            )

            if pdf_structure["flagged"]:

                st.warning(
                    f"Found {pdf_structure['eof_count']} "
                    "'%%EOF' markers at byte offsets: "
                    + ", ".join(
                        str(o)
                        for o in pdf_structure["eof_offsets"]
                    )
                    + ". This means the PDF was saved more "
                    "than once after its initial creation "
                    "(a PDF 'incremental update'). This is "
                    "not proof of forgery by itself -- "
                    "legitimate tools (form fills, digital "
                    "signatures) also do this -- but it means "
                    "the file is not in its original, "
                    "single-save state and should be reviewed."
                )

            elif pdf_structure["eof_count"] == 1:

                st.success(
                    "Single '%%EOF' marker found -- no "
                    "incremental updates detected."
                )

            else:

                st.error(
                    pdf_structure["error"]
                    or
                    "No '%%EOF' marker found -- file may not "
                    "be a valid or complete PDF."
                )


        # ========================================================
        # CONTENT CREDENTIALS
        # ========================================================

        st.subheader(
            "🔐 Content Credentials"
        )

        credentials = result[
            "content_credentials"
        ]

        st.json(
            credentials
        )


        # ========================================================
        # PAGE VISUALIZATION
        # ========================================================

        st.subheader(
            "📄 Page Analysis"
        )

        for page in result["pages"]:

            page_number = page[
                "page_number"
            ]

            possible_forgery = page[
                "possible_forgery"
            ]

            if possible_forgery:

                label = (
                    f"⚠️ Page {page_number} "
                    f"- Possible Manipulation"
                )

            elif page["is_blank"]:

                label = (
                    f"❌ Page {page_number} "
                    f"- Blank / Corrupt"
                )

            else:

                label = (
                    f"✅ Page {page_number}"
                )

            with st.expander(
                label
            ):

                col1, col2 = st.columns(2)

                # ------------------------------------------------
                # Original
                # ------------------------------------------------

                with col1:

                    st.markdown(
                        "### Original Page"
                    )

                    original = cv2.cvtColor(
                        page[
                            "visuals"
                        ][
                            "original_array"
                        ],
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        original,
                        caption=(
                            f"Page {page_number}"
                        ),
                        use_container_width=True
                    )

                # ------------------------------------------------
                # ELA
                # ------------------------------------------------

                with col2:

                    st.markdown(
                        "### ELA Analysis"
                    )

                    ela = page[
                        "visuals"
                    ][
                        "ela_processed_array"
                    ]

                    ela_rgb = cv2.cvtColor(
                        ela,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        ela_rgb,
                        caption=(
                            f"ELA - Page {page_number}"
                        ),
                        use_container_width=True
                    )

                # ------------------------------------------------
                # Page metrics
                # ------------------------------------------------

                page_metrics = page[
                    "metrics"
                ]

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "ELA Mean",
                        f"{page_metrics['ela_mean']:.2f}"
                    )

                with c2:

                    st.metric(
                        "ELA Std Dev",
                        f"{page_metrics['ela_standard_deviation']:.2f}"
                    )

                with c3:

                    st.metric(
                        "Texture Density",
                        f"{page_metrics['texture_content_density']:.2%}"
                    )

                with c4:

                    st.metric(
                        "Edge Density",
                        f"{page_metrics['text_edge_density']:.2%}"
                    )


        # ========================================================
        # INTERPRETATION
        # ========================================================

        st.subheader(
            "Interpretation"
        )

        if credentials[
            "ai_generated"
        ]:

            st.error(
                "The original document contains C2PA "
                "provenance indicating trained "
                "algorithmic / AI-generated media."
            )

        elif credentials[
            "ai_edited"
        ]:

            st.warning(
                "The C2PA provenance indicates that "
                "trained algorithmic media was used "
                "in the content."
            )

        elif not credentials[
            "has_credentials"
        ]:

            st.info(
                "No C2PA Content Credentials were found. "
                "This does NOT prove that the document "
                "was created by a human."
            )


        if pdf_structure["flagged"]:

            st.warning(
                f"This PDF contains {pdf_structure['eof_count']} "
                "'%%EOF' markers, meaning it was re-saved "
                "after its initial creation. Combine this "
                "with the ELA results and content credentials "
                "above before concluding the document was "
                "tampered with."
            )


        if result[
            "forged_pages"
        ]:

            st.warning(
                "ELA produced anomaly indicators on "
                f"{len(result['forged_pages'])} page(s). "
                "This is not proof of forgery. "
                "The flagged pages should be reviewed "
                "alongside the original document and "
                "other forensic evidence."
            )

        else:

            st.success(
                "No configured ELA anomaly was detected "
                "on any analyzed page."
            )


        if result[
            "blank_pages"
        ]:

            st.warning(
                "Blank/corrupt pages detected: "
                +
                ", ".join(
                    str(p)
                    for p in result[
                        "blank_pages"
                    ]
                )
            )


    except Exception as e:

        st.error(
            f"Validation failed: {e}"
        )

    finally:

        # --------------------------------------------------------
        # Remove temporary file
        # --------------------------------------------------------

        try:

            os.remove(
                temp_path
            )

        except Exception:

            pass