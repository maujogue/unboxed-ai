import openpyxl
import csv
import re

XLSX_FILE = "Liste examen UNBOXED finaliseģe v2 (avec mesures).xlsx"
CSV_OUTPUT = "rapports_extraits.csv"

# Ordered list of known sections (order matters for parsing)
KNOWN_SECTIONS = [
    "CLINICAL INFORMATION",
    "ASSAY",
    "NODULE CONTROL",
    "STUDY TECHNIQUE",
    "REPORT",
    "CONCLUSIONS",
]

# Build a regex that splits on section headers.
# A section header is one of the known sections followed by . or :
# We allow them to appear anywhere (start of line, mid-line after spaces, etc.)
section_pattern = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(s) for s in KNOWN_SECTIONS) + r")(?!\w)\s*[\.:]",
    re.IGNORECASE,
)


def format_report(text):
    """Reformat REPORT content: each dash-item gets its own line."""
    if not text:
        return text
    # Any whitespace + dash + whitespace → newline + "- "
    # This handles both inline " - " bullets and already-newlined "\n- " ones.
    # Compound words like "non-target" are safe because they have no surrounding whitespace.
    text = re.sub(r'\s+-\s+', '\n- ', text)
    # Remove extra consecutive blank lines
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def parse_report(text):
    """Split a report cell into a dict of {section_name: content}."""
    if not text:
        return {s: "" for s in KNOWN_SECTIONS}

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Find all section positions
    matches = list(section_pattern.finditer(text))

    result = {s: "" for s in KNOWN_SECTIONS}
    found_sections = set()

    if not matches:
        # No section found — put everything in CLINICAL INFORMATION as fallback
        result["CLINICAL INFORMATION"] = text.strip()
        result["ASSAY"] = "yes" if re.search(r'(?<!\w)ASSAY(?!\w)', text, re.IGNORECASE) else "no"
        result["NODULE CONTROL"] = "yes" if re.search(r'(?<!\w)NODULE\s+CONTROL(?!\w)', text, re.IGNORECASE) else "no"
        return result

    for i, match in enumerate(matches):
        section_name = match.group(1).upper()
        # Normalise to our canonical name
        canonical = next((s for s in KNOWN_SECTIONS if s == section_name), section_name)
        found_sections.add(canonical)

        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        # If the same section appears twice, append
        if result.get(canonical):
            result[canonical] += " " + content
        else:
            result[canonical] = content

    # Post-processing per section
    # ASSAY and NODULE CONTROL: "yes" if the keyword appears anywhere in the raw text
    result["ASSAY"] = "yes" if re.search(r'(?<!\w)ASSAY(?!\w)', text, re.IGNORECASE) else "no"
    result["NODULE CONTROL"] = "yes" if re.search(r'(?<!\w)NODULE\s+CONTROL(?!\w)', text, re.IGNORECASE) else "no"
    result["REPORT"] = format_report(result["REPORT"])

    return result


def main():
    wb = openpyxl.load_workbook(XLSX_FILE, read_only=True)
    ws = wb.active

    rows = ws.iter_rows(values_only=True)
    headers = next(rows)

    patient_col = headers.index("PatientID")
    acc_col = headers.index("AccessionNumber")
    report_col = headers.index("Clinical information data (Pseudo reports)")

    csv_columns = ["PatientID", "AccessionNumber"] + KNOWN_SECTIONS

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for row in rows:
            report_text = row[report_col]
            parsed = parse_report(report_text)

            out_row = {
                "PatientID": row[patient_col],
                "AccessionNumber": row[acc_col],
            }
            out_row.update(parsed)
            writer.writerow(out_row)

    print(f"Done. Output written to {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
