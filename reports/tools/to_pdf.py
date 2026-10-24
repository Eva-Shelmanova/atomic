#!/usr/bin/env python3
"""PDF-версии документов через LibreOffice: оглавление пересчитывается макросом, затем экспорт в PDF.

    python3 to_pdf.py "../Отчёт ….docx" [ещё .docx …]

PDF кладётся рядом с .docx. Макрос (Basic) записывается в отдельный профиль LibreOffice во временной папке.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
PROFILE = Path(tempfile.gettempdir()) / "atomic-lo-profile"
BASIC = r'''
Sub ConvertWithToc(inPath As String, outPdf As String)
    Dim args(0) As New com.sun.star.beans.PropertyValue
    args(0).Name = "Hidden"
    args(0).Value = True
    doc = StarDesktop.loadComponentFromURL(ConvertToURL(inPath), "_blank", 0, args())
    For pass = 1 To 2
        idx = doc.getDocumentIndexes()
        For i = 0 To idx.getCount() - 1
            idx.getByIndex(i).update()
        Next i
        doc.refresh()
    Next pass
    Dim p(0) As New com.sun.star.beans.PropertyValue
    p(0).Name = "FilterName"
    p(0).Value = "writer_pdf_Export"
    doc.storeToURL(ConvertToURL(outPdf), p())
    doc.close(True)
    StarDesktop.terminate()
End Sub
'''


def ensure_profile() -> None:
    std = PROFILE / "user" / "basic" / "Standard"
    if not std.exists():
        subprocess.run([SOFFICE, f"-env:UserInstallation=file://{PROFILE}", "--headless", "--terminate_after_init"],
                       check=False, timeout=180, capture_output=True)
    std.mkdir(parents=True, exist_ok=True)
    (std / "Module1.xba").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">\n'
        '<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" script:language="StarBasic">'
        + escape(BASIC, {'"': "&quot;"}) + "</script:module>\n", encoding="utf-8")
    if not (std / "script.xlb").exists():
        (std / "script.xlb").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE library:library PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "library.dtd">\n'
            '<library:library xmlns:library="http://openoffice.org/2000/library" library:name="Standard" library:readonly="false" library:passwordprotected="false">\n'
            ' <library:element library:name="Module1"/>\n</library:library>\n', encoding="utf-8")


def main(paths: list[str]) -> None:
    ensure_profile()
    for p in paths:
        src = Path(p).resolve()
        pdf = src.with_suffix(".pdf")
        pdf.unlink(missing_ok=True)
        macro = f'macro:///Standard.Module1.ConvertWithToc("{src}","{pdf}")'
        subprocess.run([SOFFICE, f"-env:UserInstallation=file://{PROFILE}", "--headless", "--norestore", macro],
                       check=False, timeout=300, capture_output=True)
        print(("ok   " if pdf.exists() else "FAIL ") + str(pdf))


if __name__ == "__main__":
    main(sys.argv[1:])
