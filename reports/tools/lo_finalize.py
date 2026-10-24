"""Обновление оглавления и выгрузка PDF через LibreOffice (UNO).

    /Applications/LibreOffice.app/Contents/Resources/python lo_finalize.py in.docx out.pdf [out_toc.docx]
"""
import os
import subprocess
import sys
import time

import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


def main(src, pdf, docx_out=None):
    soffice = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    proc = subprocess.Popen([soffice, "--headless", "--invisible", "--norestore",
                             "--accept=socket,host=127.0.0.1,port=2002;urp;"])
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        for _ in range(60):
            try:
                ctx = resolver.resolve("uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(1)
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(src)), "_blank", 0, (prop("Hidden", True),))
        idx = doc.getDocumentIndexes()
        for i in range(idx.getCount()):
            idx.getByIndex(i).update()
        doc.refresh()
        idx = doc.getDocumentIndexes()
        for i in range(idx.getCount()):
            idx.getByIndex(i).update()
        doc.storeToURL(uno.systemPathToFileUrl(os.path.abspath(pdf)), (prop("FilterName", "writer_pdf_Export"),))
        if docx_out:
            doc.storeToURL(uno.systemPathToFileUrl(os.path.abspath(docx_out)), (prop("FilterName", "MS Word 2007 XML"),))
        pages = doc.getCurrentController().getPropertyValue("PageCount") if hasattr(doc.getCurrentController(), "getPropertyValue") else None
        doc.close(True)
        print("ok", pdf, "indexes:", idx.getCount(), "pages:", pages)
    finally:
        try:
            desktop.terminate()
        except Exception:
            pass
        proc.terminate()


if __name__ == "__main__":
    main(*sys.argv[1:])
