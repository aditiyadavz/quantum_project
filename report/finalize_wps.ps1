# Uses WPS Office to update the contents lists and export a PDF of the report.
# It works only on this report: any other open WPS documents are left alone, and WPS is
# closed only if this script's document was the only one open.
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$docx = Join-Path $dir "NV62010_QML_Breast_Cancer_Report.docx"
$pdf = Join-Path $dir "NV62010_QML_Breast_Cancer_Report.pdf"
function Step($text) { "$(Get-Date -Format HH:mm:ss) $text" }
Step "starting WPS"
$app = New-Object -ComObject KWPS.Application
$app.Visible = $false
$app.DisplayAlerts = 0
"Version: " + $app.Version
$before = $app.Documents.Count
"Other documents open: $before"
$doc = $null
try {
    Step "opening report"
    $doc = $app.Documents.Open($docx)
    Step "setting contents styles"
    foreach ($name in "TOC 1", "TOC 2", "TOC 3") {
        try {
            $style = $doc.Styles.Item($name)
            $style.Font.Name = "Times New Roman"
            $style.Font.Size = 11
            $style.ParagraphFormat.LineSpacingRule = 0
            $style.ParagraphFormat.SpaceAfter = 2
            $style.ParagraphFormat.Alignment = 0
        } catch { Step "style $name not set: $($_.Exception.Message)" }
    }
    Step "updating fields"
    [void]$doc.Fields.Update()
    "Contents lists: " + $doc.TablesOfContents.Count
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    Step "repaginating"
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    Step "saving"
    $doc.Save()
    "Pages: " + $doc.ComputeStatistics(2)
    Step "exporting PDF"
    try { $doc.ExportAsFixedFormat($pdf, 17) }
    catch { Step "ExportAsFixedFormat failed: $($_.Exception.Message); trying SaveAs2"; $doc.SaveAs2($pdf, 17) }
    Step "export done"
} catch {
    Step "FAILED: $($_.Exception.Message)"
} finally {
    if ($doc) { try { $doc.Close(0) } catch { Step "close: $($_.Exception.Message)" } }
    if ($before -eq 0) { try { $app.Quit() } catch { Step "quit: $($_.Exception.Message)" } }
}
Step "finished"
