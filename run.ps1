Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       Starting RAGLens Platform (Evaluation & Benchmarks) " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "$PSScriptRoot"
python "$PSScriptRoot\main.py"
