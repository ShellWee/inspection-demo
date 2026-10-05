param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Namespace
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$assetRoot = (Resolve-Path (Join-Path $projectRoot "..\inspection-demo-hf-assets")).Path
$assetRepo = "$Namespace/inspection-demo-ecore-assets-v1"
$spaceRepo = "$Namespace/inspection-demo-ecore"
$assetVolume = "hf://datasets/$assetRepo`:/mnt/default-assets:ro"

function Invoke-Hf {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$HfArguments
    )

    & hf @HfArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face CLI failed with exit code $LASTEXITCODE`: hf $($HfArguments -join ' ')"
    }
}

Invoke-Hf auth whoami
Invoke-Hf repos create $assetRepo --type dataset --private --exist-ok

# Reserve the requested paid hardware before uploading the 943 MB asset bundle.
# A billing or entitlement failure stops here and leaves at most an empty Dataset repo.
Invoke-Hf repos create $spaceRepo `
    --type space `
    --space-sdk docker `
    --private `
    --flavor l4x1 `
    --volume $assetVolume `
    --exist-ok

Invoke-Hf upload-large-folder $assetRepo $assetRoot `
    --type dataset `
    --private `
    --num-workers 4

Push-Location $projectRoot
try {
    $uploadPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $uploadPython)) {
        throw "Project virtual environment is missing: $uploadPython"
    }
    & $uploadPython `
        (Join-Path $projectRoot "scripts\upload_existing_space.py") `
        --repo-id $spaceRepo `
        --project-root $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Existing Space upload failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Invoke-Hf spaces info $spaceRepo
