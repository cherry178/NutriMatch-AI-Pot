"""
Package Project Script
======================
Archives the complete, standalone project into a portable ZIP file for easy sharing with teammates.
Automatically packages:
- All source code (src/)
- Pretrained models (models_saved/)
- Processed dataset files (data/processed/ & data/recbole_atomic/)
- Web dashboards and templates (src/frontend/static/, data/stitch_template.html)
- One-click launcher scripts (START_SERVER.bat, START_SERVER.sh, run_demo.bat)
- Comprehensive reports and documentation (README.md, PROJECT_REPORT.md, requirements.txt)

Automatically excludes:
- Python cache (__pycache__, *.pyc)
- Temp and log files (*.log, *.tmp)
- Existing zip archives
- Virtual environments (.venv, venv, miniforge3)
- Raw unprocessed bulk dumps that are redundant with processed parquets
"""

import os
import zipfile

def package_project(output_filename="Adaptive_Recipe_Recommender_Project.zip"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(project_root, output_filename)

    exclude_dirs = {
        '.pytest_cache', '__pycache__', '.git', '.vscode', '.idea', 'miniforge3', 'venv', '.venv'
    }
    exclude_extensions = {'.pyc', '.pyo', '.pyd', '.log', '.tmp', '.zip'}

    print(f"Packaging project from: {project_root}")
    print(f"Target archive: {output_path}")

    # Remove existing zip if present
    if os.path.exists(output_path):
        os.remove(output_path)

    file_count = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, dirs, files in os.walk(project_root):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_extensions:
                    continue

                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, project_root)

                # Skip raw bulk dumps if processed files already exist to keep archive compact & fast to transfer
                if rel_file.startswith(os.path.join("data", "raw")) and (file.endswith(".pkl") or file.endswith(".csv")):
                    continue

                zipf.write(abs_file, rel_file)
                file_count += 1

    archive_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully packaged {file_count} files into {output_filename} ({archive_size_mb:.2f} MB)")
    return output_path

if __name__ == "__main__":
    package_project()
