#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation Reorganization Script for SOLIS
Safely moves and organizes all markdown documentation files.

This script:
1. Creates new directory structure
2. Moves session files from root to docs/sessions/
3. Archives old debugging docs to docs/archive/
4. Verifies all moves were successful
5. Generates summary report

Safe to run - all operations are git-tracked moves (no deletions).
"""

import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')  # Set console to UTF-8


class DocReorganizer:
    """Handles the documentation reorganization process."""

    def __init__(self, dry_run: bool = False):
        """
        Initialize reorganizer.

        Parameters
        ----------
        dry_run : bool
            If True, only print what would be done without actually moving files
        """
        self.root = Path(__file__).parent
        self.dry_run = dry_run
        self.moves: List[Tuple[Path, Path]] = []
        self.created_dirs: List[Path] = []
        self.errors: List[str] = []

    def create_directories(self):
        """Create necessary directory structure."""
        dirs_to_create = [
            self.root / 'docs' / 'sessions',
            self.root / 'docs' / 'archive',
        ]

        print("\n" + "="*70)
        print("PHASE 0: Creating Directory Structure")
        print("="*70)

        for dir_path in dirs_to_create:
            if dir_path.exists():
                print(f"[OK] Directory already exists: {dir_path.relative_to(self.root)}")
            else:
                if not self.dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"[OK] Created: {dir_path.relative_to(self.root)}")
                else:
                    print(f"[DRY RUN] Would create: {dir_path.relative_to(self.root)}")
                self.created_dirs.append(dir_path)

    def define_moves(self):
        """Define all file moves."""

        # PHASE 1: Session files from root → docs/sessions/
        session_files_root = [
            'SESSION_37_FOLLOWUP.md',
            'SESSION_37_SUMMARY.md',
            'SESSION_38_SUMMARY.md',
            'SESSION_38_INCOMPLETE.md',
            'SESSION_38_SOLUTION.md',
            'SESSION_39_PLOT_RESTORATION_PLAN.md',
            'SESSION_39_IMPLEMENTATION_COMPLETE.md',
            'SESSION_39_INCOMPLETE.md',
            'SESSION_STATE_RESTORATION_STATUS.md',
            'gui_modernization_plan.md',
            'GUI_MODERNIZATION_SESSION.md',
            'GUI_modernization_complete.md',
            'PERFORMANCE_ANALYSIS.md',
            'PHASE1_OPTIMIZATIONS.md',
            'PHASE2_OPTIMIZATIONS.md',
            'PHASE3_OPTIMIZATIONS.md',
            'SPLASH_SCREEN_UPGRADE.md',
            'PLOT_REPLAY_DESIGN.md',
            'GITHUB_SETUP_SUMMARY.md',
            'QUICK_START_GITHUB.md',
            'CLEANUP_COMPLETED.md',
        ]

        for filename in session_files_root:
            src = self.root / filename
            dst = self.root / 'docs' / 'sessions' / filename
            if src.exists():
                self.moves.append((src, dst))

        # PHASE 2: Archive old debugging/migration docs from docs/
        archive_files = [
            'docs/CRITICAL_DIFFUSION_BUG.md',
            'docs/CRITICAL_FINDING.md',
            'docs/CRITICAL_GRID_SEARCH_FLAW.md',
            'docs/DATA_PARAMETER_ANALYSIS.md',
            'docs/DIFFUSION_MODEL_CORRECTIONS.md',
            'docs/FINAL_DIAGNOSIS.md',
            'docs/FOUND_THE_BUG.md',
            'docs/HETEROGENEOUS_FIXES_APPLIED.md',
            'docs/HETEROGENEOUS_PARAMETER_ISSUES.md',
            'docs/HETEROGENEOUS_QUICK_GUIDE.md',
            'docs/MEMORY_SESSION_32_CONTINUED.md',
            'docs/MIGRATION_PLAN.md',
            'docs/MIGRATION_SUMMARY.md',
            'docs/NUMBA_ISSUE_RESOLVED.md',
            'docs/MACOS_MENU_FIX.md',
            'docs/SNR_ZERO_NOISE_FIX.md',
            'docs/SESSION_SAVE_LOAD.md',
            'docs/SESSION_31_SUMMARY.md',
            'docs/SESSION_33_FINDINGS.md',
            'docs/SESSION_33_MEMORY_UPDATE.md',
            'docs/SESSION_37_ENTRY.md',
            'docs/SINGLE_STEP_GRID_SEARCH_IMPLEMENTATION.md',
            'docs/QUICK_START.md',
            'docs/READY_TO_PUBLISH.md',
            'docs/MARKDOWN_FILES_CLASSIFICATION.md',
            'docs/CLEANUP_ACTION_PLAN.md',
        ]

        for filepath in archive_files:
            src = self.root / filepath
            filename = Path(filepath).name
            dst = self.root / 'docs' / 'archive' / filename
            if src.exists():
                self.moves.append((src, dst))

    def execute_moves(self):
        """Execute all file moves."""

        print("\n" + "="*70)
        print(f"PHASE 1-2: Moving Files ({'DRY RUN' if self.dry_run else 'EXECUTING'})")
        print("="*70)

        phase1_count = 0
        phase2_count = 0

        for src, dst in self.moves:
            src_rel = src.relative_to(self.root)
            dst_rel = dst.relative_to(self.root)

            # Determine phase
            if dst.parent.name == 'sessions':
                phase = 1
                phase1_count += 1
            else:
                phase = 2
                phase2_count += 1

            if not src.exists():
                error = f"[ERROR] Source not found: {src_rel}"
                print(error)
                self.errors.append(error)
                continue

            if dst.exists():
                error = f"[WARN] Destination already exists: {dst_rel}"
                print(error)
                self.errors.append(error)
                continue

            try:
                if not self.dry_run:
                    shutil.move(str(src), str(dst))
                    print(f"[OK] [Phase {phase}] {src_rel} -> {dst_rel}")
                else:
                    print(f"[DRY RUN] [Phase {phase}] {src_rel} -> {dst_rel}")
            except Exception as e:
                error = f"[ERROR] Error moving {src_rel}: {str(e)}"
                print(error)
                self.errors.append(error)

        print(f"\nPhase 1 (Sessions): {phase1_count} files")
        print(f"Phase 2 (Archive): {phase2_count} files")

    def verify_structure(self):
        """Verify the final directory structure."""

        print("\n" + "="*70)
        print("VERIFICATION: Final Structure")
        print("="*70)

        # Files that should remain in root
        root_essentials = [
            'README.md',
            'CONTRIBUTING.md',
            'BUILD.md',
            'SETUP_INSTRUCTIONS.md',
        ]

        print("\n[OK] Root Directory (should have 5-6 .md files):")
        root_mds = sorted([f.name for f in self.root.glob('*.md')])
        print(f"  Found {len(root_mds)} markdown files:")
        for md in root_mds:
            is_essential = md in root_essentials
            marker = "[OK]" if is_essential else "[WARN]"
            print(f"    {marker} {md}")

        # Check docs/sessions/
        sessions_dir = self.root / 'docs' / 'sessions'
        if sessions_dir.exists():
            sessions = sorted([f.name for f in sessions_dir.glob('*.md')])
            print(f"\n[OK] docs/sessions/ ({len(sessions)} files):")
            for s in sessions[:5]:  # Show first 5
                print(f"    - {s}")
            if len(sessions) > 5:
                print(f"    ... and {len(sessions) - 5} more")

        # Check docs/archive/
        archive_dir = self.root / 'docs' / 'archive'
        if archive_dir.exists():
            archives = sorted([f.name for f in archive_dir.glob('*.md')])
            print(f"\n[OK] docs/archive/ ({len(archives)} files):")
            for a in archives[:5]:  # Show first 5
                print(f"    - {a}")
            if len(archives) > 5:
                print(f"    ... and {len(archives) - 5} more")

        # Check docs/ (main docs)
        docs_dir = self.root / 'docs'
        main_docs = sorted([f.name for f in docs_dir.glob('*.md')])
        print(f"\n[OK] docs/ main ({len(main_docs)} files):")
        for d in main_docs:
            print(f"    - {d}")

    def generate_report(self):
        """Generate summary report."""

        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)

        print(f"\nMode: {'DRY RUN (no changes made)' if self.dry_run else 'EXECUTED'}")
        print(f"Directories created: {len(self.created_dirs)}")
        print(f"Files moved: {len(self.moves)}")
        print(f"Errors: {len(self.errors)}")

        if self.errors:
            print("\n[WARN] Errors encountered:")
            for error in self.errors:
                print(f"  {error}")
        else:
            print("\n[OK] No errors!")

        if not self.dry_run:
            print("\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print("1. Review the changes: git status")
            print("2. Verify imports: python -c 'from core import kinetics_analyzer'")
            print("3. Test application: python show_splash_then_load.py")
            print("4. Commit changes: git add -A && git commit -m 'docs: Reorganize documentation'")
            print("5. Optional: Update SOLIS.spec to selectively bundle docs")

    def run(self):
        """Execute the full reorganization process."""

        print("="*70)
        print("SOLIS Documentation Reorganization")
        print("="*70)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print()

        try:
            self.create_directories()
            self.define_moves()
            self.execute_moves()

            if not self.dry_run:
                self.verify_structure()

            self.generate_report()

            return len(self.errors) == 0

        except Exception as e:
            print(f"\n[FATAL] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""

    # Check if --dry-run flag is present
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    if dry_run:
        print("Running in DRY RUN mode (no changes will be made)\n")
    else:
        print("WARNING: This will move files. Use --dry-run to preview first.\n")
        response = input("Proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Aborted.")
            return 1

    reorganizer = DocReorganizer(dry_run=dry_run)
    success = reorganizer.run()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
