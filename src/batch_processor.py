"""Batch Audio Analysis Processor - Handles bulk file processing.

Provides parallel and sequential batch processing with progress tracking,
error handling, and comprehensive result aggregation and reporting.

Author: ssanjuanelog
License: MIT
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib

from audio_analyzer_engine import AudioAnalyzerEngine, AnalysisResult


@dataclass
class ProcessingStats:
    """Statistics for batch processing session."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    average_loudness: Optional[float] = None
    loudness_range: Tuple[float, float] = (0.0, 0.0)
    files_compliant: int = 0
    compliance_rate: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)


class BatchAudioProcessor:
    """Process multiple audio files with comprehensive analysis."""
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    def __init__(self, num_workers: int = 4, log_level: str = 'INFO'):
        """Initialize batch processor.
        
        Args:
            num_workers: Number of parallel processing threads
            log_level: Logging level
        """
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        self.engine = AudioAnalyzerEngine()
        self.stats = ProcessingStats()
        self.results: Dict[str, AnalysisResult] = {}
        self.file_hashes: Dict[str, str] = {}
        
        self.logger.info(f"Batch processor initialized (workers: {num_workers})")
    
    def process_directory(self, directory: str, recursive: bool = True,
                         parallel: bool = True) -> ProcessingStats:
        """Process all audio files in directory.
        
        Args:
            directory: Directory path to process
            recursive: Recursively process subdirectories
            parallel: Use parallel processing
            
        Returns:
            Processing statistics
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Collect audio files
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(directory_path.glob(f"{pattern}{ext}"))
        
        self.logger.info(f"Found {len(audio_files)} audio files to process")
        
        if parallel:
            return self._process_parallel(audio_files)
        else:
            return self._process_sequential(audio_files)
    
    def process_file_list(self, file_list: List[str],
                         parallel: bool = True) -> ProcessingStats:
        """Process specified list of audio files.
        
        Args:
            file_list: List of file paths
            parallel: Use parallel processing
            
        Returns:
            Processing statistics
        """
        # Validate files
        valid_files = []
        for filepath in file_list:
            path = Path(filepath)
            if not path.exists():
                self.logger.warning(f"File not found: {filepath}")
                self.stats.errors.append({
                    'file': filepath,
                    'error': 'File not found',
                    'type': 'FileError'
                })
                continue
            
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {filepath}")
                continue
            
            valid_files.append(path)
        
        self.logger.info(f"Processing {len(valid_files)} files")
        
        if parallel:
            return self._process_parallel(valid_files)
        else:
            return self._process_sequential(valid_files)
    
    def _process_sequential(self, files: List[Path]) -> ProcessingStats:
        """Process files sequentially.
        
        Args:
            files: List of file paths
            
        Returns:
            Processing statistics
        """
        self.stats.total_files = len(files)
        start_time = datetime.now()
        
        for i, filepath in enumerate(files):
            try:
                self.logger.info(f"Processing [{i+1}/{len(files)}]: {filepath.name}")
                result = self.engine.analyze_file(str(filepath))
                self.results[str(filepath)] = result
                self.stats.processed_files += 1
                self.stats.total_duration += result.duration
                
                # Check compliance
                if self._is_compliant(result):
                    self.stats.files_compliant += 1
                
            except Exception as e:
                self.logger.error(f"Error processing {filepath}: {e}")
                self.stats.failed_files += 1
                self.stats.errors.append({
                    'file': str(filepath),
                    'error': str(e),
                    'type': type(e).__name__
                })
        
        end_time = datetime.now()
        self._finalize_stats(end_time - start_time)
        return self.stats
    
    def _process_parallel(self, files: List[Path]) -> ProcessingStats:
        """Process files in parallel.
        
        Args:
            files: List of file paths
            
        Returns:
            Processing statistics
        """
        self.stats.total_files = len(files)
        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_single_file, filepath): filepath
                for filepath in files
            }
            
            for i, future in enumerate(as_completed(futures)):
                filepath = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.results[str(filepath)] = result
                        self.stats.processed_files += 1
                        self.stats.total_duration += result.duration
                        
                        if self._is_compliant(result):
                            self.stats.files_compliant += 1
                        
                        self.logger.info(f"Completed [{i+1}/{len(files)}]: {filepath.name}")
                    else:
                        self.stats.failed_files += 1
                except Exception as e:
                    self.logger.error(f"Error processing {filepath}: {e}")
                    self.stats.failed_files += 1
                    self.stats.errors.append({
                        'file': str(filepath),
                        'error': str(e),
                        'type': type(e).__name__
                    })
        
        end_time = datetime.now()
        self._finalize_stats(end_time - start_time)
        return self.stats
    
    def _process_single_file(self, filepath: Path) -> Optional[AnalysisResult]:
        """Process single file (for parallel execution).
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Analysis result or None if error
        """
        try:
            result = self.engine.analyze_file(str(filepath))
            # Compute file hash
            self.file_hashes[str(filepath)] = self._compute_file_hash(filepath)
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing {filepath}: {e}")
            return None
    
    def _is_compliant(self, result: AnalysisResult) -> bool:
        """Check if result meets any professional standard.
        
        Args:
            result: Analysis result
            
        Returns:
            True if compliant with at least one standard
        """
        if result.standards_compliance:
            return any(result.standards_compliance.values())
        return False
    
    def _finalize_stats(self, processing_time):
        """Finalize statistics after processing.
        
        Args:
            processing_time: Total processing duration
        """
        self.stats.processing_time = processing_time.total_seconds()
        self.stats.end_time = datetime.now().isoformat()
        
        if self.results:
            loudness_values = [r.integrated_lufs for r in self.results.values()
                             if r.integrated_lufs > -np.inf]
            if loudness_values:
                self.stats.average_loudness = np.mean(loudness_values)
                self.stats.loudness_range = (min(loudness_values), max(loudness_values))
        
        if self.stats.processed_files > 0:
            self.stats.compliance_rate = (
                self.stats.files_compliant / self.stats.processed_files
            )
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary report of batch processing.
        
        Returns:
            Dictionary with processing summary
        """
        return {
            'statistics': asdict(self.stats),
            'file_count': len(self.results),
            'success_rate': (
                self.stats.processed_files / self.stats.total_files
                if self.stats.total_files > 0 else 0
            ),
            'average_file_duration': (
                self.stats.total_duration / self.stats.processed_files
                if self.stats.processed_files > 0 else 0
            ),
            'processing_efficiency': (
                self.stats.total_duration / self.stats.processing_time
                if self.stats.processing_time > 0 else 0
            )
        }
    
    def export_results(self, output_dir: str, format: str = 'json') -> None:
        """Export all analysis results.
        
        Args:
            output_dir: Output directory for results
            format: Export format ('json', 'csv')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            for filepath, result in self.results.items():
                filename = Path(filepath).stem
                output_file = output_path / f"{filename}_analysis.json"
                with open(output_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Export summary
            summary_file = output_path / "batch_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.get_summary_report(), f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {output_dir}")
    
    def _compute_file_hash(self, filepath: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of file for deduplication.
        
        Args:
            filepath: Path to file
            chunk_size: Chunk size for reading
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()
        self.file_hashes.clear()
        self.stats = ProcessingStats()
        self.logger.info("Results cleared")


if __name__ == "__main__":
    import numpy as np
    processor = BatchAudioProcessor(num_workers=4)
    print("Batch Audio Processor initialized")
    print(f"Supported formats: {processor.SUPPORTED_FORMATS}")
